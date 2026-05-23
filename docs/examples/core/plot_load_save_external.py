"""
.. _l-example-plot-load-save-external:

Load and save ONNX models with external data
============================================

This example demonstrates how to load and save ONNX models that store tensor weights
in separate *external data* files.  The approach allows the graph structure to
be inspected independently of the (potentially very large) weight payload and
enables memory-mapping the weight file directly when the model is loaded
back through :func:`onnx_light.onnx.load`.

All operations are performed through :mod:`onnx_light.onnx`, which routes all
I/O through C++ without any Python-level tensor iteration.
"""

import os
import shutil

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

import onnx_light.onnx as onnxl


def check_w0_roundtrip(model_proto, expected: np.ndarray, label: str = "") -> None:
    """Verifies that the first initializer matches *expected* after a round-trip.

    Extracts the raw bytes of the first initializer in *model_proto*, interprets
    them as float32, reshapes to match *expected*, and asserts element-wise
    closeness.

    :param model_proto: Loaded :class:`onnxl.ModelProto` to inspect.
    :param expected: Reference numpy array to compare against.
    :param label: Optional suffix appended to the assertion message.
    """
    raw = bytes(model_proto.graph.initializer[0].raw_data)
    loaded = np.frombuffer(raw, dtype=np.float32).reshape(expected.shape)
    suffix = f" ({label})" if label else ""
    assert np.allclose(expected, loaded), f"Round-trip mismatch for W0{suffix}"
    print(f"W0 round-trip{suffix}: OK")


# %%
# Build a tiny synthetic ONNX model
# -----------------------------------
#
# The model has two ``Gemm`` nodes with float32 weight matrices.  All tensors
# are stored as initializers so they end up in the external data file when
# saved with ``location``.

DIM = 64 if os.environ.get("UNITTEST_GOING") == "1" else 256

w0 = np.random.randn(DIM, DIM).astype(np.float32)
w1 = np.random.randn(DIM, DIM).astype(np.float32)

inputs = [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, DIM])]
outputs = [oh.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [None, DIM])]
initializers = [onh.from_array(w0, name="W0"), onh.from_array(w1, name="W1")]
nodes = [
    oh.make_node("Gemm", ["X", "W0"], ["Y0"], transB=1),
    oh.make_node("Gemm", ["Y0", "W1"], ["Y1"], transB=1),
]
graph = oh.make_graph(nodes, "demo_graph", inputs, outputs, initializer=initializers)
onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)

print(f"Number of initializers: {len(onnx_model.graph.initializer)}")
print(f"Model ByteSize: {onnx_model.ByteSize() / 1024:.1f} KB")

# %%
# Save to a single .onnx file first
# -----------------------------------
#
# Write the model to disk using standard ``onnx.save`` so that we can later
# convert it to the two-file layout via :func:`onnxl.load` /
# :func:`onnxl.save`.

out_dir = "temp_plot_load_save_external"
os.makedirs(out_dir, exist_ok=True)

single_file_path = os.path.join(out_dir, "model_single.onnx")
onnx.save(onnx_model, single_file_path)
print(f"Saved single-file model: {single_file_path}")

# %%
# Load with onnx_light
# ---------------------
#
# :func:`onnxl.load` memory-maps the main ``.onnx`` file (and any external
# weights file) and can optionally parse tensors in parallel.

onnxl_model = onnxl.load(single_file_path)
print(f"Loaded model ir_version={onnxl_model.ir_version}")
print(f"Graph name: {onnxl_model.graph.name}")

# %%
# Save with external data (two files)
# -------------------------------------
#
# Passing a ``location`` argument routes all tensor raw-data to a separate
# binary file.  The ``.onnx`` file stores only the graph structure plus a
# small metadata record (offset + length) for each weight tensor.
#
# * ``model_ext.onnx`` – graph structure (kilobytes)
# * ``model_ext.onnx.data`` – raw weight bytes (megabytes)

ext_onnx = os.path.join(out_dir, "model_ext.onnx")
ext_data = ext_onnx + ".data"

onnxl.save(onnxl_model, ext_onnx, location=ext_data)

onnx_size = os.path.getsize(ext_onnx)
data_size = os.path.getsize(ext_data)
print("Saved two-file model:")
print(f"  {ext_onnx!r:<50} {onnx_size / 1024:7.1f} KB  (graph structure)")
print(f"  {ext_data!r:<50} {data_size / 1024:7.1f} KB  (tensor weights)")

# %%
# Load from the two-file layout
# ------------------------------
#
# Pass ``load_external_data=True`` so :func:`onnxl.load` scans the model
# metadata and auto-discovers the external data file from the ``location``
# entry stored in each tensor's ``external_data`` field.

loaded_ext = onnxl.load(ext_onnx, load_external_data=True)
print(f"Loaded two-file model, initializers={len(loaded_ext.graph.initializer)}")

# Verify that the first weight round-trips correctly.
check_w0_roundtrip(loaded_ext, w0)

# %%
# Override the data-file location at load time
# ----------------------------------------------
#
# If the weight file has been moved or renamed, the ``location`` keyword lets
# you override the path stored inside the ``.onnx`` metadata.

loaded_override = onnxl.load(ext_onnx, location=ext_data, load_external_data=True)
print(f"Loaded with explicit location, initializers={len(loaded_override.graph.initializer)}")

# %%
# Save with parallel I/O
# -----------------------
#
# Large models benefit from writing raw-data blocks in parallel.  Pass
# ``num_threads=N > 1 or 0`` to control
# the thread pool. The ``min_block_size`` parameter prevents spawning
# threads for tiny tensors.

ext_par_onnx = os.path.join(out_dir, "model_ext_par.onnx")
ext_par_data = ext_par_onnx + ".data"

onnxl.save(onnxl_model, ext_par_onnx, location=ext_par_data, num_threads=4)
print(f"Saved with parallel I/O: {ext_par_onnx!r}")

# %%
# Split external data across multiple files
# ------------------------------------------
#
# Use ``max_external_file_size`` to cap the size of each external weight file.
# Once the primary file (``model_split.onnx.data``) reaches the limit, a new
# file is opened automatically with the suffix ``.1``, ``.2``, and so on.
# When loading, only the primary location needs to be specified; the loader
# follows the split-file references stored in each tensor's metadata.

ext_split_onnx = os.path.join(out_dir, "model_split.onnx")
ext_split_data = ext_split_onnx + ".data"

# Cap at half the total weight size so that at least two data files are
# produced regardless of the chosen DIM.
total_weight_bytes = (w0.nbytes + w1.nbytes) // 2

onnxl.save(
    onnxl_model,
    ext_split_onnx,
    location=ext_split_data,
    max_external_file_size=total_weight_bytes,
)

split_files = sorted(p for p in os.listdir(out_dir) if p.startswith("model_split.onnx.data"))
print("Files produced by split save:")
for fname in split_files:
    fpath = os.path.join(out_dir, fname)
    print(f"  {fname!r:<40} {os.path.getsize(fpath) / 1024:7.1f} KB")

# Load back – only the primary data file is needed.
loaded_split = onnxl.load(ext_split_onnx, load_external_data=True)
check_w0_roundtrip(loaded_split, w0, "split")

# %%
# Cleanup
# --------

shutil.rmtree(out_dir, ignore_errors=True)
