"""
.. _l-example-plot-save-ort-flatbuffers:

Save an ONNX model in the ORT flatbuffer format and compare sizes
=================================================================

`onnxruntime <https://onnxruntime.ai/>`_ defines a flatbuffer
serialization (``.ort``) of an ONNX model. It is typically used in
size-constrained deployments because it can be memory-mapped directly into
the runtime and avoids a protobuf parsing step.

*onnx-light* exposes the format through
:py:class:`onnx_light.onnx.SerializeFormat`, but the C++ writer for
``ORT_FLATBUFFERS`` is not implemented yet (calls raise ``RuntimeError``).
Until it lands, this example uses :epkg:`onnxruntime` itself to produce
the ``.ort`` file and then compares the on-disk sizes of the two formats.

See :ref:`l-howto-save-ort-flatbuffers` for the short recipe.
"""

import os
import shutil

import numpy as np
import onnxruntime

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh

# %%
# Build a tiny synthetic ONNX model
# ---------------------------------
#
# The graph holds two ``Gemm`` nodes with float32 weight matrices so that
# the saved files have a non-trivial size. ``DIM`` is intentionally small
# when the example runs in the documentation build (``UNITTEST_GOING=1``).

DIM = 64 if os.environ.get("UNITTEST_GOING") == "1" else 256

w0 = np.random.randn(DIM, DIM).astype(np.float32)
w1 = np.random.randn(DIM, DIM).astype(np.float32)

inputs = [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [None, DIM])]
outputs = [oh.make_tensor_value_info("Y1", onnxl.TensorProto.FLOAT, [None, DIM])]
initializers = [onh.from_array(w0, name="W0"), onh.from_array(w1, name="W1")]
nodes = [
    oh.make_node("Gemm", ["X", "W0"], ["Y0"], transB=1),
    oh.make_node("Gemm", ["Y0", "W1"], ["Y1"], transB=1),
]
graph = oh.make_graph(nodes, "demo_graph", inputs, outputs, initializer=initializers)
onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)

# %%
# Save in the standard ONNX protobuf format
# -----------------------------------------

out_dir = "temp_plot_save_ort_flatbuffers"
os.makedirs(out_dir, exist_ok=True)

onnx_path = os.path.join(out_dir, "model.onnx")
onnxl.save(onnx_model, onnx_path)

# %%
# Save in the ORT flatbuffer format via onnxruntime
# -------------------------------------------------
#
# Disable graph optimizations so that the serialized graph stays
# structurally equivalent to the input. Setting
# ``session.save_model_format=ORT`` tells onnxruntime to dump the
# (un)optimized model as a ``.ort`` flatbuffer file at the path given by
# ``optimized_model_filepath``.

ort_path = os.path.join(out_dir, "model.ort")

session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session_options.optimized_model_filepath = ort_path
session_options.add_session_config_entry("session.save_model_format", "ORT")

# Creating the session triggers the optimized-model dump in ORT format.
onnxruntime.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])

# %%
# Compare the on-disk sizes
# -------------------------
#
# The flatbuffer payload is comparable to the protobuf one for this small
# graph. On much larger models the ``.ort`` file is typically a bit bigger
# because it embeds runtime-specific metadata, but it loads faster because
# onnxruntime memory-maps it directly without going through protobuf
# parsing.

onnx_size = os.path.getsize(onnx_path)
ort_size = os.path.getsize(ort_path)

print(f"{'format':<8} {'size (bytes)':>14}  {'ratio vs .onnx':>16}")
print("-" * 42)
print(f"{'.onnx':<8} {onnx_size:>14}  {1.0:>16.3f}")
print(f"{'.ort':<8} {ort_size:>14}  {ort_size / onnx_size:>16.3f}")

# %%
# Cleanup
# -------

shutil.rmtree(out_dir, ignore_errors=True)
