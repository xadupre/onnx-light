"""
.. _l-example-plot-onnx-time:

Measures loading and saving time for an ONNX model
====================================================

This script builds a small ONNX model and benchmarks the time to load
and save it using :mod:`onnx`, :mod:`onnx_light.onnx`, and
:mod:`onnxruntime`.
It only compares the Python bindings; the model structure is identical
in all cases.

The ``onnx_light.onnx`` implementation does not depend on protobuf and
therefore avoids the overhead of the protobuf serialization layer.
It also supports parallel loading of tensor weights through the
``parallel`` keyword and loading models stored with external data.

File loading in ``onnx_light.onnx`` uses **memory-mapped I/O** (``mmap``
on POSIX, ``CreateFileMapping`` on Windows).  The file is mapped directly
into the virtual address space so that the OS page cache is exposed as
contiguous memory; no extra system-call-per-byte buffering is required.
This makes loading from a file nearly as fast as parsing from an
already-in-memory bytes object.

One key advantage over the ``onnx`` package is zero-copy parsing:
when ``no_copy=True`` is passed to :func:`onnx_light.onnx.load` (or via
:class:`~onnx_light.onnx.ParseOptions`), tensor ``raw_data`` blobs are
**not** copied into new buffers.  Instead each ``TensorProto`` stores a
direct pointer into the serialized bytes.  This eliminates one
``malloc + memcpy`` per tensor initializer and is therefore especially
beneficial for models with many large weight tensors.

.. warning::
   When ``no_copy=True`` is used the caller must keep the original bytes
   object alive for as long as the parsed model is in use.  This
   constraint does not apply to the ``onnx`` package.

For ``onnxruntime``, the session is created with all graph optimizations
disabled (``ORT_DISABLE_ALL``) so that the measurement reflects only the
model loading overhead rather than compilation or fusion costs.

* ``onnx``, ``onnxlight``, ``ort``: use ``onnx``, ``onnx-light``, or ``onnxruntime``
* ``1filex1``: saves in a single file with 1 thread
* ``1filex4``: saves in a single file with 4 threads
* ``2filex1``: saves in a file and another for external data with 1 thread
* ``2filex4``: saves in a file and another for external data with 4 threads
"""

import os
import shutil
import time

import numpy as np
import pandas
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

import onnxruntime as ort

_ort_sess_opts = ort.SessionOptions()
_ort_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

import onnx_light.onnx as onnxl

# %%
# Setup
# ------
#
# Build a small synthetic ONNX model with several ``Gemm`` nodes and large
# initializers so that the load/save times are measurable.

N_INIT = 40
DIM = 256 if os.environ.get("UNITTEST_GOING") == "1" else 2048


def make_model(n_init: int = N_INIT, dim: int = DIM) -> onnx.ModelProto:
    """Returns a synthetic ONNX model with *n_init* Gemm initializers of size *dim*."""
    initializers = []
    nodes = []
    inputs = [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, dim])]

    prev = "X"
    for i in range(n_init):
        weight_name = f"W{i}"
        out_name = f"Y{i}"
        w = np.random.randn(dim, dim).astype(np.float32)
        initializers.append(onh.from_array(w, name=weight_name))
        nodes.append(oh.make_node("Gemm", [prev, weight_name], [out_name], transB=1))
        prev = out_name

    outputs = [oh.make_tensor_value_info(prev, onnx.TensorProto.FLOAT, [None, dim])]
    graph = oh.make_graph(nodes, "bench_graph", inputs, outputs, initializer=initializers)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
    return model


model = make_model()
size_bytes = model.ByteSize()
print(f"Model size: {size_bytes / 2 ** 20:.3f} MB")

# %%
# Write the model to a temporary file.

tmp_dir = "temp_plot_onnx_time"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
onnx_path = os.path.join(tmp_dir, "bench.onnx")
onnx.save(model, onnx_path)
file_size = os.path.getsize(onnx_path)
print(f"File size : {file_size / 2 ** 20:.3f} MB")


# %%
# Benchmark helper.

MIN_TIME_THRESHOLD = 1e-9


def measure(name: str, fn, n: int = 5) -> dict:
    """Runs *fn* *n* times and records timing statistics and CPU utilization."""
    times = []
    cpu_utils = []
    for _ in range(n):
        p0 = time.process_time()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        # Multi-threaded workloads may legitimately report CPU utilization >100%.
        cpu_util = 0.0 if dt <= MIN_TIME_THRESHOLD else (time.process_time() - p0) / dt * 100.0
        cpu_utils.append(cpu_util)
    return {
        "name": name,
        "median": float(np.median(times)),
        "avg": float(np.mean(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "cpu": float(np.mean(cpu_utils)),
    }


def print_stats(name: str, stats: dict) -> None:
    """Prints timing values in milliseconds and CPU utilization."""
    print(
        f"{name:<35} avg={stats['avg'] * 1e3:.1f} ms"
        f" median={stats['median'] * 1e3:.1f} ms"
        f" max={stats['max'] * 1e3:.1f} ms"
        f" cpu={stats['cpu']:.0f}%"
    )


data = []

# %%
# Load benchmarks
# ----------------
#
# Load with ``onnx``.

data.append(measure("load/1filex1/onnx", lambda: onnx.load(onnx_path)))
print_stats("load/1filex1/onnx", data[-1])

# %%
# Load with ``onnx_light.onnx``.

data.append(measure("load/1filex1/onnxlight", lambda: onnxl.load(onnx_path)))
print_stats("load/1filex1/onnxlight", data[-1])

# %%
# Load with ``onnx_light.onnx`` using parallel tensor loading.

data.append(
    measure("load/1filex4/onnxlight", lambda: onnxl.load(onnx_path, parallel=True, num_threads=4))
)
print_stats("load/1filex4/onnxlight", data[-1])
onxl_x4 = onnxl.load(onnx_path, parallel=True, num_threads=4)
onxl = onnxl.load(onnx_path)
onx = onnx.load(onnx_path)

# %%
# Load with ``onnxruntime`` (all optimizations disabled)
# -------------------------------------------------------
# ``InferenceSession`` is created with ``ORT_DISABLE_ALL`` so the
# measurement captures only model loading overhead, not graph optimization.

data.append(
    measure(
        "load/1filex1/ort", lambda: ort.InferenceSession(onnx_path, sess_options=_ort_sess_opts)
    )
)
print_stats("load/1filex1/ort", data[-1])

# %%
# Serialize and Parse benchmarks
# --------------------------------

opts_serial_x4 = onnxl.SerializeOptions()
opts_serial_x4.parallel = True
opts_serial_x4.num_threads = 4


def _serialize_onnx() -> bytes:
    """Serializes the ONNX model to bytes."""
    return onx.SerializeToString()


def _serialize_onnxlight() -> bytes:
    """Serializes the onnx_light model to bytes."""
    return onxl.SerializeToString()


def _serialize_onnxlight_x4() -> bytes:
    """Serializes the onnx_light model in parallel to bytes."""
    return onxl.SerializeToString(opts_serial_x4)


assert len(_serialize_onnx()) > 0
assert len(_serialize_onnxlight()) > 0
assert len(_serialize_onnxlight_x4()) > 0

data.append(measure("serialize/x1/onnx", _serialize_onnx))
print_stats("serialize/x1/onnx", data[-1])
data.append(measure("serialize/x1/onnxlight", _serialize_onnxlight))
print_stats("serialize/x1/onnxlight", data[-1])
data.append(measure("serialize/x4/onnxlight", _serialize_onnxlight_x4))
print_stats("serialize/x4/onnxlight", data[-1])

# %%
# ParseFromString comparison between ``onnx`` and ``onnx_light.onnx``.

serialized_onnx = onx.SerializeToString()
serialized_onnxlight = onxl.SerializeToString()
opts_parse_x4 = onnxl.ParseOptions()
opts_parse_x4.parallel = True
opts_parse_x4.num_threads = 4
opts_parse_nc = onnxl.ParseOptions()
opts_parse_nc.no_copy = True
opts_parse_nc_x4 = onnxl.ParseOptions()
opts_parse_nc_x4.no_copy = True
opts_parse_nc_x4.parallel = True
opts_parse_nc_x4.num_threads = 4


def _parse_onnx() -> onnx.ModelProto:
    """Parses ONNX bytes into a ModelProto."""
    parsed = onnx.ModelProto()
    parsed.ParseFromString(serialized_onnx)
    return parsed


def _parse_onnxlight() -> onnxl.ModelProto:
    """Parses onnx_light bytes into a ModelProto."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight)
    return parsed


def _parse_onnxlight_x4() -> onnxl.ModelProto:
    """Parses onnx_light bytes in parallel into a ModelProto."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight, opts_parse_x4)
    return parsed


def _parse_onnxlight_nc() -> onnxl.ModelProto:
    """Parses onnx_light bytes without copying raw tensor data (zero-copy)."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight, opts_parse_nc)
    return parsed


def _parse_onnxlight_nc_x4() -> onnxl.ModelProto:
    """Parses onnx_light bytes in parallel without copying raw tensor data (zero-copy, 4 t)."""
    parsed = onnxl.ModelProto()
    parsed.ParseFromString(serialized_onnxlight, opts_parse_nc_x4)
    return parsed


parsed_onnx = _parse_onnx()
assert parsed_onnx.ir_version == onx.ir_version
assert len(parsed_onnx.graph.node) == len(onx.graph.node)
parsed_onnxlight = _parse_onnxlight()
assert parsed_onnxlight.ir_version == onxl.ir_version
assert len(parsed_onnxlight.graph.node) == len(onxl.graph.node)
parsed_onnxlight_x4 = _parse_onnxlight_x4()
assert parsed_onnxlight_x4.ir_version == onxl.ir_version
assert len(parsed_onnxlight_x4.graph.node) == len(onxl.graph.node)
parsed_onnxlight_nc = _parse_onnxlight_nc()
assert parsed_onnxlight_nc.ir_version == onxl.ir_version
assert len(parsed_onnxlight_nc.graph.node) == len(onxl.graph.node)
parsed_onnxlight_nc_x4 = _parse_onnxlight_nc_x4()
assert parsed_onnxlight_nc_x4.ir_version == onxl.ir_version
assert len(parsed_onnxlight_nc_x4.graph.node) == len(onxl.graph.node)

data.append(measure("parse/x1/onnx", _parse_onnx))
print_stats("parse/x1/onnx", data[-1])
data.append(measure("parse/x1/onnxlight", _parse_onnxlight))
print_stats("parse/x1/onnxlight", data[-1])
data.append(measure("parse/x4/onnxlight", _parse_onnxlight_x4))
print_stats("parse/x4/onnxlight", data[-1])

# %%
# Parse with zero-copy (``no_copy=True``): raw tensor data is not copied.
# The pointer inside each TensorProto points directly into ``serialized_onnxlight``.
# The bytes object **must** remain alive for as long as the parsed model is used.

data.append(measure("parse/nc/onnxlight", _parse_onnxlight_nc))
print_stats("parse/nc/onnxlight", data[-1])

# %%
# Parse with zero-copy **and** parallel tensor reads (``no_copy=True, parallel=True``).
# Combines the allocation savings of zero-copy with multi-threaded I/O for large models.

data.append(measure("parse/ncx4/onnxlight", _parse_onnxlight_nc_x4))
print_stats("parse/ncx4/onnxlight", data[-1])

# %%
# Save benchmarks
# ----------------
#
# Save once with external data (not benchmarked) using ``onnx_light.onnx`` so
# that the in-memory model is not modified (``ClearExternalData`` restores it
# after the C++ write).
# Absolute paths ensure onnxlight stores only the basename in the ``.onnx``
# metadata, letting both ``onnx.load`` and ``onnxl.load`` resolve the data
# file automatically.

ext_load_onnx = os.path.abspath(os.path.join(tmp_dir, "ext_load.onnx"))
ext_load_data = os.path.abspath(os.path.join(tmp_dir, "ext_load.onnx.data"))
onnxl.save(onxl, ext_load_onnx, location=ext_load_data)

# %%
# Save with ``onnx``.

out_onnx = os.path.join(tmp_dir, "out_onnx.onnx")
data.append(measure("save/1filex1/onnx", lambda: onnx.save(onx, out_onnx)))
print_stats("save/1filex1/onnx", data[-1])

# %%
# Save with ``onnx`` using external data.
# This is the slow path: Python iterates every tensor, creates a numpy
# intermediate, and calls Python I/O for each weight blob.

out_onnx_ext = os.path.join(tmp_dir, "out_onnx_ext.onnx")
out_onnx_ext_location = "out_onnx_ext.data"
data.append(
    measure(
        "save/2filex1/onnx",
        lambda: onnx.save_model(
            onx,
            out_onnx_ext,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=out_onnx_ext_location,
        ),
        n=1,
    )
)
print_stats("save/2filex1/onnx", data[-1])

# %%
# The onnx file is modified to store the external data.
# Let's make sure it is not used again.
onx = None

# %%
# Save with ``onnx_light.onnx``.

out_onnxl = os.path.join(tmp_dir, "out_onnxlight.onnx")
data.append(measure("save/1filex1/onnxlight", lambda: onnxl.save(onxl, out_onnxl)))
print_stats("save/1filex1/onnxlight", data[-1])

# %%
# Save with ``onnx_light.onnx`` parallelized.

out_onnxl_x4 = os.path.join(tmp_dir, "out_onnxlight_x4.onnx")
data.append(
    measure(
        "save/1filex4/onnxlight",
        lambda: onnxl.save(onxl_x4, out_onnxl_x4, parallel=True, num_threads=4),
    )
)
print_stats("save/1filex4/onnxlight", data[-1])

# %%
# Save with ``onnx_light.onnx`` using external data.
# All work is done in C++: ``PopulateExternalData`` attaches metadata once,
# ``SerializeToStream`` routes large ``raw_data`` blobs directly to the
# weights file via ``TwoFilesWriteStream``, and ``ClearExternalData``
# restores the in-memory model.  No numpy arrays are created.

out_ext = os.path.join(tmp_dir, "out_ext.onnx")
out_ext_data = out_ext + ".data"
data.append(
    measure("save/2filex1/onnxlight", lambda: onnxl.save(onxl, out_ext, location=out_ext_data))
)
print_stats("save/2filex1/onnxlight", data[-1])

# %%
# Save with ``onnx_light.onnx`` using external data parallelized.

out_ext_x4 = os.path.join(tmp_dir, "out_ext_x4.onnx")
out_ext_x4_data = out_ext + ".data"
data.append(
    measure(
        "save/2filex4/onnxlight",
        lambda: onnxl.save(
            onxl, out_ext_x4, location=out_ext_x4_data, parallel=True, num_threads=4
        ),
    )
)
print_stats("save/2filex4/onnxlight", data[-1])

# %%
# Load with ``onnx`` using external data.
# Reload the model previously saved with external data using ``onnx.load``.

data.append(
    measure("load/2filex1/onnx", lambda: onnx.load(ext_load_onnx, load_external_data=True))
)
print_stats("load/2filex1/onnx", data[-1])

# %%
# Load with ``onnx_light.onnx`` using external data.
# Reload the same external-data model using ``onnxl.load``.

data.append(
    measure("load/2filex1/onnxlight", lambda: onnxl.load(ext_load_onnx, location=ext_load_data))
)
print_stats("load/2filex1/onnxlight", data[-1])

# %%
# Load with ``onnx_light.onnx`` using external data and parallel tensor loading.
# Combine external-data loading with ``parallel=True`` for maximum throughput.

data.append(
    measure(
        "load/2filex4/onnxlight",
        lambda: onnxl.load(ext_load_onnx, location=ext_load_data, parallel=True, num_threads=4),
    )
)
print_stats("load/2filex4/onnxlight", data[-1])

# %%
# Load with ``onnxruntime`` using external data (all optimizations disabled)
# ---------------------------------------------------------------------------
# Reload the external-data model with ``onnxruntime``, keeping
# ``ORT_DISABLE_ALL`` so only loading overhead is measured.

data.append(
    measure(
        "load/2filex1/ort",
        lambda: ort.InferenceSession(ext_load_onnx, sess_options=_ort_sess_opts),
    )
)
print_stats("load/2filex1/ort", data[-1])

# %%
# Results
# --------

df = pandas.DataFrame(data).set_index("name").sort_index()
print(df)
df = df.sort_index(ascending=False)

# %%
# Plot the results.
# The average, median, and max are shown for each operation.
# Bars are colored by library: blue family for ``onnx``, orange family for
# ``onnx_light``, green family for ``onnxruntime``.  Solid shades represent
# the average; lighter shades the median.
import matplotlib.patches as mpatches

_onnx_avg = "steelblue"
_onnx_med = "lightsteelblue"
_onnx_light_avg = "darkorange"
_onnx_light_med = "moccasin"
_ort_avg = "seagreen"
_ort_med = "lightgreen"

ax = df[["avg", "median"]].plot.barh(
    title=(
        f"onnx vs onnx_light vs ort load/save (s), size={file_size / 2 ** 20:.2f} MB "
        f"(lower is better)\n"
        f"benchmark key: <op>/<files>x<threads>/<lib>\n"
        f"op=load|save|parse|serialize, files=1|2, threads=1|4, lib=onnx|onnxlight|ort"
    ),
    xlabel="seconds",
    legend=False,
    figsize=(12, 6),
)

# Row names use "onnxlight" / "ort" as recorded during benchmarking.
row_names = df.index.tolist()
for container, col in zip(ax.containers, ["avg", "median"]):
    for bar, name in zip(container, row_names):
        if "onnxlight" in name:
            if col == "avg":
                bar.set_facecolor(_onnx_light_avg)
            elif col == "median":
                bar.set_facecolor(_onnx_light_med)
        elif "/ort" in name:
            if col == "avg":
                bar.set_facecolor(_ort_avg)
            elif col == "median":
                bar.set_facecolor(_ort_med)
        else:
            if col == "avg":
                bar.set_facecolor(_onnx_avg)
            elif col == "median":
                bar.set_facecolor(_onnx_med)

first_container = ax.containers[0]
for bar, name in zip(first_container, row_names):
    ax.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2.0,
        f" {df.loc[name, 'cpu']:.0f}%",
        va="center",
        ha="left",
    )

legend_handles = [
    mpatches.Patch(color=_onnx_avg, label="onnx avg"),
    mpatches.Patch(color=_onnx_med, label="onnx median"),
    mpatches.Patch(color=_onnx_light_avg, label="onnx_light avg"),
    mpatches.Patch(color=_onnx_light_med, label="onnx_light median"),
    mpatches.Patch(color=_ort_avg, label="ort avg"),
    mpatches.Patch(color=_ort_med, label="ort median"),
]
ax.legend(handles=legend_handles)
ax.grid(axis="x")
for label in ax.get_yticklabels():
    label.set_horizontalalignment("left")
ax.tick_params(axis="y", pad=120)
ax.figure.tight_layout()
ax.figure.savefig("plot_onnx_time.png")

# %%
# Cleanup
# --------
# Remove all temporary files created during the benchmark.

shutil.rmtree(tmp_dir, ignore_errors=True)
