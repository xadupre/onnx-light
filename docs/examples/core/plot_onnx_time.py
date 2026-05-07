"""
.. _l-example-plot-onnx-time:

Measures loading and saving time for an ONNX model
====================================================

This script builds a small ONNX model and benchmarks the time to load
and save it using :mod:`onnx` and :mod:`onnx_light.onnx`.
It only compares the Python bindings; the model structure is identical
in both cases.

The ``onnx_light.onnx`` implementation does not depend on protobuf and
therefore avoids the overhead of the protobuf serialization layer.
It also supports parallel loading of tensor weights through the
``parallel`` keyword and loading models stored with external data.

* ``onnx``, ``onnxlight``: use ``onnx`` or ``onnx-light``
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

import onnx_light.onnx as onnxl

# %%
# Build a small synthetic ONNX model
# ------------------------------------
#
# We create a model with several ``Gemm`` nodes and large initializers so
# that the load/save times are measurable.

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
# Write the model to a temporary file
# -------------------------------------

tmp_dir = "temp_plot_onnx_time"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
onnx_path = os.path.join(tmp_dir, "bench.onnx")
onnx.save(model, onnx_path)
file_size = os.path.getsize(onnx_path)
print(f"File size : {file_size / 2 ** 20:.3f} MB")


# %%
# Benchmark helper
# -----------------


def measure(name: str, fn, n: int = 5) -> dict:
    """Runs *fn* *n* times and records timing statistics."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "name": name,
        "median": float(np.median(times)),
        "avg": float(np.mean(times)),
        "min": float(np.min(times)),
    }


def print_stats(name: str, stats: dict) -> None:
    """Formats and prints the average and median timing values in milliseconds."""
    print(f"{name:<35} avg={stats['avg'] * 1e3:.1f} ms median={stats['median'] * 1e3:.1f} ms")


data = []

# %%
# Load with ``onnx``
# -------------------

data.append(measure("load/1filex1/onnx", lambda: onnx.load(onnx_path)))
print_stats("load/1filex1/onnx", data[-1])

# %%
# Load with ``onnx_light.onnx``
# ------------------------------

data.append(measure("load/1filex1/onnxlight", lambda: onnxl.load(onnx_path)))
print_stats("load/1filex1/onnxlight", data[-1])

# %%
# Load with ``onnx_light.onnx`` using parallel tensor loading
# ------------------------------------------------------------

data.append(
    measure("load/1filex4/onnxlight", lambda: onnxl.load(onnx_path, parallel=True, num_threads=4))
)
print_stats("load/1filex4/onnxlight", data[-1])
onxl_x4 = onnxl.load(onnx_path, parallel=True, num_threads=4)

# %%
# Save with ``onnx``
# -------------------

onx = onnx.load(onnx_path)
out_onnx = os.path.join(tmp_dir, "out_onnx.onnx")
data.append(measure("save/1filex1/onnx", lambda: onnx.save(onx, out_onnx)))
print_stats("save/1filex1/onnx", data[-1])

# %%
# Save with ``onnx`` using external data
# ---------------------------------------

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
    )
)
print_stats("save/2filex1/onnx", data[-1])

# %%
# Save with ``onnx_light.onnx``
# ------------------------------

onxl = onnxl.load(onnx_path)
out_onnxl = os.path.join(tmp_dir, "out_onnxlight.onnx")
data.append(measure("save/1filex1/onnxlight", lambda: onnxl.save(onxl, out_onnxl)))
print_stats("save/1filex1/onnxlight", data[-1])

# %%
# Save with onnx_light.onnx parallelized
# --------------------------------------

out_onnxl_x4 = os.path.join(tmp_dir, "out_onnxlight_x4.onnx")
data.append(
    measure(
        "save/1filex4/onnxlight",
        lambda: onnxl.save(onxl_x4, out_onnxl_x4, parallel=True, num_threads=4),
    )
)
print_stats("save/1filex4/onnxlight", data[-1])

# %%
# Save with ``onnx_light.onnx`` using external data
# ---------------------------------------------------

out_ext = os.path.join(tmp_dir, "out_ext.onnx")
out_ext_data = out_ext + ".data"
data.append(
    measure("save/2filex1/onnxlight", lambda: onnxl.save(onxl, out_ext, location=out_ext_data))
)
print_stats("save/2filex1/onnxlight", data[-1])

# %%
# Save with ``onnx_light.onnx`` using external data parallelized
# --------------------------------------------------------------

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
# Load with ``onnx`` using external data
# ----------------------------------------
# Reload the model previously saved with external data using ``onnx.load``.

out_onnx_ext_data = os.path.join(tmp_dir, out_onnx_ext_location)
data.append(
    measure("load/2filex1/onnx", lambda: onnx.load(out_onnx_ext, load_external_data=True))
)
print(f"load/2filex1/onnx      avg={data[-1]['avg'] * 1e3:.1f} ms")

# %%
# Load with ``onnx_light.onnx`` using external data
# --------------------------------------------------
# Reload the same external-data model using ``onnxl.load``.

data.append(
    measure(
        "load/2filex1/onnxlight", lambda: onnxl.load(out_onnx_ext, location=out_onnx_ext_data)
    )
)
print(f"load/2filex1/onnxlight avg={data[-1]['avg'] * 1e3:.1f} ms")

# %%
# Load with ``onnx_light.onnx`` using external data and parallel tensor loading
# -------------------------------------------------------------------------------
# Combine external-data loading with ``parallel=True`` for maximum throughput.

data.append(
    measure(
        "load/2filex4/onnxlight",
        lambda: onnxl.load(
            out_onnx_ext, location=out_onnx_ext_data, parallel=True, num_threads=4
        ),
    )
)
print(f"load/2filex4/onnxlight avg={data[-1]['avg'] * 1e3:.1f} ms")

# %%
# Results
# --------

df = pandas.DataFrame(data).set_index("name").sort_index()
print(df)
df = df.sort_index(ascending=False)

# %%
# Plot the results.
# Both the average and median are shown for each operation.
# Bars are colored by library: blue family for ``onnx``, orange family for
# ``onnx_light``.  Solid shades represent the average; lighter shades the median.
import matplotlib.patches as mpatches

_onnx_avg = "steelblue"
_onnx_med = "lightsteelblue"
_onnx_light_avg = "darkorange"
_onnx_light_med = "moccasin"

ax = df[["avg", "median"]].plot.barh(
    title=f"size={file_size / 2 ** 20:.2f} MB\nonnx vs onnx_light load/save (s)\nlower is better",
    xlabel="seconds",
    legend=False,
)

# Row names use "onnxlight" (no underscore) as recorded during benchmarking.
row_names = df.index.tolist()
for container, col in zip(ax.containers, ["avg", "median"]):
    for bar, name in zip(container, row_names):
        if "onnxlight" in name:
            bar.set_facecolor(_onnx_light_avg if col == "avg" else _onnx_light_med)
        else:
            bar.set_facecolor(_onnx_avg if col == "avg" else _onnx_med)

ax.legend(
    handles=[
        mpatches.Patch(color=_onnx_avg, label="onnx avg"),
        mpatches.Patch(color=_onnx_med, label="onnx median"),
        mpatches.Patch(color=_onnx_light_avg, label="onnx_light avg"),
        mpatches.Patch(color=_onnx_light_med, label="onnx_light median"),
    ]
)
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
