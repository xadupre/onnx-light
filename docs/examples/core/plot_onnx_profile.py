"""
.. _l-example-plot-onnx-profile:

Profile the C++ parsing and serialization code
===============================================

This script builds a small ONNX model and uses Python's :mod:`cProfile`
module to collect a function-level profile of the C++ code invoked during
parsing (``ParseFromString``) and serialization (``SerializeToString``) by
:mod:`onnx_light.onnx`.

Because :mod:`onnx_light.onnx` exposes its C++ back-end through nanobind,
``cProfile`` records every Python → C++ call boundary as a discrete entry.
The resulting profile therefore shows exactly how the Python layer
orchestrates the C++ routines, and which C++ entry points dominate the wall
time seen from Python.

The example profiles four operations:

* **parse/onnxlight** – ``ModelProto.ParseFromString`` (sequential)
* **parse/onnxlight/x4** – ``ModelProto.ParseFromString`` with
  ``parallel=True, num_threads=4``
* **serialize/onnxlight** – ``ModelProto.SerializeToString`` (sequential)
* **serialize/onnxlight/x4** – ``ModelProto.SerializeToString`` with
  ``parallel=True, num_threads=4``
"""

import cProfile
import io
import os
import pstats
import shutil

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
# We reuse the same construction as in the timing example so that the
# model is large enough for the profiler to capture meaningful call counts.

N_INIT = 40
DIM = 256 if os.environ.get("UNITTEST_GOING") == "1" else 2048


def make_model(n_init: int = N_INIT, dim: int = DIM) -> onnx.ModelProto:
    """Returns a synthetic ONNX model with *n_init* Gemm initializers of size *dim*.

    Args:
        n_init: Number of Gemm initializers to include.
        dim: Square dimension of each weight matrix.
    """
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
# Write the model to a temporary file and load it with ``onnx_light``
# ---------------------------------------------------------------------

tmp_dir = "temp_plot_onnx_profile"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
onnx_path = os.path.join(tmp_dir, "bench.onnx")
onnx.save(model, onnx_path)
file_size = os.path.getsize(onnx_path)
print(f"File size: {file_size / 2 ** 20:.3f} MB")

onxl = onnxl.load(onnx_path)
serialized = onxl.SerializeToString()

# %%
# Profiling helper
# -----------------
#
# ``run_profile`` executes *fn* inside ``cProfile`` for *n* repetitions and
# returns the aggregated :class:`pstats.Stats` object together with a
# :class:`pandas.DataFrame` of the top *top_n* entries sorted by cumulative
# time.

TOP_N = 20
N_REPS = 3 if os.environ.get("UNITTEST_GOING") == "1" else 10


def run_profile(name: str, fn, n: int = N_REPS, top_n: int = TOP_N) -> pandas.DataFrame:
    """Profiles *fn* for *n* repetitions and returns a DataFrame of the top-*top_n* calls.

    Args:
        name: Label used to identify this profile run in output tables and plots.
        fn: Callable to profile; called *n* times inside a single :class:`cProfile.Profile`.
        n: Number of repetitions to accumulate into the profile.
        top_n: Maximum number of functions to retain in the returned DataFrame.

    Returns:
        A DataFrame with columns ``ncalls``, ``tottime``, ``cumtime``, and
        ``function``, restricted to the *top_n* entries by cumulative time.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n):
        fn()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)

    output = stream.getvalue()
    print(f"\n{'=' * 60}")
    print(f"Profile: {name}  ({n} repetitions)")
    print("=" * 60)
    print(output)

    # Build a tidy DataFrame from the raw stats dict for plotting.
    rows = []
    for func_key, (_cc, nc, tt, ct, _callers) in stats.stats.items():
        filename, lineno, funcname = func_key
        rows.append(
            {
                "name": name,
                "function": f"{funcname} ({os.path.basename(filename)}:{lineno})",
                "ncalls": nc,
                "tottime": tt,
                "cumtime": ct,
            }
        )
    df = pandas.DataFrame(rows)
    df = df.sort_values("cumtime", ascending=False).head(top_n).reset_index(drop=True)
    return df


# %%
# Profile ``ParseFromString`` — sequential
# -----------------------------------------
#
# ``ParseFromString`` reads the serialized bytes and reconstructs the full
# in-memory ``ModelProto``.  The profile shows the C++ entry point
# ``ParseFromString`` and all Python helpers it calls.


def _parse() -> onnxl.ModelProto:
    """Parses the serialized bytes and returns an onnx_light ModelProto."""
    m = onnxl.ModelProto()
    m.ParseFromString(serialized)
    return m


df_parse_x1 = run_profile("parse/onnxlight/x1", _parse)
print(df_parse_x1[["function", "ncalls", "tottime", "cumtime"]].to_string(index=False))

# %%
# Profile ``ParseFromString`` — parallel (4 threads)
# ----------------------------------------------------
#
# Passing a :class:`~onnx_light.onnx.ParseOptions` with ``parallel=True``
# dispatches tensor-weight reads to a C++ thread pool.  The profile captures
# how much of the Python-visible time shifts to the thread-pool join.

opts_parse_x4 = onnxl.ParseOptions()
opts_parse_x4.parallel = True
opts_parse_x4.num_threads = 4


def _parse_x4() -> onnxl.ModelProto:
    """Parses the serialized bytes in parallel and returns an onnx_light ModelProto."""
    m = onnxl.ModelProto()
    m.ParseFromString(serialized, opts_parse_x4)
    return m


df_parse_x4 = run_profile("parse/onnxlight/x4", _parse_x4)
print(df_parse_x4[["function", "ncalls", "tottime", "cumtime"]].to_string(index=False))

# %%
# Profile ``SerializeToString`` — sequential
# -------------------------------------------
#
# ``SerializeToString`` converts the in-memory ``ModelProto`` back to bytes.
# The profile shows the dominant C++ routines (size computation + byte
# writing) as seen from Python.

onxl_parsed = _parse()


def _serialize() -> bytes:
    """Serializes an onnx_light ModelProto and returns bytes."""
    return onxl_parsed.SerializeToString()


df_serial_x1 = run_profile("serialize/onnxlight/x1", _serialize)
print(df_serial_x1[["function", "ncalls", "tottime", "cumtime"]].to_string(index=False))

# %%
# Profile ``SerializeToString`` — parallel (4 threads)
# ------------------------------------------------------
#
# Parallel serialization uses a :class:`~onnx_light.onnx.SerializeOptions`
# with ``parallel=True`` so large raw-data blobs are written by a C++ thread
# pool.

opts_serial_x4 = onnxl.SerializeOptions()
opts_serial_x4.parallel = True
opts_serial_x4.num_threads = 4


def _serialize_x4() -> bytes:
    """Serializes an onnx_light ModelProto using 4 threads and returns bytes."""
    return onxl_parsed.SerializeToString(opts_serial_x4)


df_serial_x4 = run_profile("serialize/onnxlight/x4", _serialize_x4)
print(df_serial_x4[["function", "ncalls", "tottime", "cumtime"]].to_string(index=False))

# %%
# Summary table
# --------------
#
# Concatenate the top entries from every profile run into one table.

summary = pandas.concat([df_parse_x1, df_parse_x4, df_serial_x1, df_serial_x4], ignore_index=True)
summary = (
    summary.groupby(["name", "function"])[["ncalls", "tottime", "cumtime"]]
    .sum()
    .reset_index()
    .sort_values(["name", "cumtime"], ascending=[True, False])
)
print(summary.to_string(index=False))

# %%
# Plot the cumulative time per operation
# ---------------------------------------
#
# For each of the four profiled operations we plot the top-5 functions by
# cumulative time.  This gives a visual breakdown of where time is spent
# inside the C++ back-end as seen from Python.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

profiles = [
    ("parse/onnxlight/x1", df_parse_x1),
    ("parse/onnxlight/x4", df_parse_x4),
    ("serialize/onnxlight/x1", df_serial_x1),
    ("serialize/onnxlight/x4", df_serial_x4),
]

for ax, (title, df) in zip(axes, profiles):
    top5 = df.nlargest(5, "cumtime")
    # Shorten the function label for readability.
    labels = [f[:40] + "…" if len(f) > 40 else f for f in top5["function"]]
    ax.barh(labels[::-1], top5["cumtime"].values[::-1], color="steelblue")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cumulative time (s)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x")

fig.suptitle(
    f"C++ profile: top-5 functions by cumulative time\n"
    f"model size={file_size / 2 ** 20:.2f} MB  reps={N_REPS}",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("plot_onnx_profile.png")

# %%
# Cleanup
# --------
# Remove all temporary files created during the profile run.

shutil.rmtree(tmp_dir, ignore_errors=True)
