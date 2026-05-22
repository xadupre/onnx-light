"""
.. _l-example-plot-save-external-data-time:

Profiles ONNX external-data save time
=====================================

This example profiles how long it takes to save a model with external data
using :mod:`onnx` and :mod:`onnx_light.onnx`.

It follows the same benchmark style as :ref:`l-example-plot-onnx-time` but
focuses only on the external-data save scenario.
"""

import os
import shutil
import cProfile
import pstats

import matplotlib.patches as mpatches
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pandas

import onnx_light.onnx as onnxl

N_INIT = 40
DIM = 256 if os.environ.get("UNITTEST_GOING") == "1" else 3072


def make_model(n_init: int = N_INIT, dim: int = DIM) -> onnx.ModelProto:
    """Creates a synthetic ONNX model with large initializers."""
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
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)


def profile_call(name: str, fn, repeat=1) -> dict:
    """Profiles the given callable with cProfile.

    Args:
        name: Benchmark name used in printed output and the result row.
        fn: Callable to execute under cProfile.

    Returns:
        A dictionary with the benchmark name and total profiled time in seconds.
    """
    profiler = cProfile.Profile()
    for _ in range(repeat):
        profiler.runcall(fn)
    profile_stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"\n{name}\n{'-' * len(name)}")
    profile_stats.print_stats(20)
    return {"name": name, "total": float(profile_stats.total_tt)}


def _flush_file(path: str) -> None:
    """Flushes one file descriptor so benchmark timing includes write-back."""
    with open(path, "r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


model = make_model()
size_bytes = model.ByteSize()
print(f"Model size: {size_bytes / 2 ** 20:.3f} MB")

out_dir = "temp_plot_save_external_data_time"
os.makedirs(out_dir, exist_ok=True)

onnx_model = model
onnx_input_path = os.path.join(out_dir, "bench.onnx")
onnx.save(onnx_model, onnx_input_path)
onnx_light_model = onnxl.load(onnx_input_path)

results = []

# ``onnx.save_model(..., save_as_external_data=True)`` mutates the in-memory
# model by replacing ``raw_data`` with external-data metadata. Benchmark it as a
# single-shot operation so the row reflects the full conversion + write cost
# instead of re-saving an already externalized model on later iterations.
# Both saved files are explicitly ``fsync``-ed so this row includes descriptor
# flush/write-back overhead, matching the ``onnxlight`` row.
onnx_external_path = os.path.join(out_dir, "out_onnx_ext.onnx")
onnx_external_location = "out_onnx_ext.data"
onnx_external_data_path = os.path.join(out_dir, onnx_external_location)


def _save_onnx_external_with_flush() -> None:
    onnx.save_model(
        onnx_model,
        onnx_external_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_external_location,
    )
    _flush_file(onnx_external_data_path)
    _flush_file(onnx_external_path)


results.append(profile_call("save/2filex1/onnx", _save_onnx_external_with_flush, repeat=1))
print(f"{results[-1]['name']:<35} total={results[-1]['total'] * 1e3:.1f} ms")

# ``onnx_light.onnx.save`` restores the in-memory model after the write, but we
# keep the benchmark single-shot so the rows stay directly comparable.
onnx_light_external_path = os.path.join(out_dir, "out_onnxlight_ext.onnx")
onnx_light_external_data = onnx_light_external_path + ".data"


def _save_onnxlight_external_with_flush() -> None:
    onnxl.save(onnx_light_model, onnx_light_external_path, location=onnx_light_external_data)
    _flush_file(onnx_light_external_data)
    _flush_file(onnx_light_external_path)


results.append(
    profile_call("save/2filex1/onnxlight", _save_onnxlight_external_with_flush, repeat=1)
)
print(f"{results[-1]['name']:<35} total={results[-1]['total'] * 1e3:.1f} ms")

onnx_light_external_x4_path = os.path.join(out_dir, "out_onnxlight_ext_x4.onnx")
onnx_light_external_x4_data = onnx_light_external_x4_path + ".data"
results.append(
    profile_call(
        "save/2filex4/onnxlight",
        lambda: onnxl.save(
            onnx_light_model,
            onnx_light_external_x4_path,
            location=onnx_light_external_x4_data,
            num_threads=4,
        ),
        repeat=1,
    )
)
print(f"{results[-1]['name']:<35} total={results[-1]['total'] * 1e3:.1f} ms")

# %%
# Results
# -------

df = pandas.DataFrame(results).set_index("name").sort_index()
print(df)

# %%
# Plot
# ----

ax = df[["total"]].plot.barh(
    title=f"size={size_bytes / 2 ** 20:.2f} MB\nexternal-data save (s)\nlower is better",
    xlabel="seconds",
    legend=False,
    figsize=(12, 6),
)

row_names = df.index.tolist()
for container in ax.containers:
    for bar, name in zip(container, row_names):
        bar.set_facecolor("darkorange" if "onnxlight" in name else "steelblue")

ax.legend(
    handles=[
        mpatches.Patch(color="steelblue", label="onnx"),
        mpatches.Patch(color="darkorange", label="onnxlight"),
    ]
)
ax.grid(axis="x")
ax.figure.tight_layout()
ax.figure.savefig("plot_save_external_data_time.png")

# %%
# Cleanup
# -------

shutil.rmtree(out_dir, ignore_errors=True)
