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
import time

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pandas

import onnx_light.onnx as onnxl

N_INIT = 20
DIM = 256 if os.environ.get("UNITTEST_GOING") == "1" else 2048
N_RUNS = 5


def make_model(n_init: int = N_INIT, dim: int = DIM) -> onnx.ModelProto:
    """Returns a synthetic ONNX model with large initializers."""
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


def measure(name: str, fn, n: int = N_RUNS) -> dict:
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
    """Formats and prints benchmark statistics in milliseconds."""
    print(f"{name:<35} avg={stats['avg'] * 1e3:.1f} ms median={stats['median'] * 1e3:.1f} ms")


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

onnx_external_path = os.path.join(out_dir, "out_onnx_ext.onnx")
onnx_external_location = "out_onnx_ext.data"
results.append(
    measure(
        "save/2filex1/onnx",
        lambda: onnx.save_model(
            onnx_model,
            onnx_external_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=onnx_external_location,
        ),
    )
)
print_stats(results[-1]["name"], results[-1])

onnx_light_external_path = os.path.join(out_dir, "out_onnxlight_ext.onnx")
onnx_light_external_data = onnx_light_external_path + ".data"
results.append(
    measure(
        "save/2filex1/onnxlight",
        lambda: onnxl.save(
            onnx_light_model, onnx_light_external_path, location=onnx_light_external_data
        ),
    )
)
print_stats(results[-1]["name"], results[-1])

onnx_light_external_x4_path = os.path.join(out_dir, "out_onnxlight_ext_x4.onnx")
onnx_light_external_x4_data = onnx_light_external_x4_path + ".data"
results.append(
    measure(
        "save/2filex4/onnxlight",
        lambda: onnxl.save(
            onnx_light_model,
            onnx_light_external_x4_path,
            location=onnx_light_external_x4_data,
            parallel=True,
            num_threads=4,
        ),
    )
)
print_stats(results[-1]["name"], results[-1])

# %%
# Results
# -------

df = pandas.DataFrame(results).set_index("name").sort_index()
print(df)

# %%
# Plot
# ----

ax = df[["avg", "median"]].plot.barh(
    title=f"size={size_bytes / 2 ** 20:.2f} MB\nexternal-data save (s)\nlower is better",
    xlabel="seconds",
)
ax.grid(axis="x")
ax.figure.tight_layout()
ax.figure.savefig("plot_save_external_data_time.png")

# %%
# Cleanup
# -------

shutil.rmtree(out_dir, ignore_errors=True)
