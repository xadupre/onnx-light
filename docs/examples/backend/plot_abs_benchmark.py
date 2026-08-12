"""
.. _l-example-plot-abs-benchmark:

Benchmark Abs: onnxruntime vs onnx-light
========================================

This example compares the built-in :class:`Abs` kernel in ``onnx-light`` with
``onnxruntime`` and :func:`numpy.abs` for ``float32`` vectors ranging from one
hundred to one hundred million elements.

Initialization and execution are reported separately. Constructing an
``onnx-light`` :class:`~onnx_light.onnx.reference.ReferenceEvaluator` is
lightweight, but that does not imply that each inference is faster. The
execution benchmark warms both runtimes, alternates their measurement order,
and reports median durations so the comparison does not confuse startup cost
with steady-state inference cost.

The benchmark also exercises the low-level
:func:`onnx_light.onnx_py._onnxpykernels.runtime.run_model` entry point, which
runs a whole model end-to-end from :class:`Tensor` inputs to :class:`Tensor`
outputs. It is measured two ways: ``run_model (numpy)`` starts and ends with
NumPy arrays (so the timing includes the NumPy <-> :class:`Tensor` conversions),
while ``run_model (Tensor)`` reuses a pre-built input :class:`Tensor` and keeps
the output as a :class:`Tensor`, isolating the raw model-execution cost.
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot
import numpy
import onnxruntime
from onnx_light.onnx import TensorProto, checker, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_py import _onnxpykernels

runtime = _onnxpykernels.runtime


def make_abs_model():
    """Creates a dynamic one-dimensional Abs model."""

    graph = helper.make_graph(
        [helper.make_node("Abs", ["X"], ["Y"])],
        "abs_benchmark",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


def measure(function, repeat: int, warmup: int = 3) -> float:
    """Measures a callable after warm-up iterations and returns its median time."""

    for _ in range(warmup):
        function()
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(numpy.median(timings))


def measure_pair(first, second, repeat: int, warmup: int = 3) -> tuple[float, float]:
    """Measures two callables with alternating execution order."""

    for _ in range(warmup):
        first()
        second()
    timings = ([], [])
    functions = (first, second)
    for iteration in range(repeat):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter()
            functions[index]()
            timings[index].append(time.perf_counter() - start)
    return tuple(float(numpy.median(values)) for values in timings)


# %%
# Create both runtimes
# --------------------
#
# The model is serialized before either timer starts. The setup measurements
# therefore compare runtime construction rather than model generation.

model = make_abs_model()
model_bytes = model.SerializeToString()

start = time.perf_counter()
ort_session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
ort_setup_time = time.perf_counter() - start

start = time.perf_counter()
onnx_light_session = ReferenceEvaluator(model)
onnx_light_setup_time = time.perf_counter() - start

print(f"setup: onnxruntime InferenceSession = {ort_setup_time * 1e3:.2f} ms")
print(f"setup: onnx-light ReferenceEvaluator = {onnx_light_setup_time * 1e3:.2f} ms")


def run_onnx_light(values):
    """Runs the built-in onnx-light Abs kernel."""

    return onnx_light_session.run(None, {"X": values})[0]


def run_onnx_light_run_model_numpy(values):
    """Runs the whole model through the low-level ``runtime.run_model`` API,
    starting and ending with NumPy arrays.

    The input array is wrapped in a :class:`Tensor` named after the graph input
    it feeds (``"X"``), the model is executed end-to-end, and the single declared
    output is reinterpreted back into a ``float32`` NumPy array. The timing
    therefore includes the NumPy <-> :class:`Tensor` conversions on both ends.
    """

    tensor = runtime.tensor_from_numpy(
        "X", int(TensorProto.FLOAT), list(values.shape), values.view(numpy.uint8)
    )
    (output,) = runtime.run_model(model, [tensor])
    return runtime.tensor_to_numpy(output).view(numpy.float32).reshape(values.shape)


def make_input_tensor(values):
    """Builds the named input :class:`Tensor` fed to ``runtime.run_model``."""

    return runtime.tensor_from_numpy(
        "X", int(TensorProto.FLOAT), list(values.shape), values.view(numpy.uint8)
    )


def run_onnx_light_run_model_tensor(tensor):
    """Runs the whole model through ``runtime.run_model`` with a pre-built
    :class:`Tensor` input and returns the output :class:`Tensor` directly.

    Unlike :func:`run_onnx_light_run_model_numpy`, no NumPy conversion happens
    inside the timed region: the input tensor is constructed once beforehand and
    the output is left as a :class:`Tensor`, so the measurement reflects the raw
    model-execution cost of the API.
    """

    (output,) = runtime.run_model(model, [tensor])
    return output


def run_onnxruntime(values):
    """Runs the ONNX Runtime Abs kernel."""

    return ort_session.run(None, {"X": values})[0]


# %%
# Measure steady-state execution
# ------------------------------
#
# Normal execution uses the complete logarithmic grid. Documentation tests use
# two small vectors to keep the gallery build fast. Both runtimes receive the
# same input, are warmed before timing, and alternate which runtime runs first.

if os.environ.get("UNITTEST_GOING") == "1":
    size_grid = [100, 1_000]
    minimum_repeat = 3
    warmup = 1
else:
    size_grid = [10**power for power in range(2, 9)]
    minimum_repeat = 7
    warmup = 3

random_generator = numpy.random.default_rng(0)
rows = []
for size in size_grid:
    values = random_generator.uniform(-100.0, 100.0, size=size).astype(numpy.float32)
    expected = numpy.abs(values)
    repeat = max(minimum_repeat, min(200, 2_000_000 // size))

    numpy_time = measure(lambda values=values: numpy.abs(values), repeat, warmup)
    onnx_light_time, ort_time = measure_pair(
        lambda values=values: run_onnx_light(values),
        lambda values=values: run_onnxruntime(values),
        repeat,
        warmup,
    )
    run_model_numpy_time = measure(
        lambda values=values: run_onnx_light_run_model_numpy(values), repeat, warmup
    )
    input_tensor = make_input_tensor(values)
    run_model_tensor_time = measure(
        lambda tensor=input_tensor: run_onnx_light_run_model_tensor(tensor), repeat, warmup
    )

    numpy.testing.assert_array_equal(run_onnx_light(values), expected)
    numpy.testing.assert_array_equal(run_onnxruntime(values), expected)
    numpy.testing.assert_array_equal(run_onnx_light_run_model_numpy(values), expected)
    tensor_output = run_onnx_light_run_model_tensor(input_tensor)
    numpy.testing.assert_array_equal(
        runtime.tensor_to_numpy(tensor_output).view(numpy.float32).reshape(values.shape), expected
    )
    rows.append(
        (size, numpy_time, onnx_light_time, ort_time, run_model_numpy_time, run_model_tensor_time)
    )
    print(
        f"size={size:>9} | numpy={numpy_time * 1e6:10.2f} us | "
        f"onnx-light={onnx_light_time * 1e6:10.2f} us | "
        f"run_model(numpy)={run_model_numpy_time * 1e6:10.2f} us | "
        f"run_model(Tensor)={run_model_tensor_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us | "
        f"onnx-light / onnxruntime={onnx_light_time / ort_time:5.2f}x"
    )

sizes = numpy.array([row[0] for row in rows])
numpy_times = numpy.array([row[1] for row in rows])
onnx_light_times = numpy.array([row[2] for row in rows])
ort_times = numpy.array([row[3] for row in rows])
run_model_numpy_times = numpy.array([row[4] for row in rows])
run_model_tensor_times = numpy.array([row[5] for row in rows])

# %%
# Plot execution time and relative speed
# --------------------------------------
#
# The left panel shows raw inference time. The right panel divides the
# ONNX Runtime time by each backend's time: values above one are faster than
# ONNX Runtime, while values below one are slower.

figure, (time_axis, speedup_axis) = matplotlib.pyplot.subplots(1, 2, figsize=(12, 4.5))

time_axis.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
time_axis.plot(sizes, onnx_light_times * 1e6, "o-", label="onnx-light", color="#5cb85c")
time_axis.plot(
    sizes, run_model_numpy_times * 1e6, "o:", label="run_model (numpy)", color="#2e7d32"
)
time_axis.plot(
    sizes, run_model_tensor_times * 1e6, "s:", label="run_model (Tensor)", color="#1b5e20"
)
time_axis.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
time_axis.set_xscale("log")
time_axis.set_yscale("log")
time_axis.set_xlabel("array size (elements)")
time_axis.set_ylabel("time (microseconds)")
time_axis.set_title("Abs execution time")
time_axis.legend()

speedup_axis.plot(sizes, ort_times / numpy_times, "o--", label="numpy", color="#9b7ec8")
onnx_light_speedups = ort_times / onnx_light_times
speedup_axis.plot(sizes, onnx_light_speedups, "o-", label="onnx-light", color="#5cb85c")
for size, speedup in zip(sizes, onnx_light_speedups, strict=True):
    speedup_axis.annotate(
        f"{speedup:.2f}x",
        (size, speedup),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color="#3d803d",
    )
speedup_axis.plot(
    sizes, ort_times / run_model_numpy_times, "o:", label="run_model (numpy)", color="#2e7d32"
)
speedup_axis.plot(
    sizes, ort_times / run_model_tensor_times, "s:", label="run_model (Tensor)", color="#1b5e20"
)
speedup_axis.plot(sizes, ort_times / ort_times, "o-", label="onnxruntime", color="#f4a259")
speedup_axis.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
speedup_axis.set_xscale("log")
speedup_axis.set_xlabel("array size (elements)")
speedup_axis.set_ylabel("speed-up vs onnxruntime")
speedup_axis.set_title("Abs speed-up (onnxruntime = 1)")
speedup_axis.legend()

figure.tight_layout()
figure.savefig("plot_abs_benchmark.png")
