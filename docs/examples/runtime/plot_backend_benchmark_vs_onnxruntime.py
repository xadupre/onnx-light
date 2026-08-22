"""
.. _l-example-plot-backend-benchmark-vs-onnxruntime:

Benchmark a subset of backend test cases against onnxruntime
==============================================================

The C++ backend test registry exposed by :mod:`onnx_light.onnx.backend`
does not only contain the small correctness cases used to validate the
runtime: every operator family that supports it also registers a
*benchmark* variant (``TestMode.BENCHMARK``) whose inputs are large enough
(millions of elements) for a single kernel evaluation to run long enough to
be timed reliably. Benchmark case names always contain ``"_benchmark"``.

This example selects a small subset of those benchmark cases -- a mix of
unary (``Abs``, ``Relu``, ``Sigmoid``, ``Sqrt``, ``Exp``, ``Erf``) and
binary (``Add``, ``Mul``, ``Div``) float32 element-wise operators -- and
compares the median execution time of ``onnx-light``'s
:class:`~onnx_light.onnx.reference.ReferenceEvaluator` against
``onnxruntime`` running the very same :class:`ModelProto` and input data.
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot
import numpy
import onnxruntime

from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name
from onnx_light.onnx.reference import ReferenceEvaluator

# %%
# Selecting a subset of backend benchmark cases
# ----------------------------------------------
#
# Every eligible operator family registers a benchmark case named
# ``test_cc_<lower op name>_benchmark`` (plus ``_float16`` / ``_bfloat16``
# companions this example does not use). We only keep the float32 variant so
# the comparison against ``onnxruntime`` is straightforward.

BENCHMARK_OPS = ["abs", "relu", "sigmoid", "sqrt", "exp", "erf", "add", "mul", "div"]


# %%
# Measurement grid
# ----------------
#
# Documentation builds use a single warm-up and repeat to keep the gallery
# fast; a normal run measures a more statistically stable median.

if os.environ.get("UNITTEST_GOING") == "1":
    warmup, repeat = 1, 2
else:
    warmup, repeat = 3, 7


def measure(function, warmup: int, repeat: int) -> float:
    """Measures a callable after warm-up and returns its median time per call.

    Returns:
        The median wall-clock duration, in seconds, of ``repeat`` calls to
        ``function`` (excluding the ``warmup`` calls).
    """

    for _ in range(warmup):
        function()
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(numpy.median(timings))


_DTYPE_MAP = {1: numpy.float32, 10: numpy.float16}


def tensor_to_numpy(tensor) -> numpy.ndarray:
    """Converts a C++ backend-test ``Tensor`` to a :class:`numpy.ndarray`.

    Returns:
        A :class:`numpy.ndarray` view over ``tensor``'s raw bytes, reshaped to
        ``tensor.shape``.

    Raises:
        ValueError: If ``tensor.data_type`` is not one of the float types
            handled by this example (``FLOAT`` or ``FLOAT16``).
    """

    dtype = _DTYPE_MAP.get(int(tensor.data_type))
    if dtype is None:
        raise ValueError(
            f"tensor_to_numpy does not support data_type={tensor.data_type!r}; "
            f"expected one of {sorted(_DTYPE_MAP)}."
        )
    return numpy.frombuffer(tensor.raw_data(), dtype=dtype).reshape(
        tuple(int(d) for d in tensor.shape)
    )


def benchmark_case(op_name: str) -> dict:
    """Benchmarks one backend benchmark case against onnx-light and onnxruntime.

    Returns:
        A mapping with the operator name, the number of elements processed
        and the median execution time measured for each runtime.
    """

    pattern = f"^test_cc_{op_name}_benchmark$"
    matches = collect_test_cases_by_name(pattern, mode=TestMode.BENCHMARK)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one benchmark case matching {pattern!r}, got {len(matches)}."
        )
    tc = matches[0]
    model = tc.model
    input_names = [vi.name for vi in model.graph.input]
    data_set = tc.data_sets[0]
    feeds = {name: tensor_to_numpy(tensor) for name, tensor in zip(input_names, data_set.inputs)}

    onnx_light_session = ReferenceEvaluator(model)

    def run_onnx_light():
        return onnx_light_session.run(None, feeds)[0]

    ort_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    def run_onnxruntime():
        return ort_session.run(None, feeds)[0]

    expected = tensor_to_numpy(data_set.outputs[0])
    numpy.testing.assert_allclose(run_onnx_light(), expected, rtol=tc.rtol, atol=tc.atol)
    numpy.testing.assert_allclose(run_onnxruntime(), expected, rtol=tc.rtol, atol=tc.atol)

    onnx_light_time = measure(run_onnx_light, warmup, repeat)
    ort_time = measure(run_onnxruntime, warmup, repeat)

    n_elements = int(numpy.prod(next(iter(feeds.values())).shape))
    print(
        f"[{tc.name:>28}] n={n_elements:>10} | "
        f"onnx-light={onnx_light_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us | "
        f"onnx-light / onnxruntime={onnx_light_time / ort_time:5.2f}x"
    )
    return {
        "op_name": op_name,
        "n_elements": n_elements,
        "onnx_light_time": onnx_light_time,
        "ort_time": ort_time,
    }


# %%
# Run the benchmark for every selected operator
# ----------------------------------------------

results = [benchmark_case(op_name) for op_name in BENCHMARK_OPS]

# %%
# Plot the comparison
# --------------------
#
# The left panel shows the raw median execution time for each backend, the
# right panel shows the speed-up of ``onnx-light`` relative to
# ``onnxruntime`` (values above 1.0 mean ``onnx-light`` is slower).

labels = [r["op_name"] for r in results]
onnx_light_times = numpy.array([r["onnx_light_time"] for r in results])
ort_times = numpy.array([r["ort_time"] for r in results])
x = numpy.arange(len(labels))
width = 0.35

figure, (time_axis, speedup_axis) = matplotlib.pyplot.subplots(1, 2, figsize=(12, 4.5))

time_axis.bar(x - width / 2, onnx_light_times * 1e6, width, label="onnx-light", color="#5cb85c")
time_axis.bar(x + width / 2, ort_times * 1e6, width, label="onnxruntime", color="#f4a259")
time_axis.set_xticks(x)
time_axis.set_xticklabels(labels, rotation=45, ha="right")
time_axis.set_ylabel("time (microseconds)")
time_axis.set_yscale("log")
time_axis.set_title("Backend benchmark cases: execution time")
time_axis.legend()

speedup = onnx_light_times / ort_times
speedup_axis.bar(x, speedup, color="#5cb85c")
speedup_axis.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
speedup_axis.set_xticks(x)
speedup_axis.set_xticklabels(labels, rotation=45, ha="right")
speedup_axis.set_ylabel("onnx-light / onnxruntime (lower is better)")
speedup_axis.set_title("Relative speed vs onnxruntime")

figure.tight_layout()
figure.savefig("plot_backend_benchmark_vs_onnxruntime.png")
