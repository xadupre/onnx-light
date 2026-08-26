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

import argparse
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", "--repeat", type=int, default=10 * (os.cpu_count() or 1))
parser.add_argument("-w", "--warmup", type=int, default=2 * (os.cpu_count() or 1))
parser.add_argument("-t", "--max-repeat-time", type=float, default=1.0)
args, _ = parser.parse_known_args()
if args.repeat <= 0:
    parser.error("--repeat must be greater than 0")
if args.warmup < 0:
    parser.error("--warmup must be greater than or equal to 0")
if args.max_repeat_time <= 0:
    parser.error("--max-repeat-time must be greater than 0")


# %%
# Measurement grid
# ----------------
#
# Each runtime gets up to ``--warmup`` untimed calls, then up to ``--repeat``
# measured calls. Both phases stop after ``--max-repeat-time`` cumulative
# seconds.


def measure(function, warmup: int, repeat: int, max_duration: float) -> float:
    """Measures a callable after warm-up and returns its median time per call.

    Returns:
        The median wall-clock duration, in seconds, of at most ``repeat``
        calls to ``function`` (excluding the ``warmup`` calls). Measurement
        stops once their cumulative duration reaches ``max_duration``.
    """

    warmup_duration = 0.0
    for _ in range(warmup):
        start = time.perf_counter()
        function()
        warmup_duration += time.perf_counter() - start
        if warmup_duration >= max_duration:
            break
    timings = []
    total_duration = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        duration = time.perf_counter() - start
        timings.append(duration)
        total_duration += duration
        if total_duration >= max_duration:
            break
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


def prepare_case(op_name: str) -> dict:
    """Prepares one backend benchmark case for both runtimes.

    Returns:
        The model, inputs, expected output, tolerance, and display metadata.
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
    expected = tensor_to_numpy(data_set.outputs[0])
    return {
        "op_name": op_name,
        "test_name": tc.name,
        "model": model,
        "feeds": feeds,
        "expected": expected,
        "rtol": tc.rtol,
        "atol": tc.atol,
        "n_elements": int(numpy.prod(next(iter(feeds.values())).shape)),
    }


def benchmark_case(case: dict, backend: str) -> float:
    """Benchmarks one prepared case with one runtime."""
    if backend == "onnx_light":
        session = ReferenceEvaluator(case["model"])
    elif backend == "onnxruntime":
        session = onnxruntime.InferenceSession(
            case["model"].SerializeToString(), providers=["CPUExecutionProvider"]
        )
    else:
        raise ValueError(f"Unexpected backend {backend!r}.")

    def run():
        return session.run(None, case["feeds"])[0]

    numpy.testing.assert_allclose(run(), case["expected"], rtol=case["rtol"], atol=case["atol"])
    return measure(run, args.warmup, args.repeat, args.max_repeat_time)


# %%
# Run the benchmark for every selected operator
# ----------------------------------------------
#
# Every onnx-light case is completed before any ONNX Runtime session is
# created. Both runtimes retain their default spinning behavior, but a pool
# from one runtime therefore cannot remain alive while the other runtime's
# cases are measured.

cases = [prepare_case(op_name) for op_name in BENCHMARK_OPS]
onnx_light_results = [benchmark_case(case, "onnx_light") for case in cases]
ort_results = [benchmark_case(case, "onnxruntime") for case in cases]
results = []
for case, onnx_light_time, ort_time in zip(cases, onnx_light_results, ort_results, strict=True):
    results.append(
        {
            "op_name": case["op_name"],
            "n_elements": case["n_elements"],
            "onnx_light_time": onnx_light_time,
            "ort_time": ort_time,
        }
    )
    print(
        f"[{case['test_name']:>28}] n={case['n_elements']:>10} | "
        f"onnx-light={onnx_light_time * 1e6:10.2f} us | "
        f"onnxruntime={ort_time * 1e6:10.2f} us | "
        f"onnx-light / onnxruntime={onnx_light_time / ort_time:5.2f}x"
    )

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
