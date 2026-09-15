"""
.. _l-example-plot-abs-benchmark:

Benchmark Abs: onnxruntime vs onnx-light
========================================

This example compares the built-in :class:`Abs` kernel in ``onnx-light`` with
``onnxruntime`` and :func:`numpy.abs` for vectors ranging from one hundred to
one hundred million elements. The comparison is repeated for three element
types: ``float32``, ``float16`` and ``bfloat16``. The ``bfloat16`` values rely
on the :mod:`ml_dtypes` NumPy extension, which provides a native ``bfloat16``
dtype.

``onnxruntime`` only ships a CPU ``Abs`` implementation for ``float32`` and
``float16``; there is no ``bfloat16`` kernel. The benchmark therefore skips the
``onnxruntime`` measurement and speed-up plot for ``bfloat16``.

The execution benchmark warms each runtime and reports median durations.
``onnx-light`` is measured before the ONNX Runtime session is constructed:
keeping both persistent CPU pools in one process while alternating calls causes
one runtime's spinning workers to perturb the other runtime's measurement.
The two ``onnx-light`` series use the same prepared evaluator with either a
NumPy array or a pre-built runtime ``Tensor`` as input.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot
import ml_dtypes
import numpy
import onnxruntime
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto, checker
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_py import _onnxpykernels

runtime = _onnxpykernels.runtime
ORT_MAX_IR_VERSION = 13

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
# Element types under test
# ------------------------
#
# Each entry holds the label used in the report, the ONNX ``TensorProto`` type,
# the matching NumPy dtype and whether ``onnxruntime`` provides a CPU ``Abs``
# kernel for that type. ``bfloat16`` is materialized through
# :mod:`ml_dtypes`; ``onnxruntime`` has no ``bfloat16`` ``Abs`` kernel, so it is
# excluded from that comparison.

DTYPES = [
    ("float32", TensorProto.FLOAT, numpy.dtype(numpy.float32), True),
    ("float16", TensorProto.FLOAT16, numpy.dtype(numpy.float16), True),
    ("bfloat16", TensorProto.BFLOAT16, numpy.dtype(ml_dtypes.bfloat16), False),
]


def make_abs_model(elem_type: int):
    """Creates a dynamic one-dimensional Abs model for a given element type."""

    graph = oh.make_graph(
        [oh.make_node("Abs", ["X"], ["Y"])],
        "abs_benchmark",
        [oh.make_tensor_value_info("X", elem_type, ["N"])],
        [oh.make_tensor_value_info("Y", elem_type, ["N"])],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, ORT_MAX_IR_VERSION)
    checker.check_model(model)
    return model


def measure(function, repeat: int, warmup: int, number: int, max_duration: float) -> float:
    """Measures a callable after warm-up and returns its median time per call."""

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
        for _ in range(number):
            function()
        duration = time.perf_counter() - start
        timings.append(duration / number)
        total_duration += duration
        if total_duration >= max_duration:
            break
    return float(numpy.median(timings))


# %%
# Measurement grid
# ----------------
#
# Normal execution uses the complete logarithmic grid. Documentation tests use
# two small vectors to keep the gallery build fast.

if os.environ.get("UNITTEST_GOING") == "1":
    size_grid = [100, 1_000]
else:
    size_grid = [10**power for power in range(2, 9)]


def benchmark_dtype(label: str, elem_type: int, np_dtype, ort_supported: bool) -> dict:
    """Benchmarks the Abs kernel for a single element type.

    Both runtimes receive inputs generated from the same seed and are warmed
    before timing. All ``onnx-light`` measurements finish before the ONNX
    Runtime session is constructed, keeping their persistent CPU pools from
    perturbing each other. ``onnxruntime`` is only exercised when it provides a
    CPU ``Abs`` kernel for ``elem_type``.

    Returns:
        A mapping with the measured ``sizes`` and, for each backend, the
        median execution times. ``onnxruntime`` times are ``None`` when the
        element type is unsupported.
    """

    model = make_abs_model(elem_type)
    model_bytes = model.SerializeToString()

    onnx_light_session = ReferenceEvaluator(model)

    def run_onnx_light(values):
        """Runs the built-in onnx-light Abs kernel."""

        return onnx_light_session.run(None, {"X": values})[0]

    def run_onnx_light_tensor(tensor):
        """Runs onnx-light with a pre-built runtime Tensor."""

        return onnx_light_session.run(None, {"X": tensor})[0]

    def make_input_tensor(values):
        """Creates a zero-copy runtime Tensor over a NumPy input."""

        return runtime.tensor_from_numpy(
            "X", int(elem_type), list(values.shape), values.view(numpy.uint8), copy=False
        )

    random_generator = numpy.random.default_rng(0)
    rows_by_size = {}
    for size in size_grid:
        values = random_generator.uniform(-100.0, 100.0, size=size).astype(np_dtype)
        expected = numpy.abs(values)
        number = max(1, min(20, 10_000_000 // size))

        numpy_time = measure(
            lambda values=values: numpy.abs(values),
            args.repeat,
            args.warmup,
            number,
            args.max_repeat_time,
        )
        onnx_light_time = measure(
            lambda values=values: run_onnx_light(values),
            args.repeat,
            args.warmup,
            number,
            args.max_repeat_time,
        )
        input_tensor = make_input_tensor(values)
        onnx_light_tensor_time = measure(
            lambda tensor=input_tensor: run_onnx_light_tensor(tensor),
            args.repeat,
            args.warmup,
            number,
            args.max_repeat_time,
        )
        numpy.testing.assert_array_equal(run_onnx_light(values), expected)
        numpy.testing.assert_array_equal(run_onnx_light_tensor(input_tensor), expected)
        rows_by_size[size] = [size, numpy_time, onnx_light_time, onnx_light_tensor_time, None]

    rows = [tuple(rows_by_size[size]) for size in size_grid]
    return {
        "label": label,
        "ort_supported": ort_supported,
        "model_bytes": model_bytes,
        "sizes": numpy.array([row[0] for row in rows]),
        "numpy_times": numpy.array([row[1] for row in rows]),
        "onnx_light_times": numpy.array([row[2] for row in rows]),
        "onnx_light_tensor_times": numpy.array([row[3] for row in rows]),
        "ort_times": None,
    }


def benchmark_onnxruntime(result: dict, np_dtype) -> None:
    """Measures ONNX Runtime after every onnx-light case has completed."""
    if not result["ort_supported"]:
        return
    session = onnxruntime.InferenceSession(
        result["model_bytes"], providers=["CPUExecutionProvider"]
    )
    random_generator = numpy.random.default_rng(0)
    ort_times = []
    for size in result["sizes"]:
        values = random_generator.uniform(-100.0, 100.0, size=int(size)).astype(np_dtype)
        expected = numpy.abs(values)
        number = max(1, min(20, 10_000_000 // int(size)))

        def run(values=values):
            return session.run(None, {"X": values})[0]

        ort_times.append(measure(run, args.repeat, args.warmup, number, args.max_repeat_time))
        numpy.testing.assert_array_equal(run(), expected)
    result["ort_times"] = numpy.array(ort_times)


def print_result(result: dict) -> None:
    """Prints one element type after both runtime phases have completed."""
    for size, numpy_time, onnx_light_time, tensor_time, ort_time in zip(
        result["sizes"],
        result["numpy_times"],
        result["onnx_light_times"],
        result["onnx_light_tensor_times"],
        result["ort_times"] if result["ort_times"] is not None else [None] * len(result["sizes"]),
        strict=True,
    ):
        ort_report = "n/a" if ort_time is None else f"{ort_time * 1e6:10.2f} us"
        ratio_report = "n/a" if ort_time is None else f"{onnx_light_time / ort_time:5.2f}x"
        print(
            f"[{result['label']:>8}] size={size:>9} | numpy={numpy_time * 1e6:10.2f} us | "
            f"onnx-light={onnx_light_time * 1e6:10.2f} us | "
            f"onnx-light (Tensor)={tensor_time * 1e6:10.2f} us | "
            f"onnxruntime={ort_report} | onnx-light / onnxruntime={ratio_report}"
        )


# %%
# Measure steady-state execution for every element type
# -----------------------------------------------------

results = [benchmark_dtype(*entry) for entry in DTYPES]
for result, (_, _, np_dtype, _) in zip(results, DTYPES, strict=True):
    benchmark_onnxruntime(result, np_dtype)
    print_result(result)

# %%
# Plot execution time and relative speed
# --------------------------------------
#
# One row is drawn per element type. The left panel shows raw inference time.
# The right panel shows the speed-up relative to ``onnxruntime`` when it
# provides a CPU kernel, on a logarithmic scale so that speed-ups and
# slowdowns are equally readable around the baseline. No speed-up is reported
# for ``bfloat16`` because changing the baseline to NumPy would make that row
# incomparable.

figure, axes = matplotlib.pyplot.subplots(
    len(results), 2, figsize=(12, 4.5 * len(results)), squeeze=False
)

for row_index, result in enumerate(results):
    label = result["label"]
    sizes = result["sizes"]
    numpy_times = result["numpy_times"]
    onnx_light_times = result["onnx_light_times"]
    onnx_light_tensor_times = result["onnx_light_tensor_times"]
    ort_times = result["ort_times"]

    time_axis = axes[row_index][0]
    speedup_axis = axes[row_index][1]

    time_axis.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
    time_axis.plot(sizes, onnx_light_times * 1e6, "o-", label="onnx-light", color="#5cb85c")
    time_axis.plot(
        sizes, onnx_light_tensor_times * 1e6, "s:", label="onnx-light (Tensor)", color="#1b5e20"
    )
    if ort_times is not None:
        time_axis.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
    time_axis.set_xscale("log")
    time_axis.set_yscale("log")
    time_axis.set_xlabel("array size (elements)")
    time_axis.set_ylabel("time (microseconds)")
    time_axis.set_title(f"Abs execution time ({label})")
    time_axis.legend()

    if ort_times is None:
        speedup_axis.axis("off")
        speedup_axis.text(
            0.5,
            0.5,
            "No ONNX Runtime CPU bfloat16 Abs kernel\nspeed-up not reported",
            ha="center",
            va="center",
            transform=speedup_axis.transAxes,
        )
        continue

    # The baseline itself is a flat line at 1.0, shown by the reference
    # ``axhline`` below, so it is not plotted as its own series.
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
        sizes,
        ort_times / onnx_light_tensor_times,
        "s:",
        label="onnx-light (Tensor)",
        color="#1b5e20",
    )
    speedup_axis.axhline(
        1.0, color="grey", linewidth=0.8, linestyle=":", label="onnxruntime (baseline)"
    )
    speedup_axis.set_xscale("log")
    speedup_axis.set_yscale("log")
    speedup_axis.set_xlabel("array size (elements)")
    speedup_axis.set_ylabel("speed-up vs onnxruntime")
    speedup_axis.set_title(f"Abs speed-up ({label}, onnxruntime = 1)")
    speedup_axis.legend()

figure.tight_layout()
figure.savefig("plot_abs_benchmark.png")
