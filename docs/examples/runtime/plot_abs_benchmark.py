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
``onnxruntime`` measurement for ``bfloat16`` and uses :func:`numpy.abs` as the
baseline for that element type instead.

Initialization and execution are reported separately. Constructing an
``onnx-light`` :class:`~onnx_light.onnx.reference.ReferenceEvaluator` is
lightweight, but that does not imply that each inference is faster. The
execution benchmark warms both runtimes, alternates their measurement order,
and reports median durations so the comparison does not confuse startup cost
with steady-state inference cost.

The benchmark also exercises the low-level
:class:`onnx_light.onnx_py._onnxpykernels.runtime.RuntimeSession` entry point,
which runs a whole model end-to-end from :class:`Tensor` inputs to
:class:`Tensor` outputs. The ``run (RuntimeSession)`` series reuses a *pre-built*
:class:`RuntimeSession` and :class:`RuntimeContext`, feeds them a pre-built
input :class:`Tensor`, and keeps the output as a :class:`Tensor`, isolating the
raw model-execution cost.

The convenience helper
:func:`onnx_light.onnx_py._onnxpykernels.runtime.run_model` is deliberately
*not* used inside the timing loop: it rebuilds the whole
:class:`RuntimeSession` on every call — parsing the model, registering its
functions and building a fresh :class:`~onnx_light.onnx_py._onnxpykernels.runtime.ExecutionPlan`
— so timing it would measure session-construction cost, not steady-state
execution. That one-time setup dwarfs the elementwise ``Abs`` work and would
make the low-level series look many times slower than it really is (a very bad,
far-below-one speed-up). Building the session once and running it repeatedly
brings the measurement back to the raw kernel cost, close to the other
``onnx-light`` line.
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot
import ml_dtypes
import numpy
import onnxruntime
from onnx_light.onnx import TensorProto, checker, helper
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_py import _onnxpykernels

runtime = _onnxpykernels.runtime
ORT_MAX_IR_VERSION = 13
OPSET_VERSION = 18

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

    graph = helper.make_graph(
        [helper.make_node("Abs", ["X"], ["Y"])],
        "abs_benchmark",
        [helper.make_tensor_value_info("X", elem_type, ["N"])],
        [helper.make_tensor_value_info("Y", elem_type, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET_VERSION)])
    model.ir_version = min(model.ir_version, ORT_MAX_IR_VERSION)
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
# Measurement grid
# ----------------
#
# Normal execution uses the complete logarithmic grid. Documentation tests use
# two small vectors to keep the gallery build fast.

if os.environ.get("UNITTEST_GOING") == "1":
    size_grid = [100, 1_000]
    minimum_repeat = 3
    warmup = 1
else:
    size_grid = [10**power for power in range(2, 9)]
    minimum_repeat = 7
    warmup = 3


def benchmark_dtype(label: str, elem_type: int, np_dtype, ort_supported: bool) -> dict:
    """Benchmarks the Abs kernel for a single element type.

    Both runtimes receive the same input, are warmed before timing, and
    alternate which runtime runs first. ``onnxruntime`` is only exercised when
    it provides a CPU ``Abs`` kernel for ``elem_type``.

    Returns:
        A mapping with the measured ``sizes`` and, for each backend, the
        median execution times. ``onnxruntime`` times are ``None`` when the
        element type is unsupported.
    """

    model = make_abs_model(elem_type)
    model_bytes = model.SerializeToString()

    onnx_light_session = ReferenceEvaluator(model)
    runtime_session = runtime.RuntimeSession(model)
    ort_session = None
    if ort_supported:
        ort_session = onnxruntime.InferenceSession(
            model_bytes, providers=["CPUExecutionProvider"]
        )

    def run_onnx_light(values):
        """Runs the built-in onnx-light Abs kernel."""

        return onnx_light_session.run(None, {"X": values})[0]

    def make_input_context(values):
        """Builds a :class:`RuntimeContext` seeded with the input :class:`Tensor`.

        The context is built once per size, outside the timing loop, so that the
        measured ``runtime_session.run`` call only pays for the model execution
        and not for building the input tensor or the context.
        """

        input_tensor = runtime.tensor_from_numpy(
            "X", int(elem_type), list(values.shape), values.view(numpy.uint8)
        )
        context = runtime.RuntimeContext(
            runtime.KernelContext(runtime.default_opset(OPSET_VERSION))
        )
        context.set("X", input_tensor, "input")
        return context

    def run_onnx_light_session(context):
        """Runs the whole model through a reused :class:`RuntimeSession`.

        The session and context are built once and reused across iterations, so
        this measures the raw model-execution cost. Using the
        :func:`runtime.run_model` convenience helper here instead would rebuild
        the session on every call and dominate the measurement with
        session-construction overhead.
        """

        runtime_session.run(context)
        return context.get("Y")

    def run_onnxruntime(values):
        """Runs the ONNX Runtime Abs kernel."""

        return ort_session.run(None, {"X": values})[0]

    random_generator = numpy.random.default_rng(0)
    rows = []
    for size in size_grid:
        values = random_generator.uniform(-100.0, 100.0, size=size).astype(np_dtype)
        expected = numpy.abs(values)
        repeat = max(minimum_repeat, min(200, 2_000_000 // size))

        numpy_time = measure(lambda values=values: numpy.abs(values), repeat, warmup)
        if ort_supported:
            onnx_light_time, ort_time = measure_pair(
                lambda values=values: run_onnx_light(values),
                lambda values=values: run_onnxruntime(values),
                repeat,
                warmup,
            )
        else:
            onnx_light_time = measure(
                lambda values=values: run_onnx_light(values), repeat, warmup
            )
            ort_time = None
        input_context = make_input_context(values)
        run_model_tensor_time = measure(
            lambda context=input_context: run_onnx_light_session(context), repeat, warmup
        )

        numpy.testing.assert_array_equal(run_onnx_light(values), expected)
        if ort_supported:
            numpy.testing.assert_array_equal(run_onnxruntime(values), expected)
        tensor_output = run_onnx_light_session(input_context)
        numpy.testing.assert_array_equal(
            runtime.tensor_to_numpy(tensor_output).view(np_dtype).reshape(values.shape), expected
        )
        rows.append((size, numpy_time, onnx_light_time, ort_time, run_model_tensor_time))
        ort_report = "n/a" if ort_time is None else f"{ort_time * 1e6:10.2f} us"
        ratio_report = "n/a" if ort_time is None else f"{onnx_light_time / ort_time:5.2f}x"
        print(
            f"[{label:>8}] size={size:>9} | numpy={numpy_time * 1e6:10.2f} us | "
            f"onnx-light={onnx_light_time * 1e6:10.2f} us | "
            f"run(RuntimeSession)={run_model_tensor_time * 1e6:10.2f} us | "
            f"onnxruntime={ort_report} | onnx-light / onnxruntime={ratio_report}"
        )

    return {
        "label": label,
        "ort_supported": ort_supported,
        "sizes": numpy.array([row[0] for row in rows]),
        "numpy_times": numpy.array([row[1] for row in rows]),
        "onnx_light_times": numpy.array([row[2] for row in rows]),
        "ort_times": None if not ort_supported else numpy.array([row[3] for row in rows]),
        "run_model_tensor_times": numpy.array([row[4] for row in rows]),
    }


# %%
# Measure steady-state execution for every element type
# -----------------------------------------------------

results = [benchmark_dtype(*entry) for entry in DTYPES]

# %%
# Plot execution time and relative speed
# --------------------------------------
#
# One row is drawn per element type. The left panel shows raw inference time.
# The right panel shows the speed-up relative to a baseline: ``onnxruntime``
# when it provides a kernel, otherwise :func:`numpy.abs`. Values above one are
# faster than the baseline, while values below one are slower.

figure, axes = matplotlib.pyplot.subplots(
    len(results), 2, figsize=(12, 4.5 * len(results)), squeeze=False
)

for row_index, result in enumerate(results):
    label = result["label"]
    sizes = result["sizes"]
    numpy_times = result["numpy_times"]
    onnx_light_times = result["onnx_light_times"]
    ort_times = result["ort_times"]
    run_model_tensor_times = result["run_model_tensor_times"]

    time_axis = axes[row_index][0]
    speedup_axis = axes[row_index][1]

    time_axis.plot(sizes, numpy_times * 1e6, "o--", label="numpy", color="#9b7ec8")
    time_axis.plot(sizes, onnx_light_times * 1e6, "o-", label="onnx-light", color="#5cb85c")
    time_axis.plot(
        sizes, run_model_tensor_times * 1e6, "s:", label="run (RuntimeSession)", color="#1b5e20"
    )
    if ort_times is not None:
        time_axis.plot(sizes, ort_times * 1e6, "o-", label="onnxruntime", color="#f4a259")
    time_axis.set_xscale("log")
    time_axis.set_yscale("log")
    time_axis.set_xlabel("array size (elements)")
    time_axis.set_ylabel("time (microseconds)")
    time_axis.set_title(f"Abs execution time ({label})")
    time_axis.legend()

    if ort_times is not None:
        baseline = ort_times
        baseline_name = "onnxruntime"
    else:
        baseline = numpy_times
        baseline_name = "numpy"

    # The baseline itself is a flat line at 1.0, shown by the reference
    # ``axhline`` below, so it is not plotted as its own series.
    if ort_times is not None:
        speedup_axis.plot(sizes, baseline / numpy_times, "o--", label="numpy", color="#9b7ec8")
    onnx_light_speedups = baseline / onnx_light_times
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
        baseline / run_model_tensor_times,
        "s:",
        label="run (RuntimeSession)",
        color="#1b5e20",
    )
    speedup_axis.axhline(
        1.0, color="grey", linewidth=0.8, linestyle=":", label=f"{baseline_name} (baseline)"
    )
    speedup_axis.set_xscale("log")
    speedup_axis.set_xlabel("array size (elements)")
    speedup_axis.set_ylabel(f"speed-up vs {baseline_name}")
    speedup_axis.set_title(f"Abs speed-up ({label}, {baseline_name} = 1)")
    speedup_axis.legend()

# %%
# ``onnx-light`` is expected to beat ``onnxruntime`` on the smallest ``float32``
# vector, where the fixed per-call overhead dominates.

float32_result = results[0]
float32_speedups = float32_result["ort_times"] / float32_result["onnx_light_times"]
assert float32_speedups[0] > 1.0, (
    "onnx-light is expected to be faster than onnxruntime for the first (smallest) size, "
    f"got a speed-up of {float32_speedups[0]:.2f}x for size {float32_result['sizes'][0]}"
)

figure.tight_layout()
figure.savefig("plot_abs_benchmark.png")
