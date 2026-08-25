# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Deterministic benchmark corpus and cross-machine baseline report (Step E).

See :ref:`l-next-steps-kernel-parallelization`. This module runs a fixed,
representative shape corpus for a small set of native ``onnx-light`` kernels
under an explicit serial CPU policy and the default session-thread policy,
and combines the measurements with the Step D
:mod:`onnx_light.tools.kernel_inventory` coverage state into one
machine-readable report.

The tool never invokes ``onnxruntime`` and never persists kernel tuning
values: only :func:`onnx_light.kernel_tuning.kernel_tuning_parameters` (a
read-only inspection call) and ordinary
:class:`onnx_light.onnx_py._onnxpykernels.runtime.RuntimeSession` runs are
used, so it can run as often as needed without disturbing the tuning cache.
"""

from __future__ import annotations

import os
import platform
import re
import time
from types import ModuleType
from typing import Any

import numpy

from . import kernel_inventory

resource: ModuleType | None
try:
    import resource  # Unix-only; process CPU time falls back to None on Windows.
except ImportError:  # pragma: no cover - exercised only on Windows.
    resource = None

__all__ = [
    "BENCHMARK_CORPUS",
    "CPU_POLICIES",
    "get_cpu_descriptor",
    "run_benchmark_corpus",
    "run_kernel_baseline_report",
]

# ``(label, size)`` pairs shared by every benchmark case unless overridden.
# ``size`` is the flat element count for elementwise cases and the square
# matrix dimension (M = N = K) for ``Gemm``.
_DEFAULT_SHAPES = (("small", 1_000), ("medium", 100_000), ("large", 4_000_000))
_GEMM_SHAPES = (("small", 32), ("medium", 256), ("large", 1024))

# Representative kernels: one memory-bound unary float kernel with an existing
# tuning schema (Abs), one compute-bound tunable kernel (Gemm), and one
# boolean/logical kernel with a fixed parallel policy (Not).
BENCHMARK_CORPUS: tuple[dict[str, Any], ...] = (
    {"op_type": "Abs", "arity": "unary", "element_type": "FLOAT", "shapes": _DEFAULT_SHAPES},
    {"op_type": "Not", "arity": "unary", "element_type": "BOOL", "shapes": _DEFAULT_SHAPES},
    {"op_type": "Gemm", "arity": "gemm", "element_type": "FLOAT", "shapes": _GEMM_SHAPES},
)

# ``(label, num_threads)`` CPU policies. ``num_threads = 1`` forces the
# serial path; ``num_threads = 0`` resolves to the default session-thread
# policy (one participant per detected physical core).
CPU_POLICIES: tuple[tuple[str, int], ...] = (("serial", 1), ("session_thread", 0))


def get_cpu_descriptor() -> dict[str, Any]:
    """Returns a best-effort, portable CPU descriptor for the local machine.

    Missing information is omitted rather than replaced by an invented value.

    Returns:
        A mapping with at least ``architecture`` and ``logical_cores``.
    """
    descriptor: dict[str, Any] = {
        "architecture": platform.machine(),
        "system": platform.system(),
        "system_release": platform.release(),
        "logical_cores": os.cpu_count(),
    }
    processor = platform.processor()
    if processor:
        descriptor["processor"] = processor
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as cpuinfo_file:
                text = cpuinfo_file.read()
        except OSError:
            text = ""
        model_name = re.search(r"model name\s*:\s*(.+)", text)
        if model_name:
            descriptor["model_name"] = model_name.group(1).strip()
        vendor_id = re.search(r"vendor_id\s*:\s*(.+)", text)
        if vendor_id:
            descriptor["vendor"] = vendor_id.group(1).strip()
    return descriptor


def _make_model(case: dict[str, Any], size: int):
    """Builds a one-node ONNX model for the given benchmark ``case`` and ``size``."""
    import onnx_light.onnx.helper as oh
    from onnx_light.onnx import TensorProto

    elem_type = int(getattr(TensorProto, case["element_type"]))
    if case["arity"] == "unary":
        graph = oh.make_graph(
            [oh.make_node(case["op_type"], ["X"], ["Y"])],
            f"{case['op_type']}_baseline",
            [oh.make_tensor_value_info("X", elem_type, [size])],
            [oh.make_tensor_value_info("Y", elem_type, [size])],
        )
    elif case["arity"] == "gemm":
        graph = oh.make_graph(
            [oh.make_node("Gemm", ["A", "B"], ["Y"])],
            "Gemm_baseline",
            [
                oh.make_tensor_value_info("A", elem_type, [size, size]),
                oh.make_tensor_value_info("B", elem_type, [size, size]),
            ],
            [oh.make_tensor_value_info("Y", elem_type, [size, size])],
        )
    else:
        raise ValueError(f"Unsupported benchmark arity: {case['arity']!r}")
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
    return model


def _make_inputs(case: dict[str, Any], size: int, seed: int) -> dict[str, numpy.ndarray]:
    """Builds deterministic NumPy inputs for the given benchmark ``case``."""
    generator = numpy.random.default_rng(seed)

    def make_array(shape: tuple[int, ...]) -> numpy.ndarray:
        if case["element_type"] == "BOOL":
            return generator.integers(0, 2, size=shape).astype(numpy.bool_)
        return generator.uniform(-10.0, 10.0, size=shape).astype(numpy.float32)

    if case["arity"] == "unary":
        return {"X": make_array((size,))}
    if case["arity"] == "gemm":
        return {"A": make_array((size, size)), "B": make_array((size, size))}
    raise ValueError(f"Unsupported benchmark arity: {case['arity']!r}")


def _run_policy_case(
    model: Any,
    inputs: dict[str, numpy.ndarray],
    *,
    elem_type: int,
    num_threads: int,
    repeat: int,
    warmup: int,
    collect_diagnostics: bool,
) -> dict[str, Any]:
    """Runs one ``(model, cpu policy)`` combination and measures timing and CPU use."""
    from onnx_light.onnx_py import _onnxpykernels  # type: ignore[attr-defined]

    runtime = _onnxpykernels.runtime

    policy = runtime.CpuExecutionPolicy()
    policy.num_threads = num_threads
    resolved = runtime.resolve_cpu_execution_policy(policy)

    def _make_context(opts):
        session = runtime.RuntimeSession(model, opts)
        context = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))
        for name, array in inputs.items():
            raw = numpy.ascontiguousarray(array).view(numpy.uint8).ravel()
            tensor = runtime.tensor_from_numpy(
                name, elem_type, list(array.shape), raw, copy=False
            )
            context.set(name, tensor)
        return session, context

    startup_start = time.perf_counter()
    session, context = _make_context(runtime.RuntimeSessionOptions(cpu_execution=policy))
    session.run(context)  # first run performs kernel resolution/tuning: part of startup.
    startup_seconds = time.perf_counter() - startup_start

    for _ in range(warmup):
        session.run(context)

    ru_before = resource.getrusage(resource.RUSAGE_SELF) if resource is not None else None
    wall_before = time.perf_counter()
    per_call_seconds = []
    for _ in range(repeat):
        call_start = time.perf_counter()
        session.run(context)
        per_call_seconds.append(time.perf_counter() - call_start)
    wall_seconds = time.perf_counter() - wall_before
    if resource is not None and ru_before is not None:
        ru_after = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = (ru_after.ru_utime + ru_after.ru_stime) - (
            ru_before.ru_utime + ru_before.ru_stime
        )
    else:
        cpu_seconds = None
    cpu_utilization = (
        cpu_seconds / wall_seconds if cpu_seconds is not None and wall_seconds > 0 else None
    )

    diagnostics: dict[str, Any] | None = None
    if collect_diagnostics:
        collector = runtime.ParallelRegionCollector(capacity=64, hardware_counters=True)
        diag_options = runtime.RuntimeSessionOptions(
            cpu_execution=policy, parallel_region_collector=collector
        )
        diag_session, diag_context = _make_context(diag_options)
        diag_session.run(diag_context)
        report = diag_session.parallel_region_report()
        events = list(report.events)
        if events:
            representative = max(events, key=lambda event: event.wall_time_ns or 0)
            diagnostics = {
                "grain_size": representative.grain_size,
                "requested_threads": representative.requested_threads,
                "admitted_threads": representative.admitted_threads,
                "observed_threads": representative.observed_threads,
                "counter_status": representative.counter_status,
                "ipc": representative.ipc,
                "llc_miss_rate": representative.llc_miss_rate,
                "dropped_events": report.dropped_events,
            }
        else:
            diagnostics = {"dropped_events": report.dropped_events}

    per_call_seconds.sort()
    median_seconds = per_call_seconds[len(per_call_seconds) // 2]
    return {
        "num_threads_requested": num_threads,
        "effective_threads": resolved.effective_threads,
        "startup_seconds": startup_seconds,
        "median_kernel_execution_seconds": median_seconds,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "cpu_utilization": cpu_utilization,
        "repeat": repeat,
        "diagnostics": diagnostics,
    }


def run_benchmark_corpus(
    *,
    cases: tuple[dict[str, Any], ...] | None = None,
    cpu_policies: tuple[tuple[str, int], ...] | None = None,
    repeat: int = 5,
    warmup: int = 2,
    seed: int = 0,
    collect_diagnostics: bool = True,
) -> list[dict[str, Any]]:
    """Runs every ``(case, shape, cpu policy)`` combination in the corpus.

    Args:
        cases: Benchmark cases; defaults to :data:`BENCHMARK_CORPUS`.
        cpu_policies: ``(label, num_threads)`` pairs; defaults to :data:`CPU_POLICIES`.

    Returns:
        A flat list of result rows; see :func:`run_kernel_baseline_report` for
        the combined report schema.
    """
    from onnx_light.onnx import TensorProto

    if cases is None:
        cases = BENCHMARK_CORPUS
    if cpu_policies is None:
        cpu_policies = CPU_POLICIES

    results = []
    for case in cases:
        elem_type = int(getattr(TensorProto, case["element_type"]))
        for shape_label, size in case["shapes"]:
            model = _make_model(case, size)
            inputs = _make_inputs(case, size, seed)
            for policy_label, num_threads in cpu_policies:
                measurement = _run_policy_case(
                    model,
                    inputs,
                    elem_type=elem_type,
                    num_threads=num_threads,
                    repeat=repeat,
                    warmup=warmup,
                    collect_diagnostics=collect_diagnostics,
                )
                results.append(
                    {
                        "op_type": case["op_type"],
                        "element_type": case["element_type"],
                        "shape_label": shape_label,
                        "size": size,
                        "cpu_policy": policy_label,
                        **measurement,
                    }
                )
    return results


def run_kernel_baseline_report(
    *,
    cases: tuple[dict[str, Any], ...] | None = None,
    cpu_policies: tuple[tuple[str, int], ...] | None = None,
    repeat: int = 5,
    warmup: int = 2,
    seed: int = 0,
    collect_diagnostics: bool = True,
) -> dict[str, Any]:
    """Produces the combined Step D + Step E machine-readable report.

    It does not modify the kernel tuning cache and does not invoke
    ``onnxruntime``, so only native ``onnx-light`` kernel execution enters the
    report.

    Args:
        cases: Benchmark cases; defaults to :data:`BENCHMARK_CORPUS`.
        cpu_policies: ``(label, num_threads)`` pairs; defaults to :data:`CPU_POLICIES`.

    Returns:
        A mapping with ``cpu_descriptor``, ``inventory`` (Step D rows) and
        ``benchmarks`` (Step E rows).
    """
    inventory = kernel_inventory.build_kernel_inventory()
    kernel_inventory.validate_inventory(inventory)
    benchmarks = run_benchmark_corpus(
        cases=cases,
        cpu_policies=cpu_policies,
        repeat=repeat,
        warmup=warmup,
        seed=seed,
        collect_diagnostics=collect_diagnostics,
    )
    return {
        "cpu_descriptor": get_cpu_descriptor(),
        "inventory": inventory,
        "benchmarks": benchmarks,
    }
