# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Implements isolated workers for the backend command."""

from typing import Any


def _backend_test_map_to_dict(value: Any) -> dict[Any, Any]:
    """Converts a backend-test map value to a Python dictionary."""
    from onnx_light.onnx_lib.backend.test.case.base import _tensor_to_np

    keys = _tensor_to_np(value.keys)
    values = _tensor_to_np(value.values)
    return dict(zip(keys.tolist(), values.tolist()))


def _backend_test_data_set_feed(data_set: Any) -> dict[str, Any]:
    """Builds a name-keyed evaluator feed from one backend-test data set."""
    feed = {tensor.name: tensor for tensor in data_set.inputs}
    feed.update({value.name: _backend_test_map_to_dict(value) for value in data_set.maps})
    return feed


def _measure_backend_test_case(case: Any, repeat: int, warmup: int) -> dict[str, Any]:
    """Measures one materialized backend test case."""
    import statistics
    import time

    from onnx_light.onnx.reference import ReferenceEvaluator

    materialization_start = time.perf_counter()
    model = case.model
    data_sets = list(case.data_sets)
    feeds = [_backend_test_data_set_feed(data_set) for data_set in data_sets]
    materialization_seconds = time.perf_counter() - materialization_start

    setup_start = time.perf_counter()
    evaluator = ReferenceEvaluator(model)
    setup_seconds = time.perf_counter() - setup_start

    warmup_start = time.perf_counter()
    for _ in range(warmup):
        for feed in feeds:
            evaluator.run(None, feed)
    warmup_seconds = time.perf_counter() - warmup_start

    iteration_seconds = []
    for _ in range(repeat):
        iteration_start = time.perf_counter()
        for feed in feeds:
            evaluator.run(None, feed)
        iteration_seconds.append(time.perf_counter() - iteration_start)

    return {
        "name": case.name,
        "kind": case.kind,
        "tag": case.tag,
        "status": "completed",
        "timed_out": False,
        "error": None,
        "data_sets": len(data_sets),
        "materialization_seconds": materialization_seconds,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "iteration_seconds": iteration_seconds,
        "run_seconds": sum(iteration_seconds),
        "mean_seconds": statistics.fmean(iteration_seconds),
        "min_seconds": min(iteration_seconds),
        "max_seconds": max(iteration_seconds),
    }


def initialize_backend_test_worker() -> None:
    """Loads backend dependencies before timed work starts."""
    from onnx_light.onnx import backend  # noqa: F401


def backend_test_worker_ready() -> bool:
    """Confirms that the backend worker finished initializing."""
    return True


def measure_backend_test_case_by_name(
    case_name: str, mode: str, include_big: bool, repeat: int, warmup: int
) -> dict[str, Any]:
    """Collects and measures one backend test case."""
    import re

    from onnx_light.onnx import backend

    test_mode = backend.TestMode.TEST if mode == "test" else backend.TestMode.BENCHMARK
    cases = backend.collect_test_cases_by_name(
        f"^{re.escape(case_name)}$", include_big=include_big, mode=test_mode
    )
    if len(cases) != 1:
        raise RuntimeError(f"expected one backend case named {case_name!r}, got {len(cases)}")
    return _measure_backend_test_case(cases[0], repeat, warmup)
