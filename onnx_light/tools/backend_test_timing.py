# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Measures selected C++ backend test cases with the onnx-light runtime."""

from __future__ import annotations

import statistics
import time
from typing import Any

__all__ = ["run_backend_test_timing"]


def _map_to_dict(value: Any) -> dict[Any, Any]:
    """Converts a backend-test map value to a Python dictionary."""
    from onnx_light.onnx_lib.backend.test.case.base import _tensor_to_np

    keys = _tensor_to_np(value.keys)
    values = _tensor_to_np(value.values)
    return dict(zip(keys.tolist(), values.tolist()))


def _data_set_feed(data_set: Any) -> dict[str, Any]:
    """Builds a name-keyed evaluator feed from one backend-test data set."""
    feed = {tensor.name: tensor for tensor in data_set.inputs}
    feed.update({value.name: _map_to_dict(value) for value in data_set.maps})
    return feed


def run_backend_test_timing(
    *,
    name_regex: str = "",
    mode: str = "test",
    include_big: bool = False,
    repeat: int = 1,
    warmup: int = 0,
) -> dict[str, Any]:
    """Measures backend test cases selected by name and generation mode.

    Args:
        name_regex: ECMAScript regular expression searched in test case names.
        mode: Case generation mode, either ``"test"`` or ``"benchmark"``.
        include_big: Whether cases whose name contains ``"_big_"`` are included.
        repeat: Number of measured iterations per case.
        warmup: Number of unmeasured iterations per case.

    Returns:
        A report containing discovery, setup, and per-case execution timings.
    """
    if mode not in {"test", "benchmark"}:
        raise ValueError(f"Unknown backend test mode {mode!r}; expected 'test' or 'benchmark'.")
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}.")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")

    from onnx_light.onnx import backend
    from onnx_light.onnx.reference import ReferenceEvaluator

    test_mode = backend.TestMode.TEST if mode == "test" else backend.TestMode.BENCHMARK
    total_start = time.perf_counter()
    collection_start = time.perf_counter()
    cases = backend.collect_test_cases_by_name(
        name_regex, include_big=include_big, mode=test_mode
    )
    collection_seconds = time.perf_counter() - collection_start

    case_reports = []
    for case in cases:
        materialization_start = time.perf_counter()
        model = case.model
        data_sets = list(case.data_sets)
        feeds = [_data_set_feed(data_set) for data_set in data_sets]
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

        case_reports.append(
            {
                "name": case.name,
                "kind": case.kind,
                "tag": case.tag,
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
        )

    return {
        "name_regex": name_regex,
        "mode": mode,
        "include_big": include_big,
        "repeat": repeat,
        "warmup": warmup,
        "selected": len(case_reports),
        "collection_seconds": collection_seconds,
        "cases": case_reports,
        "total_seconds": time.perf_counter() - total_start,
    }
