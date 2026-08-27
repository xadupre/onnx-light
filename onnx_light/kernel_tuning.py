# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Inspects, calibrates, and updates processor-aware kernel tuning values."""

from __future__ import annotations

from typing import Any, Sequence

from onnx_light.onnx_py._onnxpykernels.runtime import (  # type: ignore[import]
    analyze_kernel_tuning_latencies,
    calibrate_kernel_tuning,
    default_kernel_tuning_cache_path,
    inspect_kernel_tuning_cache,
    kernel_tuning_parameters,
    load_kernel_tuning_cache,
    set_kernel_tuning_parameters,
)


def propose_kernel_tuning_updates(
    *,
    kernels: Sequence[str] | None = None,
    element_types: Sequence[int] | None = None,
    library: str = "onnx_light",
    implementation: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Proposes calibrations for exact kernel keys missing from the local cache.

    Args:
        kernels: Optional kernel-name subset.
        element_types: Optional ONNX element-type subset.
        library: Tuning library identifier.
        implementation: Optional implementation identifier.
        path: Optional cache path.

    Returns:
        A report separating calibratable and unsupported missing keys.
    """
    report = kernel_tuning_parameters(library=library, implementation=implementation, path=path)
    kernel_filter = set(kernels or ())
    element_type_filter = set(element_types or ())
    selected = [
        item
        for item in report["kernels"]
        if (not kernel_filter or item["kernel"] in kernel_filter)
        and (not element_type_filter or item["element_type"] in element_type_filter)
    ]
    missing = [item for item in selected if item["cached_values"] is None]
    calibratable = [item for item in missing if item["calibratable"]]
    unsupported = [item for item in missing if not item["calibratable"]]
    return {
        "cache_status": report["cache_status"],
        "cache_path": report["cache_path"],
        "diagnostics": report["diagnostics"],
        "selected": len(selected),
        "covered": len(selected) - len(missing),
        "missing": missing,
        "calibratable": calibratable,
        "unsupported": unsupported,
    }


def apply_kernel_tuning_updates(
    *,
    kernels: Sequence[str] | None = None,
    element_types: Sequence[int] | None = None,
    library: str = "onnx_light",
    implementation: str | None = None,
    path: str | None = None,
    maximum_duration_ms: int = 0,
    maximum_memory_bytes: int = 0,
) -> dict[str, Any]:
    """Calibrates and persists every calibratable key missing from the cache.

    Returns:
        The initial proposal, calibration reports, and remaining coverage gaps.
    """
    proposal = propose_kernel_tuning_updates(
        kernels=kernels,
        element_types=element_types,
        library=library,
        implementation=implementation,
        path=path,
    )
    updates = []
    for item in proposal["calibratable"]:
        updates.append(
            calibrate_kernel_tuning(
                item["kernel"],
                element_types=[item["element_type"]],
                library=item["library"],
                implementation=item["implementation"],
                maximum_duration_ms=maximum_duration_ms,
                maximum_memory_bytes=maximum_memory_bytes,
                save=True,
                path=path,
            )
        )
    remaining = propose_kernel_tuning_updates(
        kernels=kernels,
        element_types=element_types,
        library=library,
        implementation=implementation,
        path=path,
    )
    return {"proposal": proposal, "updates": updates, "remaining": remaining}


__all__ = [
    "analyze_kernel_tuning_latencies",
    "apply_kernel_tuning_updates",
    "calibrate_kernel_tuning",
    "default_kernel_tuning_cache_path",
    "inspect_kernel_tuning_cache",
    "kernel_tuning_parameters",
    "load_kernel_tuning_cache",
    "propose_kernel_tuning_updates",
    "set_kernel_tuning_parameters",
]
