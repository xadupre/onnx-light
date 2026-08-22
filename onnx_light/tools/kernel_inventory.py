# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Enumerates every registered native kernel path and classifies its coverage.

This module implements Step D of
:ref:`l-next-steps-kernel-parallelization`: it builds one inventory row per
``(domain, op_type, device, element_type, implementation)`` path covering
every operator registered in the built-in
``onnx_kernels::BuiltinKernelFunctions`` dispatch table
(``onnx_light/onnx_extensions/kernels/kernel_dispatch_table.cc``).

Each row is classified into exactly one coverage state:

``serial``
    No ``ParallelFor`` call site was found in the kernel's implementation file
    and no tuning schema is registered for it.
``parallel_fixed_policy``
    The kernel calls ``ParallelFor`` but has no registered tuning schema, so
    its grain size and participant limits are compiled constants.
``tunable``
    The kernel registers a ``KernelTuningSchema`` (a portable default that may
    be overridden by a persisted profile).
``calibratable``
    A ``tunable`` path that additionally registers a
    ``KernelCalibrationFunction``.

Building the inventory only reads static kernel source files and the
in-process tuning registry (via
:func:`onnx_light.kernel_tuning.kernel_tuning_parameters`); it never mutates
the kernel tuning cache.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DISPATCH_TABLE_PATH = (
    _REPO_ROOT / "onnx_light" / "onnx_extensions" / "kernels" / "kernel_dispatch_table.cc"
)
_DEFAULT_KERNEL_SOURCE_ROOT = (
    _REPO_ROOT / "onnx_light" / "onnx_extensions" / "kernels" / "kernels"
)

# Matches one dispatch-table entry, e.g.:
#   {"ai.onnx:Abs", MakeKernel<onnx_kernels::kernel::Abs>()},
_DISPATCH_ENTRY_RE = re.compile(r'\{"([^"]+)",\s*MakeKernel<onnx_kernels::kernel::(\w+)>\(\)\}')
_PARALLEL_FOR_RE = re.compile(r"\bParallelFor\s*[(<]")

__all__ = [
    "build_kernel_inventory",
    "dispatch_table_entries",
    "validate_inventory",
]


def dispatch_table_entries(
    dispatch_table_path: Path | str | None = None,
) -> list[tuple[str, str, str]]:
    """Parses the built-in kernel dispatch table into ``(domain, op_type, class_name)`` rows.

    Args:
        dispatch_table_path: Optional override of the ``kernel_dispatch_table.cc`` path,
            mainly for tests.

    Returns:
        One entry per registered ``(domain, op_type)`` identifier, in file order.
    """
    path = Path(dispatch_table_path) if dispatch_table_path is not None else (
        _DEFAULT_DISPATCH_TABLE_PATH
    )
    text = path.read_text(encoding="utf-8")
    entries = []
    for identifier, class_name in _DISPATCH_ENTRY_RE.findall(text):
        domain, _, op_type = identifier.partition(":")
        entries.append((domain, op_type, class_name))
    return entries


def _kernel_source_files(kernel_source_root: Path) -> list[Path]:
    return sorted(kernel_source_root.rglob("*.cc"))


def _find_owning_source(
    class_name: str, source_files: list[Path], cache: dict[str, Path | None]
) -> Path | None:
    """Returns the ``.cc`` file implementing ``class_name`` (``ClassName::`` marker)."""
    if class_name in cache:
        return cache[class_name]
    marker = re.compile(r"\b" + re.escape(class_name) + r"::")
    found = None
    for source_file in source_files:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        if marker.search(text):
            found = source_file
            break
    cache[class_name] = found
    return found


def _serial_reason(op_type: str, source_rel: str | None) -> str:
    if source_rel is None:
        return (
            f"no implementation file was found by static source scan for kernel "
            f"class backing '{op_type}'; classified as serial conservatively"
        )
    return (
        f"no ParallelFor call site found in {source_rel}; "
        f"'{op_type}' always executes on the calling thread"
    )


def build_kernel_inventory(
    *,
    library: str = "onnx_light",
    dispatch_table_path: Path | str | None = None,
    kernel_source_root: Path | str | None = None,
    tuning_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Builds the full kernel-path inventory (Step D).

    Args:
        library: Tuning library identifier passed to
            :func:`onnx_light.kernel_tuning.kernel_tuning_parameters` when
            ``tuning_report`` is not supplied.
        dispatch_table_path: Optional override of the dispatch-table source path.
        kernel_source_root: Optional override of the kernel implementation source root.
        tuning_report: Optional pre-fetched ``kernel_tuning_parameters()`` result, mainly
            for tests that must not depend on the compiled extension.

    Returns:
        A list of inventory rows, sorted by ``(domain, op_type, element_type,
        implementation)``. Every registered kernel path appears exactly once.
    """
    entries = dispatch_table_entries(dispatch_table_path)
    source_root = Path(kernel_source_root) if kernel_source_root is not None else (
        _DEFAULT_KERNEL_SOURCE_ROOT
    )
    source_files = _kernel_source_files(source_root)
    owner_cache: dict[str, Path | None] = {}

    if tuning_report is None:
        from onnx_light.kernel_tuning import kernel_tuning_parameters

        tuning_report = kernel_tuning_parameters(library=library)

    tuning_by_op_type: dict[str, list[dict[str, Any]]] = {}
    for item in tuning_report["kernels"]:
        tuning_by_op_type.setdefault(item["kernel"], []).append(item)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int | None, str]] = set()
    for domain, op_type, class_name in entries:
        source = _find_owning_source(class_name, source_files, owner_cache)
        source_rel = str(source.relative_to(_REPO_ROOT)) if source is not None else None
        uses_parallel_for = False
        if source is not None:
            text = source.read_text(encoding="utf-8", errors="ignore")
            uses_parallel_for = bool(_PARALLEL_FOR_RE.search(text))

        tuning_entries = tuning_by_op_type.get(op_type)
        if tuning_entries:
            for item in tuning_entries:
                key = (domain, op_type, "CPU", item["element_type"], item["implementation"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "library": item["library"],
                        "domain": domain,
                        "op_type": op_type,
                        "device": "CPU",
                        "element_type": item["element_type"],
                        "implementation": item["implementation"],
                        "tuning_abi": item["tuning_abi"],
                        "coverage_state": "calibratable" if item["calibratable"] else "tunable",
                        "uses_parallel_for": uses_parallel_for,
                        "source_file": source_rel,
                        "serial_reason": None,
                        "parameter_names": item["parameter_names"],
                    }
                )
        else:
            key = (domain, op_type, "CPU", None, "portable")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "library": library,
                    "domain": domain,
                    "op_type": op_type,
                    "device": "CPU",
                    "element_type": None,
                    "implementation": "portable",
                    "tuning_abi": None,
                    "coverage_state": "parallel_fixed_policy"
                    if uses_parallel_for
                    else "serial",
                    "uses_parallel_for": uses_parallel_for,
                    "source_file": source_rel,
                    "serial_reason": None
                    if uses_parallel_for
                    else _serial_reason(op_type, source_rel),
                    "parameter_names": [],
                }
            )

    rows.sort(
        key=lambda row: (
            row["domain"],
            row["op_type"],
            -1 if row["element_type"] is None else row["element_type"],
            row["implementation"],
        )
    )
    return rows


def validate_inventory(rows: list[dict[str, Any]]) -> None:
    """Raises ``ValueError`` when a kernel path is missing or duplicated.

    Args:
        rows: The result of :func:`build_kernel_inventory`.

    Raises:
        ValueError: When two rows share the same
            ``(domain, op_type, device, element_type, implementation)`` identity,
            or when a row is missing an explicit ``coverage_state``.
    """
    seen: set[tuple[str, str, str, int | None, str]] = set()
    for row in rows:
        key = (
            row["domain"],
            row["op_type"],
            row["device"],
            row["element_type"],
            row["implementation"],
        )
        if key in seen:
            raise ValueError(f"Duplicate kernel path in inventory: {key!r}")
        seen.add(key)
        if row.get("coverage_state") not in (
            "serial",
            "parallel_fixed_policy",
            "tunable",
            "calibratable",
        ):
            raise ValueError(f"Kernel path {key!r} has no explicit coverage state.")
