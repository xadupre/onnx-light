# source: https://github.com/onnx/onnx/blob/main/onnx/inliner.py
"""Inliner helpers for onnx_light."""

from __future__ import annotations

from typing import Any, Sequence

from ..onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

__all__ = [
    "inline_local_functions",
    "inline_selected_functions",
]

_inliner = _C.inliner  # type: ignore
# UntypedModelProto aliases _onnxpy.ModelProto, which is runtime-only for type checkers.
UntypedModelProto = Any


def inline_local_functions(
    model: UntypedModelProto, convert_version: bool = False
) -> UntypedModelProto:
    """Inlines all model-local functions in the given model.

    Returns:
        A new :class:`ModelProto` with all model-local functions inlined.

    Raises:
        checker.ValidationError: If the model contains cyclic function references.
    """
    return _inliner.inline_local_functions(model, convert_version)


def inline_selected_functions(
    model: UntypedModelProto,
    functions: Sequence[tuple[str, str]],
    *,
    exclude: bool = False,
    inline_schema_functions: bool = False,
) -> UntypedModelProto:
    """Inlines the selected functions in the given model.

    Args:
        model: The model in which functions will be inlined.
        functions: A sequence of ``(domain, name)`` pairs identifying the functions
            to inline (or to exclude when *exclude* is True).
        exclude: When False (default), only the listed functions are inlined.
            When True, all functions **except** the listed ones are inlined.
        inline_schema_functions: When True, schema-defined functions (e.g. Softsign)
            are also eligible for inlining in addition to model-local functions.

    Returns:
        A new :class:`ModelProto` with the selected functions inlined.
    """
    function_ids = list(functions)
    if inline_schema_functions:
        return _inliner.inline_selected_functions(model, function_ids, exclude)
    return _inliner.inline_selected_local_functions(model, function_ids, exclude)
