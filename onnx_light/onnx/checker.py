# source: https://github.com/onnx/onnx/blob/main/onnx/checker.py
"""Checker helpers for onnx_light."""

from __future__ import annotations

from .onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

_checker = _C.checker

#: Raised when a model or proto fails validation.
ValidationError = _checker.ValidationError


def check_model(model: _C.ModelProto) -> None:
    """Checks a model and raises checker.ValidationError on invalid content.

    Returns:
        None.
    """
    _checker.check_model(model)


def check_function_call_cycles(model: _C.ModelProto) -> None:
    """Checks for cycles in model-local function call graph.

    Raises:
        ValidationError: If the model contains cyclic function references.
    """
    _checker.check_function_call_cycles(model)
