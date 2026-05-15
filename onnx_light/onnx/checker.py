# source: https://github.com/onnx/onnx/blob/main/onnx/checker.py
"""Checker helpers for onnx_light."""

from __future__ import annotations

from .onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

_checker = _C.checker

#: Raised when a model or proto fails validation.
ValidationError = _checker.ValidationError


def check_model(model: _C.ModelProto) -> None:
    """Checks that a ModelProto is valid.

    Args:
        model: The model to validate.

    Raises:
        ValidationError: If the model fails validation.
    """
    from . import pychecker as _pychecker

    try:
        _pychecker.check_model(model)
    except _pychecker.ValidationError as exc:
        raise ValidationError(str(exc)) from exc


def check_function_call_cycles(model: _C.ModelProto) -> None:
    """Checks for cycles in model-local function call graph.

    Raises:
        ValidationError: If the model contains cyclic function references.
    """
    _checker.check_function_call_cycles(model)
