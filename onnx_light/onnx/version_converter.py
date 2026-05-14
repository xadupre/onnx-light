# source: https://github.com/onnx/onnx/blob/main/onnx/version_converter/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from .onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

if TYPE_CHECKING:
    from . import ModelProto

_vc = _C.version_converter
ConvertError = _vc.ConvertError


def convert_version(model: ModelProto, target_version: int) -> ModelProto:
    """Converts the model to the given target opset version.

    Args:
        model: The input ModelProto to convert.
        target_version: The target opset version for the ``ai.onnx`` domain.

    Returns:
        A new ModelProto converted to the target opset version.

    Raises:
        ConvertError: If the conversion cannot be completed.
    """
    return _vc.convert_version(model, target_version)
