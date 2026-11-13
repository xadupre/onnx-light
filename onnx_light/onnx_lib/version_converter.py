# source : https: // github.com/onnx/onnx/blob/main/onnx/version_converter/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from ..onnx_py import _onnxpyprotolib as _C  # type: ignore

if TYPE_CHECKING:
    from . import ModelProto

_vc = _C.version_converter  # type: ignore
ConvertError = _vc.ConvertError


def convert_version(model: ModelProto, target_version: int) -> ModelProto:
    """Converts the model to the given target opset version.

    Retains local functions and converts their bodies from their own default-domain
    opset imports. Preserves references to function attributes.

    Args:
        model: The input ModelProto to convert.
        target_version: The target opset version for the ``ai.onnx`` domain.

    Returns:
        A new ModelProto converted to the target opset version.

    Raises:
        ConvertError: If the conversion cannot be completed, including when any
            graph contains unsupported sparse initializers.
        RuntimeError: If an adapter requires a concrete value for an unresolved
            function attribute or unavailable type or shape information.
    """
    return _vc.convert_version(model, target_version)
