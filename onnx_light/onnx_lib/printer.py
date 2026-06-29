"""ONNX text-format printer, mirroring :mod:`onnx.printer`.

Provides :func:`to_text` which converts a protobuf object (ModelProto,
GraphProto, or FunctionProto) to its compact ONNX textual representation.
"""

from __future__ import annotations

from ..onnx_py import _onnxpyprotolib as _C  # type: ignore
from ._proto_compat import coerce_proto, matches_proto_class

from . import FunctionProto, GraphProto, ModelProto


def to_text(proto: ModelProto | FunctionProto | GraphProto) -> str:
    """Converts a protobuf object to the ONNX textual representation.

    Args:
        proto: A ModelProto, FunctionProto, or GraphProto instance.

    Returns:
        The ONNX textual representation of the proto.

    Raises:
        TypeError: If *proto* is not a supported type.
    """
    if matches_proto_class(proto, ModelProto):
        return _C.printer.model_to_text(coerce_proto(proto, ModelProto))
    if matches_proto_class(proto, FunctionProto):
        return _C.printer.function_to_text(coerce_proto(proto, FunctionProto))
    if matches_proto_class(proto, GraphProto):
        return _C.printer.graph_to_text(coerce_proto(proto, GraphProto))
    raise TypeError(
        f"Unsupported argument type: {type(proto).__name__}. "
        "Expected ModelProto, FunctionProto, or GraphProto."
    )
