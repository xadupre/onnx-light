# source: https://github.com/onnx/onnx-light/onnx-light/blob/main/onnx/parser.py
# source: https://github.com/onnx/onnx/blob/main/onnx/parser.py
"""Parser helpers for onnx_light."""

from __future__ import annotations

from ..onnx_proto import _onnxpy as _C  # type: ignore[missing-module-attribute]

_parser = _C.parser


def parse_model(text: str) -> _C.ModelProto:
    """Parses an ONNX model from its text representation.

    Returns:
        The parsed :class:`ModelProto`.

    Raises:
        ValueError: If parsing fails.
    """
    ok, msg, proto = _parser.parse_model(text)
    if not ok:
        raise ValueError(f"Failed to parse model: {msg}")
    return proto


def parse_graph(text: str) -> _C.GraphProto:
    """Parses an ONNX graph from its text representation.

    Returns:
        The parsed :class:`GraphProto`.

    Raises:
        ValueError: If parsing fails.
    """
    ok, msg, proto = _parser.parse_graph(text)
    if not ok:
        raise ValueError(f"Failed to parse graph: {msg}")
    return proto


def parse_function(text: str) -> _C.FunctionProto:
    """Parses an ONNX function from its text representation.

    Returns:
        The parsed :class:`FunctionProto`.

    Raises:
        ValueError: If parsing fails.
    """
    ok, msg, proto = _parser.parse_function(text)
    if not ok:
        raise ValueError(f"Failed to parse function: {msg}")
    return proto


def parse_node(text: str) -> _C.NodeProto:
    """Parses an ONNX node from its text representation.

    Returns:
        The parsed :class:`NodeProto`.

    Raises:
        ValueError: If parsing fails.
    """
    ok, msg, proto = _parser.parse_node(text)
    if not ok:
        raise ValueError(f"Failed to parse node: {msg}")
    return proto
