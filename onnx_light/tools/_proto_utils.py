"""Shared, duck-typed helpers for inspecting ONNX protobuf messages.

These helpers are used by the rendering tools in this sub-package
(:mod:`onnx_light.tools.mermaid` and :mod:`onnx_light.tools.svg`).  They
only rely on the attributes of the standard ONNX message types
(``ModelProto``, ``GraphProto``, ``NodeProto``, ``ValueInfoProto``,
``TensorProto`` and ``TensorShapeProto``) so they work both with messages
built by :mod:`onnx_light` and with messages built by the upstream
:mod:`onnx` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only.
    from collections.abc import Iterable


def _s(value: Any) -> str:
    """Returns ``value`` as a regular Python :class:`str`.

    ``onnx_light`` exposes ``utils::String`` objects through the C++
    extension and ``onnx`` returns plain :class:`str` instances.  This
    helper accepts both.
    """

    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def _format_shape(type_proto: Any) -> str:
    """Returns a compact textual representation of a ``TypeProto``.

    Examples: ``float[1,3,224,224]``, ``int64[?]``, ``seq(float[N])``.
    Returns an empty string when no useful information can be extracted.
    """

    if type_proto is None:
        return ""

    # TensorType
    tensor_type = getattr(type_proto, "tensor_type", None)
    if tensor_type is not None and getattr(tensor_type, "elem_type", None):
        dtype = _dtype_name(int(tensor_type.elem_type))
        shape = _format_tensor_shape(getattr(tensor_type, "shape", None))
        return f"{dtype}[{shape}]" if shape else dtype

    # SequenceType
    seq_type = getattr(type_proto, "sequence_type", None)
    if seq_type is not None and getattr(seq_type, "elem_type", None) is not None:
        inner = _format_shape(seq_type.elem_type)
        return f"seq({inner})" if inner else "seq"

    # OptionalType
    opt_type = getattr(type_proto, "optional_type", None)
    if opt_type is not None and getattr(opt_type, "elem_type", None) is not None:
        inner = _format_shape(opt_type.elem_type)
        return f"optional({inner})" if inner else "optional"

    # MapType
    map_type = getattr(type_proto, "map_type", None)
    if map_type is not None and getattr(map_type, "key_type", None):
        key = _dtype_name(int(map_type.key_type))
        value = _format_shape(getattr(map_type, "value_type", None))
        return f"map({key}, {value})" if value else f"map({key})"

    return ""


def _format_tensor_shape(shape: Any) -> str:
    """Returns a comma-separated string of a ``TensorShapeProto``'s dims."""
    if shape is None:
        return ""
    dims = getattr(shape, "dim", None)
    if dims is None:
        return ""
    parts: list[str] = []
    for d in dims:
        # A TensorShapeProto.Dimension has either dim_value or dim_param.
        dim_value = getattr(d, "dim_value", 0)
        dim_param = _s(getattr(d, "dim_param", ""))
        if dim_param:
            parts.append(dim_param)
        elif dim_value:
            parts.append(str(int(dim_value)))
        else:
            parts.append("?")
    return ",".join(parts)


# Mirrors ``onnx.TensorProto.DataType`` names without importing ``onnx``.
_DTYPE_NAMES = {
    0: "UNDEFINED",
    1: "float",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
    17: "float8e4m3fn",
    18: "float8e4m3fnuz",
    19: "float8e5m2",
    20: "float8e5m2fnuz",
    21: "uint4",
    22: "int4",
    23: "float4e2m1",
}


def _dtype_name(elem_type: int) -> str:
    """Returns the textual name of a ``TensorProto.DataType`` value."""
    return _DTYPE_NAMES.get(int(elem_type), f"dtype{int(elem_type)}")


def _iter(seq: Any) -> Iterable[Any]:
    """Returns ``seq`` itself, or an empty tuple when it is ``None``."""
    if seq is None:
        return ()
    return seq


def _extract_graph(model_or_graph: Any) -> Any:
    """Returns ``model_or_graph.graph`` when given a ``ModelProto``."""

    if _looks_like_graph(model_or_graph):
        return model_or_graph
    graph = getattr(model_or_graph, "graph", None)
    if graph is not None and _looks_like_graph(graph):
        return graph
    raise TypeError(f"expected a ModelProto or GraphProto, got {type(model_or_graph).__name__}.")


def _looks_like_graph(obj: Any) -> bool:
    """Returns :data:`True` when ``obj`` quacks like a ``GraphProto``."""
    return (
        hasattr(obj, "node")
        and hasattr(obj, "input")
        and hasattr(obj, "output")
        and not hasattr(obj, "graph")
    )
