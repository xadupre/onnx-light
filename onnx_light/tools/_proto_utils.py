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

import importlib.util
import contextlib
from typing import TYPE_CHECKING, Any

_shape_inference: Any = None

if importlib.util.find_spec("onnx_light.onnx_py._onnxpycore") is not None:
    with contextlib.suppress(ImportError):  # pragma: no cover
        from ..onnx_core import shape_inference as _shape_inference

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


def _short_display_name(value: Any, max_length: int = 15) -> str:
    """Returns a display name, shortening long names to their trailing characters.

    Long names are reduced to their trailing characters so graph renderings stay
    readable while still exposing the most distinguishing suffix.

    Parameters:
        value: The original value name to shorten for display.
        max_length: The maximum number of trailing characters to keep.

    Returns:
        The original name when it already fits, otherwise its trailing
        ``max_length`` characters.
    """

    text = _s(value)
    if len(text) <= max_length:
        return text
    return text[-max_length:]


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


def _node_metadata_value(node: Any, key: str) -> str:
    """Returns the value of a node's ``metadata_props`` entry, or ``""``."""
    for entry in _iter(getattr(node, "metadata_props", ())):
        if _s(getattr(entry, "key", "")) == key:
            return _s(getattr(entry, "value", ""))
    return ""


# Metadata keys written by tooling:
# - per-value tag -> VALUE_TAG_METADATA_KEY
# - per-node tag -> NODE_TAG_METADATA_KEY
# - per-node in-place reuse map -> INPLACE_REUSE_METADATA_KEY
# - per-node release-after tensor list -> RELEASE_AFTER_METADATA_KEY
# - per-node last-use inputs/initializers list -> NOT_USED_AFTER_METADATA_KEY
# - per-node shape-tagged release-after list -> RELEASE_AFTER_SHAPE_TAG_METADATA_KEY
VALUE_TAG_METADATA_KEY = "onnx_light.value_tag"
VALUE_TAGS_METADATA_KEY = "onnx_light.value_tags"
NODE_TAG_METADATA_KEY = "onnx_light.node_tag"
INPLACE_REUSE_METADATA_KEY = "onnx_light.inplace_reuse"
RELEASE_AFTER_METADATA_KEY = "onnx_light.release_after"
NOT_USED_AFTER_METADATA_KEY = "onnx_light.not_used_after"
RELEASE_AFTER_SHAPE_TAG_METADATA_KEY = "onnx_light.release_after_shape_tag"
VALUE_TAGS = {"shape", "axes", "weight", "ambiguous"}
VALUE_TAG_COLORS = {
    "shape": {"fill": "#f4d6ff", "stroke": "#8744a2"},
    "axes": {"fill": "#ffe9a8", "stroke": "#9e7a00"},
    "weight": {"fill": "#e0e0e0", "stroke": "#666666"},
    "ambiguous": {"fill": "#ffd9d9", "stroke": "#a33a3a"},
}


def _normalise_value_tag(value: str) -> str:
    value = _s(value).strip().lower()
    return value if value in VALUE_TAGS else ""


def _require_shape_inference_extension() -> Any:
    """Gets the shape inference nanobind module.

    Returns:
        The imported ``onnx_light.onnx_core.shape_inference`` module.
    """
    if _shape_inference is None:
        raise RuntimeError(
            "onnx_light.onnx_py._onnxpycore is unavailable, so "
            "onnx_light.onnx_core.shape_inference cannot be used. "
            "Install the onnx_light C++ extension to use value/node tag inference."
        )
    return _shape_inference


def compute_value_and_node_tags(
    graph_or_nodes_or_function: Any, verbose: int = 0
) -> tuple[dict[str, str], list[str]]:
    """Infers semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for values and nodes.

    Returns:
        A pair ``(value_tags, node_tags)`` where ``value_tags`` maps value
        names to tags and ``node_tags`` is ordered like the processed node list.
    """
    return _require_shape_inference_extension().compute_value_and_node_tags(
        graph_or_nodes_or_function, verbose=verbose
    )


infer_value_and_node_tags = compute_value_and_node_tags


def write_value_and_node_tags_to_metadata(graph_or_nodes_or_function: Any) -> None:
    """Writes inferred ``shape``/``axes``/``weight``/``ambiguous`` tags into metadata.

    Returns:
        ``None``. The function mutates metadata fields in place.
    """
    _require_shape_inference_extension().write_value_and_node_tags_to_metadata(
        graph_or_nodes_or_function
    )


def _format_inplace_reuse(node: Any) -> str:
    """Returns a compact description of a node's in-place reuse opportunities.

    Parses the ``onnx_light.inplace_reuse`` metadata entry written by the
    in-place reuse analysis and renders it as a human-readable string such
    as ``inplace: out0=in1(equal)``.  Returns an empty string when the node
    carries no such metadata.
    """

    raw = _node_metadata_value(node, INPLACE_REUSE_METADATA_KEY)
    if not raw:
        return ""
    parts: list[str] = []
    for triplet in raw.split(";"):
        triplet = triplet.strip()
        if not triplet:
            continue
        fields = triplet.split(":")
        if len(fields) < 2:
            continue
        out_index = fields[0].strip()
        in_index = fields[1].strip()
        kind = fields[2].strip() if len(fields) > 2 else ""
        label = f"out{out_index}=in{in_index}"
        if kind:
            label = f"{label}({kind})"
        parts.append(label)
    if not parts:
        return ""
    return "inplace: " + ", ".join(parts)


def _format_release_after(node: Any) -> str:
    """Returns a compact description of a node's post-execution release hints.

    Parses the ``onnx_light.release_after`` metadata entry written by the
    in-place reuse analysis and renders it as a human-readable string such
    as ``release: A, B``. When present, the
    ``onnx_light.not_used_after`` metadata entry is appended as
    ``not used after: X, W``. Returns an empty string when the node carries
    neither metadata key.
    """
    parts: list[str] = []
    raw_release = _node_metadata_value(node, RELEASE_AFTER_METADATA_KEY)
    names_release = [name.strip() for name in raw_release.split(";") if name.strip()]
    if names_release:
        parts.append("release: " + ", ".join(names_release))
    raw_not_used = _node_metadata_value(node, NOT_USED_AFTER_METADATA_KEY)
    names_not_used = [name.strip() for name in raw_not_used.split(";") if name.strip()]
    if names_not_used:
        parts.append("not used after: " + ", ".join(names_not_used))
    return "; ".join(parts)


def _graph_value_tags(graph: Any) -> dict[str, str]:
    """Collects value tags from graph/value metadata.

    Returns:
        A dictionary mapping value names to normalized tags.
    """
    tags: dict[str, str] = {}
    for collection in ("input", "value_info", "output", "initializer"):
        for value in _iter(getattr(graph, collection, ())):
            name = _s(getattr(value, "name", ""))
            if not name:
                continue
            tag = _normalise_value_tag(_node_metadata_value(value, VALUE_TAG_METADATA_KEY))
            if tag:
                tags[name] = tag
    return tags


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
