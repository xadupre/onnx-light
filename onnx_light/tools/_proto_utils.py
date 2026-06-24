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
import json
from typing import TYPE_CHECKING, Any

if importlib.util.find_spec("onnx_light.onnx_py._onnxpyoptim") is not None:
    from ..onnx_optim import shape_inference as _shape_inference
else:  # pragma: no cover
    _shape_inference = None

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


def _node_metadata_value(node: Any, key: str) -> str:
    """Returns the value of a node's ``metadata_props`` entry, or ``""``."""
    for entry in _iter(getattr(node, "metadata_props", ())):
        if _s(getattr(entry, "key", "")) == key:
            return _s(getattr(entry, "value", ""))
    return ""


def _set_metadata_value(obj: Any, key: str, value: str) -> None:
    """Sets a metadata key/value pair on an ONNX-like proto object.

    The helper first prefers ``obj.add_metadata(key, value)`` when this
    method exists. Otherwise it falls back to mutating ``obj.metadata_props``:
    an existing entry is updated in place, and when no entry exists a new one
    is appended. Objects without either API are ignored.
    """
    add_metadata = getattr(obj, "add_metadata", None)
    if callable(add_metadata):
        add_metadata(key, value)
        return
    entries = getattr(obj, "metadata_props", None)
    if entries is None:
        return
    for entry in entries:
        if _s(getattr(entry, "key", "")) == key:
            entry.value = value
            return
    entry_type = type(entries[0]) if entries else None
    if entry_type is None:

        class _Meta:
            def __init__(self, k: str, v: str):
                self.key = k
                self.value = v

        entries.append(_Meta(key, value))
    else:
        entries.append(entry_type(key=key, value=value))


# Metadata keys written by tooling:
# - per-value tag -> VALUE_TAG_METADATA_KEY
# - graph/function-level value-tag map -> VALUE_TAGS_METADATA_KEY
# - per-node tag -> NODE_TAG_METADATA_KEY
# - per-node in-place reuse map -> INPLACE_REUSE_METADATA_KEY
VALUE_TAG_METADATA_KEY = "onnx_light.value_tag"
VALUE_TAGS_METADATA_KEY = "onnx_light.value_tags"
NODE_TAG_METADATA_KEY = "onnx_light.node_tag"
INPLACE_REUSE_METADATA_KEY = "onnx_light.inplace_reuse"
VALUE_TAGS = {"shape", "axes", "weight"}
VALUE_TAG_COLORS = {
    "shape": {"fill": "#f4d6ff", "stroke": "#8744a2"},
    "axes": {"fill": "#ffe9a8", "stroke": "#9e7a00"},
    "weight": {"fill": "#e0e0e0", "stroke": "#666666"},
}


def _normalise_value_tag(value: str) -> str:
    value = _s(value).strip().lower()
    return value if value in VALUE_TAGS else ""


def _infer_value_and_node_tags_python(
    graph_or_nodes_or_function: Any,
) -> tuple[dict[str, str], list[str]]:
    """Infers semantic ``shape``/``axes``/``weight`` tags for values and nodes.

    Returns:
        A pair ``(value_tags, node_tags)`` where ``value_tags`` maps value
        names to tags and ``node_tags`` is ordered like the processed node list.
    """
    if hasattr(graph_or_nodes_or_function, "graph"):
        graph_or_nodes_or_function = graph_or_nodes_or_function.graph
    if hasattr(graph_or_nodes_or_function, "node"):
        nodes = list(_iter(getattr(graph_or_nodes_or_function, "node", ())))
        graph_like = graph_or_nodes_or_function
    else:
        nodes = list(_iter(graph_or_nodes_or_function))
        graph_like = None

    value_tags: dict[str, str] = {}
    node_tags: list[str] = []

    def set_value_tag(name: str, tag: str) -> None:
        if not name:
            return
        norm = _normalise_value_tag(tag)
        if norm:
            value_tags[name] = norm

    if graph_like is not None:
        for value_info in _iter(getattr(graph_like, "input", ())):
            name = _s(getattr(value_info, "name", ""))
            tag = _normalise_value_tag(_node_metadata_value(value_info, VALUE_TAG_METADATA_KEY))
            if tag:
                set_value_tag(name, tag)
        for init in _iter(getattr(graph_like, "initializer", ())):
            name = _s(getattr(init, "name", ""))
            tag = (
                _normalise_value_tag(_node_metadata_value(init, VALUE_TAG_METADATA_KEY))
                or "weight"
            )
            set_value_tag(name, tag)
        for value_info in _iter(getattr(graph_like, "value_info", ())):
            name = _s(getattr(value_info, "name", ""))
            tag = _normalise_value_tag(_node_metadata_value(value_info, VALUE_TAG_METADATA_KEY))
            if tag:
                set_value_tag(name, tag)
        for value_info in _iter(getattr(graph_like, "output", ())):
            name = _s(getattr(value_info, "name", ""))
            tag = _normalise_value_tag(_node_metadata_value(value_info, VALUE_TAG_METADATA_KEY))
            if tag:
                set_value_tag(name, tag)
        try:
            payload = json.loads(_node_metadata_value(graph_like, VALUE_TAGS_METADATA_KEY))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            for name, tag in payload.items():
                set_value_tag(_s(name), _s(tag))

    for node in nodes:
        op_type = _s(getattr(node, "op_type", ""))
        inputs = [_s(inp) for inp in _iter(getattr(node, "input", ()))]
        outputs = [_s(out) for out in _iter(getattr(node, "output", ()))]
        explicit_output_tag = ""

        if op_type in {"Shape", "Size"}:
            explicit_output_tag = "shape"
        elif op_type == "Constant":
            explicit_output_tag = "weight"

        if len(inputs) >= 2:
            if op_type in {"Reshape", "Expand", "Slice"}:
                set_value_tag(inputs[1], "shape")
            elif op_type in {
                "Squeeze",
                "Unsqueeze",
                "ReduceSum",
                "ReduceMean",
                "ReduceMax",
                "ReduceMin",
            }:
                set_value_tag(inputs[1], "axes")
        if op_type == "Slice":
            for idx, tag in ((2, "shape"), (3, "axes"), (4, "shape")):
                if idx < len(inputs):
                    set_value_tag(inputs[idx], tag)

        inherited = ""
        if inputs:
            inherited = value_tags.get(inputs[0], "")
        node_tag = explicit_output_tag or inherited
        node_tags.append(node_tag)
        out_tag = explicit_output_tag or inherited
        if out_tag:
            for out in outputs:
                set_value_tag(out, out_tag)

        for attr in _iter(getattr(node, "attribute", ())):
            subgraphs = []
            subgraph = getattr(attr, "g", None)
            if subgraph is not None:
                subgraphs.append(subgraph)
            subgraphs.extend(_iter(getattr(attr, "graphs", ())))
            for subgraph in subgraphs:
                _infer_value_and_node_tags_python(subgraph)

    return value_tags, node_tags


def _write_value_and_node_tags_to_metadata_python(graph_or_nodes_or_function: Any) -> None:
    """Writes inferred ``shape``/``axes``/``weight`` tags into metadata.

    Values are stored under ``onnx_light.value_tag`` (for value infos,
    initializers, inputs and outputs) and as a graph/function-level JSON map
    under ``onnx_light.value_tags``. Node tags are stored under
    ``onnx_light.node_tag``. Subgraphs nested in node attributes are processed
    recursively.

    Returns:
        ``None``. The function mutates metadata fields in place.
    """
    value_tags, node_tags = _infer_value_and_node_tags_python(graph_or_nodes_or_function)
    if hasattr(graph_or_nodes_or_function, "graph"):
        graph_or_nodes_or_function = graph_or_nodes_or_function.graph
    nodes = _iter(getattr(graph_or_nodes_or_function, "node", graph_or_nodes_or_function))
    for node, tag in zip(nodes, node_tags, strict=True):
        if tag:
            _set_metadata_value(node, NODE_TAG_METADATA_KEY, tag)
    if hasattr(graph_or_nodes_or_function, "add_metadata") or hasattr(
        graph_or_nodes_or_function, "metadata_props"
    ):
        payload = dict(sorted(value_tags.items()))
        _set_metadata_value(
            graph_or_nodes_or_function, VALUE_TAGS_METADATA_KEY, json.dumps(payload)
        )
        for collection in ("input", "value_info", "output", "initializer"):
            for value in _iter(getattr(graph_or_nodes_or_function, collection, ())):
                name = _s(getattr(value, "name", ""))
                tag = value_tags.get(name, "")
                if tag:
                    _set_metadata_value(value, VALUE_TAG_METADATA_KEY, tag)
    for node in _iter(getattr(graph_or_nodes_or_function, "node", ())):
        for attr in _iter(getattr(node, "attribute", ())):
            subgraphs = []
            subgraph = getattr(attr, "g", None)
            if subgraph is not None:
                subgraphs.append(subgraph)
            subgraphs.extend(_iter(getattr(attr, "graphs", ())))
            for subgraph in subgraphs:
                _write_value_and_node_tags_to_metadata_python(subgraph)


def _is_onnx_light_proto_or_node_list(obj: Any) -> bool:
    module_name = type(obj).__module__
    if module_name.startswith("onnx_light."):
        return True
    if (
        isinstance(obj, list)
        and obj
        and all(type(n).__module__.startswith("onnx_light.") for n in obj)
    ):
        return True
    return False


def infer_value_and_node_tags(
    graph_or_nodes_or_function: Any,
) -> tuple[dict[str, str], list[str]]:
    """Infers semantic ``shape``/``axes``/``weight`` tags for values and nodes.

    Returns:
        A pair ``(value_tags, node_tags)`` where ``value_tags`` maps value
        names to tags and ``node_tags`` is ordered like the processed node list.
    """
    if _shape_inference is not None and _is_onnx_light_proto_or_node_list(
        graph_or_nodes_or_function
    ):
        return _shape_inference.infer_value_and_node_tags(graph_or_nodes_or_function)
    return _infer_value_and_node_tags_python(graph_or_nodes_or_function)


def write_value_and_node_tags_to_metadata(graph_or_nodes_or_function: Any) -> None:
    """Writes inferred ``shape``/``axes``/``weight`` tags into metadata.

    Returns:
        ``None``. The function mutates metadata fields in place.
    """
    if _shape_inference is not None and _is_onnx_light_proto_or_node_list(
        graph_or_nodes_or_function
    ):
        _shape_inference.write_value_and_node_tags_to_metadata(graph_or_nodes_or_function)
        return
    _write_value_and_node_tags_to_metadata_python(graph_or_nodes_or_function)


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
    raw = _node_metadata_value(graph, VALUE_TAGS_METADATA_KEY)
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            for name, tag in payload.items():
                norm = _normalise_value_tag(_s(tag))
                if norm:
                    tags[_s(name)] = norm
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
