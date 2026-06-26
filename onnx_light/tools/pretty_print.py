"""Pretty-print an ONNX proto in a compact, human-readable form.

The renderer mirrors the style of ``yobx.helpers.onnx_helper.pretty_onnx``
(see https://github.com/xadupre/yet-another-onnx-builder): each node is
rendered as ``OpType(inputs) -> outputs`` (optionally prefixed by the
node domain), value infos as ``dtype[shape] name`` and graphs/models as
a "simple text plot" listing opsets, inputs, initializers, nodes and
outputs.

The implementation is pure Python and duck-typed against the standard
ONNX message API (``ModelProto``, ``GraphProto``, ``FunctionProto``,
``NodeProto``, ``ValueInfoProto``, ``TensorProto``, ``TypeProto``,
``AttributeProto`` and ``TensorShapeProto``); it therefore works both
with messages built by :mod:`onnx_light` and with messages built by the
upstream :mod:`onnx` package.

Example::

    from onnx_light.tools import pretty_onnx

    print(pretty_onnx(model))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._proto_utils import (
    NODE_TAG_METADATA_KEY,
    VALUE_TAGS,
    _format_inplace_reuse,
    _format_release_after,
    _node_metadata_value,
)
from .mermaid import _dtype_name, _s

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only.
    from collections.abc import Iterable

__all__ = ["pretty_onnx"]


# AttributeProto type ids (mirror ``onnx.AttributeProto.AttributeType``).
_ATTR_FLOAT = 1
_ATTR_INT = 2
_ATTR_STRING = 3
_ATTR_TENSOR = 4
_ATTR_GRAPH = 5
_ATTR_FLOATS = 6
_ATTR_INTS = 7
_ATTR_STRINGS = 8
_ATTR_TENSORS = 9
_ATTR_GRAPHS = 10
_ATTR_SPARSE_TENSOR = 11
_ATTR_SPARSE_TENSORS = 12
_ATTR_TYPE_PROTO = 13
_ATTR_TYPE_PROTOS = 14


# ---------------------------------------------------------------------------
# Detection helpers (duck typing)
# ---------------------------------------------------------------------------


def _is_value_info(obj: Any) -> bool:
    return hasattr(obj, "name") and hasattr(obj, "type") and not hasattr(obj, "node")


def _is_type_proto(obj: Any) -> bool:
    return hasattr(obj, "tensor_type") and not hasattr(obj, "name") and not hasattr(obj, "type")


def _is_attribute(obj: Any) -> bool:
    # AttributeProto has both ``type`` and ``i``/``f``/``s`` scalar fields.
    return (
        hasattr(obj, "name") and hasattr(obj, "type") and hasattr(obj, "i") and hasattr(obj, "f")
    )


def _is_node(obj: Any) -> bool:
    return hasattr(obj, "op_type") and hasattr(obj, "input") and hasattr(obj, "output")


def _is_tensor(obj: Any) -> bool:
    return hasattr(obj, "data_type") and hasattr(obj, "dims")


def _is_graph(obj: Any) -> bool:
    return (
        hasattr(obj, "node")
        and hasattr(obj, "input")
        and hasattr(obj, "output")
        and hasattr(obj, "initializer")
    )


def _is_model(obj: Any) -> bool:
    graph = getattr(obj, "graph", None)
    return graph is not None and _is_graph(graph)


def _is_function(obj: Any) -> bool:
    return (
        hasattr(obj, "node")
        and hasattr(obj, "input")
        and hasattr(obj, "output")
        and not hasattr(obj, "initializer")
    )


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------


def _format_shape_dims(shape: Any) -> str:
    """Returns the comma-separated dim list for a ``TensorShapeProto``."""
    if shape is None:
        return ""
    dims = getattr(shape, "dim", None) or []
    parts: list[str] = []
    for d in dims:
        dim_param = _s(getattr(d, "dim_param", "") or "")
        if dim_param:
            parts.append(dim_param)
            continue
        dim_value = getattr(d, "dim_value", 0) or 0
        parts.append(str(int(dim_value)))
    return ",".join(parts)


def _format_type_proto(type_proto: Any) -> str:
    """Returns ``dtype[shape]`` for a TensorType-backed TypeProto."""
    if type_proto is None:
        return ""
    tensor_type = getattr(type_proto, "tensor_type", None)
    if tensor_type is not None and getattr(tensor_type, "elem_type", None):
        dtype = _dtype_name(int(tensor_type.elem_type))
        shape = _format_shape_dims(getattr(tensor_type, "shape", None))
        return f"{dtype}[{shape}]"
    seq_type = getattr(type_proto, "sequence_type", None)
    if seq_type is not None and getattr(seq_type, "elem_type", None) is not None:
        return f"seq({_format_type_proto(seq_type.elem_type)})"
    opt_type = getattr(type_proto, "optional_type", None)
    if opt_type is not None and getattr(opt_type, "elem_type", None) is not None:
        return f"optional({_format_type_proto(opt_type.elem_type)})"
    map_type = getattr(type_proto, "map_type", None)
    if map_type is not None and getattr(map_type, "key_type", None):
        key = _dtype_name(int(map_type.key_type))
        value = _format_type_proto(getattr(map_type, "value_type", None))
        return f"map({key}, {value})" if value else f"map({key})"
    return ""


def _format_value_info(vi: Any) -> str:
    name = _s(getattr(vi, "name", "")) or ""
    typed = _format_type_proto(getattr(vi, "type", None))
    if typed and name:
        return f"{typed} {name}"
    return typed or name


def _format_tensor_proto(tensor: Any) -> str:
    """Returns ``onnx.TensorProto:dtype:shape:name`` like yobx's pretty_onnx."""
    dtype = int(getattr(tensor, "data_type", 0) or 0)
    dims = list(getattr(tensor, "dims", []) or [])
    shape = "x".join(str(int(d)) for d in dims)
    name = _s(getattr(tensor, "name", "")) or ""
    return f"onnx.TensorProto:{dtype}:{shape}:{name}"


def _attr_value(attr: Any) -> str:
    attr_type = int(getattr(attr, "type", 0) or 0)
    name = _s(getattr(attr, "name", "")) or ""
    if attr_type == _ATTR_INT:
        return f"{name}={int(getattr(attr, 'i', 0))}"
    if attr_type == _ATTR_INTS:
        ints = list(getattr(attr, "ints", []) or [])
        return f"{name}=[{', '.join(str(int(v)) for v in ints)}]"
    if attr_type == _ATTR_FLOAT:
        return f"{name}={float(getattr(attr, 'f', 0.0))}"
    if attr_type == _ATTR_FLOATS:
        floats = list(getattr(attr, "floats", []) or [])
        return f"{name}=[{', '.join(repr(float(v)) for v in floats)}]"
    if attr_type == _ATTR_STRING:
        return f"{name}={_s(getattr(attr, 's', b''))!r}"
    if attr_type == _ATTR_STRINGS:
        strings = [_s(v) for v in getattr(attr, "strings", []) or []]
        return f"{name}=[{', '.join(repr(v) for v in strings)}]"
    if attr_type == _ATTR_TENSOR:
        t = getattr(attr, "t", None)
        if t is None:
            return f"{name}=<tensor>"
        return f"{name}={_format_tensor_proto(t)}"
    if attr_type == _ATTR_GRAPH:
        return f"{name}=<graph>"
    if attr_type == _ATTR_GRAPHS:
        graphs = list(getattr(attr, "graphs", []) or [])
        return f"{name}=<{len(graphs)} graphs>"
    if attr_type == _ATTR_SPARSE_TENSOR:
        return f"{name}=<sparse_tensor>"
    if attr_type == _ATTR_SPARSE_TENSORS:
        sparse = list(getattr(attr, "sparse_tensors", []) or [])
        return f"{name}=<{len(sparse)} sparse_tensors>"
    if attr_type == _ATTR_TYPE_PROTO:
        return f"{name}={_format_type_proto(getattr(attr, 'tp', None)) or '<type>'}"
    if attr_type == _ATTR_TYPE_PROTOS:
        items = [
            _format_type_proto(v) or "<type>" for v in getattr(attr, "type_protos", []) or []
        ]
        return f"{name}=[{', '.join(items)}]"
    # Fall back: try populated scalar fields.
    for field, fmt in (
        ("i", lambda v: str(int(v))),
        ("f", lambda v: str(float(v))),
        ("s", lambda v: repr(_s(v))),
    ):
        if getattr(attr, field, None):
            return f"{name}={fmt(getattr(attr, field))}"
    return name


def _format_node(
    node: Any,
    with_attributes: bool,
    highlight: set[str] | None,
    include_node_tags: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> str:
    def _high(n: str) -> str:
        text = _s(n)
        if highlight and text in highlight:
            return f"**{text}**"
        return text

    inputs = ", ".join(_high(i) for i in getattr(node, "input", []) or [])
    outputs = ", ".join(_high(o) for o in getattr(node, "output", []) or [])
    op_type = _s(getattr(node, "op_type", "")) or "?"
    domain = _s(getattr(node, "domain", ""))
    head = f"{domain}.{op_type}" if domain else op_type
    text = f"{head}({inputs}) -> {outputs}"

    if include_node_tags:
        tag = _s(_node_metadata_value(node, NODE_TAG_METADATA_KEY)).lower()
        if tag in VALUE_TAGS:
            text = f"[{tag}] {text}"

    if include_inplace:
        inplace_label = _format_inplace_reuse(node)
        if inplace_label:
            text = f"{text}  {inplace_label}"

    if include_release:
        release_label = _format_release_after(node)
        if release_label:
            text = f"{text}  {release_label}"

    attrs = list(getattr(node, "attribute", []) or [])
    if not with_attributes or not attrs:
        return text
    rows = [_attr_value(a) for a in attrs]
    if len(rows) > 1:
        return text + "\n" + "\n".join(f"    {r}" for r in rows)
    return f"{text}  ---  {rows[0]}"


# ---------------------------------------------------------------------------
# Graph / Function / Model rendering ("simple text plot")
# ---------------------------------------------------------------------------


def _format_opsets(opsets: Iterable[Any]) -> list[str]:
    lines: list[str] = []
    for opset in opsets or []:
        domain = _s(getattr(opset, "domain", "") or "")
        version = int(getattr(opset, "version", 0) or 0)
        lines.append(f"opset: domain={domain!r} version={version}")
    return lines


def _format_initializer(tensor: Any) -> str:
    dtype = _dtype_name(int(getattr(tensor, "data_type", 0) or 0))
    dims = list(getattr(tensor, "dims", []) or [])
    shape = ",".join(str(int(d)) for d in dims)
    name = _s(getattr(tensor, "name", "")) or ""
    return f"init: {dtype}[{shape}] {name}"


def _format_graph_lines(
    graph: Any,
    with_attributes: bool,
    highlight: set[str] | None,
    indent: str = "",
    include_node_tags: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> list[str]:
    lines: list[str] = []
    name = _s(getattr(graph, "name", ""))
    if name:
        lines.append(f"{indent}graph: name={name!r}")
    for vi in getattr(graph, "input", []) or []:
        lines.append(f"{indent}input: {_format_value_info(vi)}")
    for init in getattr(graph, "initializer", []) or []:
        lines.append(f"{indent}{_format_initializer(init)}")
    for index, node in enumerate(getattr(graph, "node", []) or []):
        rendered = _format_node(
            node,
            with_attributes=with_attributes,
            highlight=highlight,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        )
        node_lines = rendered.splitlines()
        if node_lines:
            lines.append(f"{indent}{index}: {node_lines[0]}")
            padding = " " * (len(f"{index}: "))
            for line in node_lines[1:]:
                lines.append(f"{indent}{padding}{line}")
    for vi in getattr(graph, "output", []) or []:
        lines.append(f"{indent}output: {_format_value_info(vi)}")
    return lines


def _format_function_lines(
    fn: Any,
    with_attributes: bool,
    highlight: set[str] | None,
    include_node_tags: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> list[str]:
    name = _s(getattr(fn, "name", "")) or ""
    domain = _s(getattr(fn, "domain", "")) or ""
    lines = [f"function: {name}[{domain}]"]
    for i in getattr(fn, "input", []) or []:
        lines.append(f"input: {_s(i)}")
    for index, node in enumerate(getattr(fn, "node", []) or []):
        node_lines = _format_node(
            node,
            with_attributes=with_attributes,
            highlight=highlight,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        ).splitlines()
        if node_lines:
            lines.append(f"{index}: {node_lines[0]}")
            padding = " " * (len(f"{index}: "))
            for line in node_lines[1:]:
                lines.append(f"{padding}{line}")
    for o in getattr(fn, "output", []) or []:
        lines.append(f"output: {_s(o)}")
    return lines


def _format_model(
    model: Any,
    with_attributes: bool,
    highlight: set[str] | None,
    include_node_tags: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> str:
    lines: list[str] = []
    lines.extend(_format_opsets(getattr(model, "opset_import", []) or []))
    lines.extend(
        _format_graph_lines(
            model.graph,
            with_attributes=with_attributes,
            highlight=highlight,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        )
    )
    for fn in getattr(model, "functions", []) or []:
        lines.append("")
        lines.extend(
            _format_function_lines(
                fn,
                with_attributes=with_attributes,
                highlight=highlight,
                include_node_tags=include_node_tags,
                include_inplace=include_inplace,
                include_release=include_release,
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def pretty_onnx(
    onx: Any,
    with_attributes: bool = False,
    highlight: set[str] | None = None,
    shape_inference: bool = False,
    include_node_tags: bool = False,
    include_inplace: bool = False,
    include_release: bool = False,
) -> str:
    """Returns a compact, human-readable string for any ONNX proto.

    The argument may be a ``ModelProto``, ``GraphProto``,
    ``FunctionProto``, ``NodeProto``, ``ValueInfoProto``, ``TypeProto``,
    ``AttributeProto``, ``TensorProto`` or a file path (``str``).  The
    rendering style mirrors ``yobx.helpers.onnx_helper.pretty_onnx``:
    nodes appear as ``OpType(inputs) -> outputs``, value infos as
    ``dtype[shape] name``, and graphs/models as a list of opsets,
    inputs, initializers, nodes and outputs.

    :param onx: ONNX proto, or a path to a model file.
    :param with_attributes: when True, node attributes are appended to
        the node line (each ``name=value`` pair on its own line, or
        inline after ``---`` when there is a single attribute).
    :param highlight: optional set of tensor names to wrap in ``**``
        markers in the rendered I/O lists.
    :param shape_inference: when True and ``onx`` is a model, run
        :mod:`onnx_light` shape inference before rendering.
    :param include_node_tags: when True, nodes that carry a
        ``onnx_light.node_tag`` metadata entry (``shape``, ``axes`` or
        ``weight``) are prefixed with ``[tag]`` in the rendered output.
    :param include_inplace: when True, nodes that carry
        ``onnx_light.inplace_reuse`` metadata have the inplace reuse
        opportunities appended to their line, e.g.
        ``inplace: out0=in0(equal)``.
    :param include_release: when True, nodes that carry
        ``onnx_light.release_after`` metadata have the release hints
        appended to their line, e.g. ``release: A, B``.
    :return: the formatted text.
    """
    assert onx is not None, "onx cannot be None"

    if isinstance(onx, str):
        from ..onnx import load

        onx = load(onx, load_external_data=False)

    if shape_inference:
        assert _is_model(onx), f"shape_inference only works for ModelProto, not {type(onx)}"
        from ..onnx import shape_inference as _shape_inference

        onx = _shape_inference.infer_shapes(onx)

    # Dispatch in a duck-typed way; the order matters because some types
    # share attribute names (e.g. NodeProto and FunctionProto both have
    # ``input``/``output``).
    if _is_attribute(onx):
        return _attr_value(onx)

    if _is_node(onx):
        return _format_node(
            onx,
            with_attributes=with_attributes,
            highlight=highlight,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        )

    if _is_tensor(onx):
        return _format_tensor_proto(onx)

    if _is_value_info(onx):
        return _format_value_info(onx)

    if _is_type_proto(onx):
        return _format_type_proto(onx)

    if _is_model(onx):
        return _format_model(
            onx,
            with_attributes=with_attributes,
            highlight=highlight,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        )

    if _is_graph(onx):
        return "\n".join(
            _format_graph_lines(
                onx,
                with_attributes=with_attributes,
                highlight=highlight,
                include_node_tags=include_node_tags,
                include_inplace=include_inplace,
                include_release=include_release,
            )
        )

    if _is_function(onx):
        return "\n".join(
            _format_function_lines(
                onx,
                with_attributes=with_attributes,
                highlight=highlight,
                include_node_tags=include_node_tags,
                include_inplace=include_inplace,
                include_release=include_release,
            )
        )

    return str(onx)
