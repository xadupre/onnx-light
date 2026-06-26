"""Convert an ONNX model or graph to a `Graphviz <https://graphviz.org/>`_
DOT source string.

The resulting string can be written to a ``.dot`` file and then rendered
with ``dot -Tsvg model.dot -o model.svg`` or any other Graphviz tool::

    from onnx_light.tools import to_dot

    with open("model.dot", "w", encoding="utf-8") as f:
        f.write(to_dot(model))

The converter is implemented in pure Python and only depends on the
attributes of the standard ONNX message types (``ModelProto``,
``GraphProto``, ``NodeProto``, ``ValueInfoProto``, ``TensorProto`` and
``TensorShapeProto``).  It therefore works both with messages built by
:mod:`onnx_light` and with messages built by the upstream :mod:`onnx`
package.
"""

from __future__ import annotations

from typing import Any

from ._proto_utils import (
    NODE_TAG_METADATA_KEY,
    VALUE_TAG_COLORS,
    _dtype_name,
    _extract_graph,
    _format_inplace_reuse,
    _format_shape,
    _graph_value_tags,
    _iter,
    _looks_like_graph,
    _node_metadata_value,
    _short_display_name,
    _s,
)

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

_DOT_STYLES: dict[str, dict[str, str]] = {
    "input": {"shape": "ellipse", "fillcolor": "#cde4ff", "color": "#3a6ea5", "style": "filled"},
    "output": {"shape": "ellipse", "fillcolor": "#ffe1b3", "color": "#a35a00", "style": "filled"},
    "initializer": {
        "shape": "cylinder",
        "fillcolor": "#eeeeee",
        "color": "#888888",
        "style": "filled,dashed",
    },
    "op": {"shape": "box", "fillcolor": "#d4ecd4", "color": "#3a8c3a", "style": "filled"},
}

_DOT_TAG_COLORS = {
    tag: {"fillcolor": v["fill"], "color": v["stroke"]} for tag, v in VALUE_TAG_COLORS.items()
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _escape_dot_label(text: str) -> str:
    """Escapes characters that have special meaning inside DOT double-quoted labels."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("<", "\\<")
        .replace(">", "\\>")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("|", "\\|")
    )


def _dot_node_stmt(node_id: str, label: str, kind: str, tag: str = "") -> str:
    """Returns a DOT node statement string."""
    style = dict(_DOT_STYLES[kind])
    if tag in _DOT_TAG_COLORS:
        style.update(_DOT_TAG_COLORS[tag])
    attrs = (
        f'label="{_escape_dot_label(label)}", '
        f'shape={style["shape"]}, '
        f'style="{style["style"]}", '
        f'fillcolor="{style["fillcolor"]}", '
        f'color="{style["color"]}"'
    )
    return f'    "{node_id}" [{attrs}];'


def _dot_edge_stmt(from_id: str, to_id: str, label: str = "") -> str:
    """Returns a DOT edge statement string."""
    if label:
        return f'    "{from_id}" -> "{to_id}" [label="{_escape_dot_label(label)}"];'
    return f'    "{from_id}" -> "{to_id}";'


class _IdAllocator:
    """Allocates stable, DOT-safe identifiers for ONNX tensors and nodes.

    DOT identifiers are wrapped in double quotes, so technically any string
    is valid.  The allocator still sanitises names so that identifiers remain
    human-readable in the source and unique within the graph.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._used: set[str] = set()
        self._by_name: dict[str, str] = {}

    def get(self, name: str) -> str:
        """Returns a stable identifier for *name*, creating one if needed."""
        if name in self._by_name:
            return self._by_name[name]
        base = self._sanitise(name) if name else self._prefix
        candidate = base
        i = 1
        while candidate in self._used:
            i += 1
            candidate = f"{base}_{i}"
        self._used.add(candidate)
        self._by_name[name] = candidate
        return candidate

    def fresh(self, base: str) -> str:
        """Creates a new identifier derived from *base* that does not collide."""
        sanitised = self._sanitise(base) or self._prefix
        candidate = sanitised
        i = 1
        while candidate in self._used:
            i += 1
            candidate = f"{sanitised}_{i}"
        self._used.add(candidate)
        return candidate

    def _sanitise(self, name: str) -> str:
        out_chars: list[str] = []
        for ch in name:
            if ch.isalnum() or ch == "_":
                out_chars.append(ch)
            else:
                out_chars.append("_")
        sanitised = "".join(out_chars)
        if not sanitised:
            sanitised = self._prefix
        if sanitised[0].isdigit():
            sanitised = f"{self._prefix}_{sanitised}"
        return f"{self._prefix}_{sanitised}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_dot(
    model_or_graph: Any,
    *,
    direction: str = "TB",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
    include_inplace: bool = False,
) -> str:
    """Renders an ONNX ``ModelProto`` or ``GraphProto`` as a Graphviz DOT string.

    Args:
        model_or_graph: A ``ModelProto`` or ``GraphProto`` instance.  Both
            :mod:`onnx_light` and :mod:`onnx` messages are accepted.
        direction: Graphviz layout direction; one of ``"TB"``
            (top-to-bottom), ``"BT"`` (bottom-to-top), ``"LR"``
            (left-to-right), or ``"RL"`` (right-to-left).  Defaults to
            ``"TB"``.
        include_initializers: When :data:`True`, initializers are rendered
            as separate (cylinder) nodes and connected to the consumers
            they feed.  When :data:`False`, initializer tensors are not
            shown.
        include_shapes: When :data:`True`, tensor type/shape information
            available in graph inputs, outputs, ``value_info`` and
            initializers is appended to the corresponding node labels.
        include_attributes: When :data:`True`, node attribute names are
            listed inside the operator label.
        include_inplace: When :data:`True`, the in-place reuse
            opportunities recorded in each node's ``metadata_props``
            (under the ``onnx_light.inplace_reuse`` key) are appended to
            the operator label, for example ``inplace: out0=in1(equal)``.

    Returns:
        The DOT source as a single ``str`` (newline-separated).  Write it
        to a ``.dot`` file and render it with, for example,
        ``dot -Tsvg model.dot -o model.svg``.

    Raises:
        TypeError: If ``model_or_graph`` is neither a ``ModelProto`` nor
            a ``GraphProto``.
        ValueError: If ``direction`` is not a supported Graphviz direction.

    The example below builds a small ``Abs`` chain and renders the DOT source:

    .. runpython::

        from onnx_light.onnx_lib import TensorProto
        from onnx_light.onnx.helper import (
            make_graph,
            make_model,
            make_node,
            make_opsetid,
            make_tensor_value_info,
        )
        from onnx_light.tools import to_dot

        X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
        graph = make_graph(
            [
                make_node("Abs", ["X"], ["A"]),
                make_node("Abs", ["A"], ["B"]),
                make_node("Abs", ["B"], ["Y"]),
            ],
            "example",
            [X],
            [Y],
        )
        model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        model.ir_version = 8

        print(to_dot(model))
    """

    valid_directions = {"TB", "BT", "LR", "RL"}
    if direction not in valid_directions:
        raise ValueError(
            f"Unsupported direction {direction!r}; "
            f"expected one of {sorted(valid_directions)}."
        )

    graph = _extract_graph(model_or_graph)
    return to_dot_graph(
        graph,
        direction=direction,
        include_initializers=include_initializers,
        include_shapes=include_shapes,
        include_attributes=include_attributes,
        include_inplace=include_inplace,
    )


def to_dot_graph(
    graph: Any,
    *,
    direction: str = "TB",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
    include_inplace: bool = False,
) -> str:
    """Renders a ``GraphProto`` as a Graphviz DOT string.

    See :func:`to_dot` for the meaning of every parameter.
    """

    if not _looks_like_graph(graph):
        raise TypeError(
            "to_dot_graph expected a GraphProto-like object "
            f"with 'node', 'input' and 'output' fields, got {type(graph).__name__}."
        )

    tensor_ids = _IdAllocator(prefix="t")
    node_ids = _IdAllocator(prefix="n")
    value_tags = _graph_value_tags(graph)

    lines: list[str] = [
        "digraph onnx {",
        f'    rankdir="{direction}";',
        '    node [fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=9];',
    ]

    # Collect shape annotations from inputs, outputs and value_info.
    shape_lookup: dict[str, str] = {}
    if include_shapes:
        for value_info in _iter(getattr(graph, "input", ())):
            shape_lookup[_s(value_info.name)] = _format_shape(getattr(value_info, "type", None))
        for value_info in _iter(getattr(graph, "output", ())):
            shape_lookup[_s(value_info.name)] = _format_shape(getattr(value_info, "type", None))
        for value_info in _iter(getattr(graph, "value_info", ())):
            shape_lookup[_s(value_info.name)] = _format_shape(getattr(value_info, "type", None))

    initializer_names: set[str] = set()
    initializer_shapes: dict[str, str] = {}
    for init in _iter(getattr(graph, "initializer", ())):
        name = _s(init.name)
        initializer_names.add(name)
        if include_shapes:
            dims = ",".join(str(int(d)) for d in getattr(init, "dims", ()))
            dtype = _dtype_name(int(getattr(init, "data_type", 0)))
            initializer_shapes[name] = f"{dtype}[{dims}]" if dims else dtype

    # Emit input nodes (ellipse shape).
    input_names = [_s(v.name) for v in _iter(getattr(graph, "input", ()))]
    for name in input_names:
        if not include_initializers and name in initializer_names:
            continue
        node_id = tensor_ids.get(name)
        label_parts: list[str] = [_short_display_name(name) or "(unnamed)"]
        if include_shapes and shape_lookup.get(name):
            label_parts.append(shape_lookup[name])
        label = "\n".join(label_parts)
        tag = value_tags.get(name, "")
        lines.append(_dot_node_stmt(node_id, label, "input", tag))

    # Emit initializer nodes (cylinder shape).  Only render initializers that
    # are not also listed as inputs, to avoid duplicates.
    if include_initializers:
        input_name_set = set(input_names)
        for name in sorted(initializer_names):
            if name in input_name_set:
                continue
            node_id = tensor_ids.get(name)
            label_parts = [_short_display_name(name) or "(unnamed)"]
            if include_shapes and initializer_shapes.get(name):
                label_parts.append(initializer_shapes[name])
            label = "\n".join(label_parts)
            tag = value_tags.get(name, "")
            lines.append(_dot_node_stmt(node_id, label, "initializer", tag))

    # Emit operator nodes and edges.
    for index, node in enumerate(_iter(getattr(graph, "node", ()))):
        op_type = _s(getattr(node, "op_type", "")) or "Op"
        raw_name = _s(getattr(node, "name", ""))
        node_id = node_ids.fresh(raw_name or f"{op_type}_{index}")

        label_parts = [op_type]
        if raw_name:
            label_parts.append(_short_display_name(raw_name))
        if include_attributes:
            attr_names = [_s(a.name) for a in _iter(getattr(node, "attribute", ()))]
            if attr_names:
                label_parts.append(", ".join(sorted(attr_names)))
        if include_inplace:
            inplace_label = _format_inplace_reuse(node)
            if inplace_label:
                label_parts.append(inplace_label)
        label = "\n".join(label_parts)

        node_tag = _s(_node_metadata_value(node, NODE_TAG_METADATA_KEY)).lower()
        lines.append(_dot_node_stmt(node_id, label, "op", node_tag))

        for inp in _iter(getattr(node, "input", ())):
            inp_name = _s(inp)
            if not inp_name:
                # Optional input that was left empty.
                continue
            if not include_initializers and inp_name in initializer_names:
                continue
            tensor_id = tensor_ids.get(inp_name)
            edge_label = ""
            if include_shapes:
                shape = shape_lookup.get(inp_name) or initializer_shapes.get(inp_name, "")
                if shape:
                    edge_label = shape
            lines.append(_dot_edge_stmt(tensor_id, node_id, edge_label))

        for out in _iter(getattr(node, "output", ())):
            out_name = _s(out)
            if not out_name:
                continue
            tensor_id = tensor_ids.get(out_name)
            edge_label = ""
            if include_shapes and shape_lookup.get(out_name):
                edge_label = shape_lookup[out_name]
            lines.append(_dot_edge_stmt(node_id, tensor_id, edge_label))

    # Emit output nodes (ellipse shape, output colour).  The tensor
    # identifiers already exist from the producing nodes; we redeclare them
    # with the output style.
    for value_info in _iter(getattr(graph, "output", ())):
        name = _s(value_info.name)
        if not name:
            continue
        node_id = tensor_ids.get(name)
        label_parts = [_short_display_name(name)]
        if include_shapes and shape_lookup.get(name):
            label_parts.append(shape_lookup[name])
        label = "\n".join(label_parts)
        tag = value_tags.get(name, "")
        lines.append(_dot_node_stmt(node_id, label, "output", tag))

    lines.append("}")
    return "\n".join(lines)
