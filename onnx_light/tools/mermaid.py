"""Convert an ONNX model or graph to a `Mermaid <https://mermaid.js.org/>`_
``flowchart`` diagram.

The resulting string can be embedded directly in Markdown or in a
Sphinx page using the ``mermaid`` directive::

    from onnx_light.tools import to_mermaid

    print(to_mermaid(model))

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
    _dtype_name,
    _extract_graph,
    _format_inplace_reuse,
    _format_shape,
    _iter,
    _looks_like_graph,
    _s,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Characters that must be escaped inside a Mermaid label that is wrapped
# in double quotes.  Mermaid happily renders ``[``, ``(``, ``{``, ``|``
# and HTML tags such as ``<br/>`` inside a quoted label, so only the
# few characters that would otherwise terminate the string (or be
# interpreted as HTML) are escaped.
_MERMAID_ESCAPE = {'"': "#quot;", "<": "#lt;", ">": "#gt;"}


def _escape_label(label: str) -> str:
    """Escape characters that have a special meaning in Mermaid labels.

    The output is intended to be placed between double quotes; the
    rendered text will contain newlines (encoded as ``<br/>``) and any
    other characters verbatim.  HTML-like substitutions use Mermaid's
    ``#`` escape syntax (e.g. ``#quot;``) rather than ``&entity;`` so
    that the entity passes through HTML sanitisers used by Mermaid.
    """

    # Pre-substitute ``<br/>`` placeholders for newlines.  We do this
    # *after* the per-character escape pass for ``<``/``>`` by using a
    # sentinel that cannot appear in user input.
    out: list[str] = []
    for ch in label:
        if ch == "\n":
            out.append("\x00BR\x00")
        elif ch in _MERMAID_ESCAPE:
            out.append(_MERMAID_ESCAPE[ch])
        else:
            out.append(ch)
    return "".join(out).replace("\x00BR\x00", "<br/>")


def _join_label(*parts: str) -> str:
    """Escape each part and join them with a Mermaid line break."""
    return "<br/>".join(_escape_label(p) for p in parts if p)


# ---------------------------------------------------------------------------
# Identifier allocation
# ---------------------------------------------------------------------------


class _IdAllocator:
    """Allocate stable, Mermaid-safe identifiers for ONNX tensors and nodes.

    Mermaid identifiers must match ``[A-Za-z0-9_]+``.  Two distinct ONNX
    names that map to the same sanitised form must still receive
    different identifiers, so a small counter is appended when needed.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._used: set[str] = set()
        self._by_name: dict[str, str] = {}

    def get(self, name: str) -> str:
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


def to_mermaid(
    model_or_graph: Any,
    *,
    direction: str = "TB",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
    include_inplace: bool = False,
) -> str:
    """Render an ONNX ``ModelProto`` or ``GraphProto`` as a Mermaid flowchart.

    Args:
        model_or_graph: A ``ModelProto`` or ``GraphProto`` instance.  Both
            :mod:`onnx_light` and :mod:`onnx` messages are accepted.
        direction: Mermaid flowchart direction; one of ``"TB"``, ``"TD"``,
            ``"BT"``, ``"LR"`` or ``"RL"``.  Defaults to ``"TB"``
            (top-to-bottom).
        include_initializers: When :data:`True`, initializers are rendered
            as separate (cylinder) nodes and connected to the consumers
            they feed.  When :data:`False`, initializer tensors are not
            shown.
        include_shapes: When :data:`True`, tensor type/shape information
            available in graph inputs, outputs, ``value_info`` and
            initializers is appended to the corresponding node labels.
        include_attributes: When :data:`True`, node attribute names are
            listed inside the operator label.
        include_inplace: When :data:`True`, the in-place reuse opportunities
            recorded in each node's ``metadata_props`` (under the
            ``onnx_light.inplace_reuse`` key) are appended to the operator
            label, for example ``inplace: out0=in1(equal)``.

    Returns:
        The Mermaid source as a single ``str`` (newline-separated).  The
        returned text is **not** wrapped in a fenced code block so the
        caller can choose between ``"```mermaid\\n...\\n```"`` for
        Markdown and the ``.. mermaid::`` directive for Sphinx.

    Raises:
        TypeError: If ``model_or_graph`` is neither a ``ModelProto`` nor
            a ``GraphProto``.
        ValueError: If ``direction`` is not a supported Mermaid
            flowchart direction.
    """

    valid_directions = {"TB", "TD", "BT", "LR", "RL"}
    if direction not in valid_directions:
        raise ValueError(
            f"Unsupported Mermaid direction {direction!r}; "
            f"expected one of {sorted(valid_directions)}."
        )

    graph = _extract_graph(model_or_graph)
    return to_mermaid_graph(
        graph,
        direction=direction,
        include_initializers=include_initializers,
        include_shapes=include_shapes,
        include_attributes=include_attributes,
        include_inplace=include_inplace,
    )


def to_mermaid_graph(
    graph: Any,
    *,
    direction: str = "TB",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
    include_inplace: bool = False,
) -> str:
    """Render a ``GraphProto`` as a Mermaid flowchart.

    See :func:`to_mermaid` for the meaning of every parameter.
    """

    if not _looks_like_graph(graph):
        raise TypeError(
            "to_mermaid_graph expected a GraphProto-like object "
            f"with 'node', 'input' and 'output' fields, got {type(graph).__name__}."
        )

    tensor_ids = _IdAllocator(prefix="t")
    node_ids = _IdAllocator(prefix="n")

    lines: list[str] = [f"flowchart {direction}"]

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

    # Emit input nodes (stadium shape).
    input_names = [_s(v.name) for v in _iter(getattr(graph, "input", ()))]
    for name in input_names:
        if not include_initializers and name in initializer_names:
            continue
        node_id = tensor_ids.get(name)
        label = _join_label(name or "(unnamed)", shape_lookup.get(name, ""))
        lines.append(f'    {node_id}(["{label}"]):::onnxInput')

    # Emit initializer nodes (cylinder shape).  Only render initializers
    # that are not also listed as inputs, to avoid duplicates.
    if include_initializers:
        input_name_set = set(input_names)
        for name in sorted(initializer_names):
            if name in input_name_set:
                continue
            node_id = tensor_ids.get(name)
            label = _join_label(name or "(unnamed)", initializer_shapes.get(name, ""))
            lines.append(f'    {node_id}[("{label}")]:::onnxInitializer')

    # Emit operator nodes and edges.
    for index, node in enumerate(_iter(getattr(graph, "node", ()))):
        op_type = _s(getattr(node, "op_type", "")) or "Op"
        raw_name = _s(getattr(node, "name", ""))
        node_id = node_ids.fresh(raw_name or f"{op_type}_{index}")

        label_parts: list[str] = [op_type]
        if raw_name:
            # The name is rendered on a second line; italics are added
            # post-escape so the HTML tags survive.
            label_parts.append(raw_name)
        attr_suffix = ""
        if include_attributes:
            attr_names = [_s(a.name) for a in _iter(getattr(node, "attribute", ()))]
            if attr_names:
                attr_suffix = ", ".join(sorted(attr_names))
        if attr_suffix:
            label_parts.append(attr_suffix)
        if include_inplace:
            inplace_label = _format_inplace_reuse(node)
            if inplace_label:
                label_parts.append(inplace_label)
        label = _join_label(*label_parts)
        lines.append(f'    {node_id}["{label}"]:::onnxOp')

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
                    edge_label = f'|"{_escape_label(shape)}"|'
            lines.append(f"    {tensor_id} -->{edge_label} {node_id}")

        for out in _iter(getattr(node, "output", ())):
            out_name = _s(out)
            if not out_name:
                continue
            tensor_id = tensor_ids.get(out_name)
            edge_label = ""
            if include_shapes and shape_lookup.get(out_name):
                edge_label = f'|"{_escape_label(shape_lookup[out_name])}"|'
            lines.append(f"    {node_id} -->{edge_label} {tensor_id}")

    # Emit output nodes (stadium shape).  We only need to add the styling
    # since the identifiers already exist from the producing nodes; we
    # therefore redeclare them with the output shape.
    for value_info in _iter(getattr(graph, "output", ())):
        name = _s(value_info.name)
        if not name:
            continue
        node_id = tensor_ids.get(name)
        label = _join_label(name, shape_lookup.get(name, ""))
        lines.append(f'    {node_id}(["{label}"]):::onnxOutput')

    # Append style classes.
    lines.append("    classDef onnxInput fill:#cde4ff,stroke:#3a6ea5,color:#000;")
    lines.append("    classDef onnxOutput fill:#ffe1b3,stroke:#a35a00,color:#000;")
    lines.append("    classDef onnxInitializer fill:#eeeeee,stroke:#888,color:#000;")
    lines.append("    classDef onnxOp fill:#d4ecd4,stroke:#3a8c3a,color:#000;")

    return "\n".join(lines)
