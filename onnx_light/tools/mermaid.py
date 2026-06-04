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

from typing import TYPE_CHECKING, Any

__all__ = ["to_mermaid", "to_mermaid_graph"]


if TYPE_CHECKING:  # pragma: no cover - imports for type hints only.
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s(value: Any) -> str:
    """Return ``value`` as a regular Python :class:`str`.

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


def _format_shape(type_proto: Any) -> str:
    """Return a compact textual representation of a ``TypeProto``.

    Examples: ``float[1,3,224,224]``, ``int64[?]``, ``seq(float[N])``.
    Returns an empty string when no useful information can be extracted.
    """

    if type_proto is None:
        return ""

    # TensorType
    tensor_type = getattr(type_proto, "tensor_type", None)
    if tensor_type is not None and getattr(tensor_type, "elem_type", 0):
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
    if map_type is not None and getattr(map_type, "key_type", 0):
        key = _dtype_name(int(map_type.key_type))
        value = _format_shape(getattr(map_type, "value_type", None))
        return f"map({key}, {value})" if value else f"map({key})"

    return ""


def _format_tensor_shape(shape: Any) -> str:
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
    return _DTYPE_NAMES.get(int(elem_type), f"dtype{int(elem_type)}")


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
    )


def to_mermaid_graph(
    graph: Any,
    *,
    direction: str = "TB",
    include_initializers: bool = True,
    include_shapes: bool = True,
    include_attributes: bool = False,
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
    # therefore re-declare them with the output shape.
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


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _iter(seq: Any) -> Iterable[Any]:
    if seq is None:
        return ()
    return seq


def _extract_graph(model_or_graph: Any) -> Any:
    """Return ``model_or_graph.graph`` when given a ``ModelProto``."""

    if _looks_like_graph(model_or_graph):
        return model_or_graph
    graph = getattr(model_or_graph, "graph", None)
    if graph is not None and _looks_like_graph(graph):
        return graph
    raise TypeError(
        f"to_mermaid expected a ModelProto or GraphProto, got {type(model_or_graph).__name__}."
    )


def _looks_like_graph(obj: Any) -> bool:
    return (
        hasattr(obj, "node")
        and hasattr(obj, "input")
        and hasattr(obj, "output")
        and not hasattr(obj, "graph")
    )
