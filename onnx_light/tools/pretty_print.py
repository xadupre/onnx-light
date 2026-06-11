"""Pretty-print an ONNX model, graph, function or node as human-readable text.

The output is loosely based on the ONNX text representation produced by
:func:`onnx.printer.to_text` upstream: each operator is rendered on a
single line as ``outputs = op_type<attrs>(inputs)`` and the surrounding
graph is wrapped with its name, inputs and outputs annotated with their
type and shape.

The converter is implemented in pure Python and is duck-typed against the
standard ONNX message API (``ModelProto``, ``GraphProto``, ``NodeProto``,
``ValueInfoProto``, ``TensorProto``, ``AttributeProto`` and
``TensorShapeProto``).  It therefore works both with messages built by
:mod:`onnx_light` and with messages built by the upstream :mod:`onnx`
package.

Example::

    from onnx_light.tools import pretty_print

    print(pretty_print(model))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .mermaid import _dtype_name, _format_shape, _s

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only.
    from collections.abc import Iterable


__all__ = ["pretty_print", "pretty_print_graph", "pretty_print_node"]


_INDENT = "  "


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


# Mirrors ``onnx.AttributeProto.AttributeType`` values without importing
# ``onnx``.  Only the subset of types actually serialised by attributes
# is needed here; unknown values fall back to a generic representation.
_ATTR_FLOAT = 1
_ATTR_INT = 2
_ATTR_STRING = 3
_ATTR_TENSOR = 4
_ATTR_GRAPH = 5
_ATTR_SPARSE_TENSOR = 11
_ATTR_TYPE_PROTO = 13
_ATTR_FLOATS = 6
_ATTR_INTS = 7
_ATTR_STRINGS = 8
_ATTR_TENSORS = 9
_ATTR_GRAPHS = 10
_ATTR_SPARSE_TENSORS = 12
_ATTR_TYPE_PROTOS = 14


def _format_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            text = value.decode("utf-8", errors="replace")
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, float):
        # Keep the textual form compact yet round-trippable.
        return repr(value)
    return str(value)


def _format_string(value: Any) -> str:
    text = _s(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_tensor(tensor: Any) -> str:
    """Return a compact textual representation of a :class:`TensorProto`."""

    if tensor is None:
        return "<tensor>"
    dtype = _dtype_name(int(getattr(tensor, "data_type", 0) or 0))
    dims = list(getattr(tensor, "dims", []) or [])
    shape = ",".join(str(int(d)) for d in dims)
    name = _s(getattr(tensor, "name", "")) or ""
    label = f"{dtype}[{shape}]" if shape else f"{dtype}[]"
    if name:
        return f"{label} {name}"
    return label


def _format_attribute(attr: Any) -> str:
    name = _s(getattr(attr, "name", "")) or ""
    attr_type = int(getattr(attr, "type", 0) or 0)

    if attr_type == _ATTR_FLOAT:
        return f"{name} = {_format_scalar(float(getattr(attr, 'f', 0.0)))}"
    if attr_type == _ATTR_INT:
        return f"{name} = {int(getattr(attr, 'i', 0))}"
    if attr_type == _ATTR_STRING:
        return f"{name} = {_format_string(getattr(attr, 's', b''))}"
    if attr_type == _ATTR_TENSOR:
        return f"{name} = {_format_tensor(getattr(attr, 't', None))}"
    if attr_type == _ATTR_GRAPH:
        return f"{name} = <graph>"
    if attr_type == _ATTR_SPARSE_TENSOR:
        return f"{name} = <sparse_tensor>"
    if attr_type == _ATTR_TYPE_PROTO:
        return f"{name} = {_format_shape(getattr(attr, 'tp', None)) or '<type>'}"
    if attr_type == _ATTR_FLOATS:
        items = ", ".join(_format_scalar(float(v)) for v in getattr(attr, "floats", []) or [])
        return f"{name} = [{items}]"
    if attr_type == _ATTR_INTS:
        items = ", ".join(str(int(v)) for v in getattr(attr, "ints", []) or [])
        return f"{name} = [{items}]"
    if attr_type == _ATTR_STRINGS:
        items = ", ".join(_format_string(v) for v in getattr(attr, "strings", []) or [])
        return f"{name} = [{items}]"
    if attr_type == _ATTR_TENSORS:
        items = ", ".join(_format_tensor(v) for v in getattr(attr, "tensors", []) or [])
        return f"{name} = [{items}]"
    if attr_type == _ATTR_GRAPHS:
        return f"{name} = [<graph> x {len(getattr(attr, 'graphs', []) or [])}]"
    if attr_type == _ATTR_SPARSE_TENSORS:
        return (
            f"{name} = [<sparse_tensor> x " f"{len(getattr(attr, 'sparse_tensors', []) or [])}]"
        )
    if attr_type == _ATTR_TYPE_PROTOS:
        items = ", ".join(
            _format_shape(v) or "<type>" for v in getattr(attr, "type_protos", []) or []
        )
        return f"{name} = [{items}]"

    # Fall back to whatever single scalar field happens to be set.  This
    # also covers attributes built without an explicit ``type`` value
    # (e.g. constructed by hand in tests).
    for field, formatter in (
        ("f", lambda v: _format_scalar(float(v))),
        ("i", lambda v: str(int(v))),
        ("s", _format_string),
        ("t", _format_tensor),
    ):
        if getattr(attr, field, None):
            return f"{name} = {formatter(getattr(attr, field))}"
    return name


def _format_attributes(attributes: Iterable[Any]) -> str:
    attrs = list(attributes or [])
    if not attrs:
        return ""
    return "<" + ", ".join(_format_attribute(a) for a in attrs) + ">"


# ---------------------------------------------------------------------------
# Node / Graph
# ---------------------------------------------------------------------------


def _format_name(value: Any) -> str:
    text = _s(value)
    return text if text else '""'


def pretty_print_node(node: Any) -> str:
    """Return a one-line textual representation of a ``NodeProto``."""

    inputs = ", ".join(_format_name(i) for i in getattr(node, "input", []) or [])
    outputs = ", ".join(_format_name(o) for o in getattr(node, "output", []) or [])
    op_type = _s(getattr(node, "op_type", "")) or "?"
    domain = _s(getattr(node, "domain", ""))
    qualified = f"{domain}.{op_type}" if domain else op_type
    attrs = _format_attributes(getattr(node, "attribute", []) or [])

    name = _s(getattr(node, "name", ""))
    prefix = f"# {name}\n" if name else ""
    return f"{prefix}{outputs} = {qualified}{attrs}({inputs})"


def _format_value_info(value_info: Any) -> str:
    name = _s(getattr(value_info, "name", "")) or ""
    type_text = _format_shape(getattr(value_info, "type", None))
    if type_text and name:
        return f"{type_text} {name}"
    return type_text or name or '""'


def _format_initializer(tensor: Any) -> str:
    name = _s(getattr(tensor, "name", "")) or ""
    dtype = _dtype_name(int(getattr(tensor, "data_type", 0) or 0))
    dims = list(getattr(tensor, "dims", []) or [])
    shape = ",".join(str(int(d)) for d in dims)
    label = f"{dtype}[{shape}]" if shape else f"{dtype}[]"
    if name:
        return f"{label} {name}"
    return label


def pretty_print_graph(graph: Any, indent: str = "") -> str:
    """Return a multi-line textual representation of a ``GraphProto``."""

    inputs = list(getattr(graph, "input", []) or [])
    outputs = list(getattr(graph, "output", []) or [])
    initializers = list(getattr(graph, "initializer", []) or [])
    nodes = list(getattr(graph, "node", []) or [])
    name = _s(getattr(graph, "name", "")) or ""

    header_inputs = ", ".join(_format_value_info(i) for i in inputs)
    header_outputs = ", ".join(_format_value_info(o) for o in outputs)

    lines: list[str] = []
    title = name if name else "graph"
    lines.append(f"{indent}graph {title} ({header_inputs}) => ({header_outputs}) {{")

    body_indent = indent + _INDENT
    if initializers:
        lines.append(f"{body_indent}# initializers")
        for init in initializers:
            lines.append(f"{body_indent}{_format_initializer(init)}")
        lines.append("")
    for node in nodes:
        for line in pretty_print_node(node).splitlines():
            lines.append(f"{body_indent}{line}")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def _format_opsets(opsets: Iterable[Any]) -> list[str]:
    lines: list[str] = []
    for opset in opsets or []:
        domain = _s(getattr(opset, "domain", "")) or "ai.onnx"
        version = int(getattr(opset, "version", 0) or 0)
        lines.append(f'opset_import: "{domain}" : {version}')
    return lines


def _format_function(fn: Any) -> str:
    name = _s(getattr(fn, "name", "")) or ""
    domain = _s(getattr(fn, "domain", "")) or ""
    qualified = f"{domain}.{name}" if domain else name
    inputs = ", ".join(_format_name(i) for i in getattr(fn, "input", []) or [])
    outputs = ", ".join(_format_name(o) for o in getattr(fn, "output", []) or [])
    attrs = list(getattr(fn, "attribute", []) or [])
    attr_text = f"<{', '.join(_s(a) for a in attrs)}>" if attrs else ""

    lines: list[str] = [f"function {qualified}{attr_text}({inputs}) => ({outputs}) {{"]
    for node in getattr(fn, "node", []) or []:
        for line in pretty_print_node(node).splitlines():
            lines.append(f"{_INDENT}{line}")
    lines.append("}")
    return "\n".join(lines)


def pretty_print(model_or_graph: Any) -> str:
    """Return a human-readable text representation of an ONNX object.

    The argument may be a ``ModelProto``, a ``GraphProto``, a
    ``FunctionProto`` or a ``NodeProto``; any other object is rendered by
    delegating to its ``__str__`` method.
    """

    # NodeProto: identified by the presence of ``op_type`` and ``input``.
    if hasattr(model_or_graph, "op_type") and hasattr(model_or_graph, "input"):
        return pretty_print_node(model_or_graph)

    # ModelProto: has a single ``graph`` attribute that itself looks like
    # a GraphProto (it has a ``node`` field).
    graph_attr = getattr(model_or_graph, "graph", None)
    if graph_attr is not None and hasattr(graph_attr, "node"):
        lines: list[str] = []
        ir_version = getattr(model_or_graph, "ir_version", None)
        if ir_version:
            lines.append(f"ir_version: {int(ir_version)}")
        producer = _s(getattr(model_or_graph, "producer_name", ""))
        if producer:
            lines.append(f'producer_name: "{producer}"')
        producer_version = _s(getattr(model_or_graph, "producer_version", ""))
        if producer_version:
            lines.append(f'producer_version: "{producer_version}"')
        domain = _s(getattr(model_or_graph, "domain", ""))
        if domain:
            lines.append(f'domain: "{domain}"')
        model_version = getattr(model_or_graph, "model_version", None)
        if model_version:
            lines.append(f"model_version: {int(model_version)}")
        lines.extend(_format_opsets(getattr(model_or_graph, "opset_import", []) or []))
        if lines:
            lines.append("")
        lines.append(pretty_print_graph(graph_attr))
        for fn in getattr(model_or_graph, "functions", []) or []:
            lines.append("")
            lines.append(_format_function(fn))
        return "\n".join(lines)

    # GraphProto: has ``node`` directly.
    if hasattr(model_or_graph, "node"):
        return pretty_print_graph(model_or_graph)

    # FunctionProto: has ``node`` and ``op_type`` is missing but ``name``
    # and ``input`` are present.  This branch is reached when the model
    # check above does not apply because there is no ``graph`` attribute.
    if hasattr(model_or_graph, "input") and hasattr(model_or_graph, "output"):
        return _format_function(model_or_graph)

    return str(model_or_graph)
