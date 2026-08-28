"""Translate an ONNX model or graph into Python code that rebuilds it.

Two output *flavours* (``api``) are supported:

* ``"onnx-compact"`` -- a single nested expression building the model with
  :mod:`onnx_light.onnx.helper` (``oh.make_model(oh.make_graph([...], ...))``),
  mirroring the *onnx-compact* API of
  `yet-another-onnx-builder <https://github.com/xadupre/yet-another-onnx-builder>`_.
* ``"builder"`` -- a plain Python script that rebuilds the same model with the
  incremental :class:`onnx_light.onnx_core.graph_builder.GraphBuilder`
  (``g.inp(...)``, ``g.init(...)``, ``g.op.<operator>(...)``,
  ``g.out(...)``, ``g.to_onnx(...)``).

Both flavours are pure Python and only rely on the attributes of the standard
ONNX message types (``ModelProto``, ``GraphProto``, ``NodeProto``,
``ValueInfoProto``, ``TensorProto``, ``AttributeProto`` and
``TensorShapeProto``); they therefore work both with messages built by
:mod:`onnx_light` and with messages built by the upstream :mod:`onnx` package.

Example::

    from onnx_light.tools import translate, translate_header

    code = translate_header("onnx-compact") + translate(model, api="onnx-compact")
    print(code)
"""

from __future__ import annotations

import keyword
import textwrap
from typing import Any

import numpy as np

from ._proto_utils import (
    _dtype_enum_name,
    _extract_graph,
    _elem_type,
    _iter,
    _opsets,
    _s,
    _shape_tuple,
    _tensor_to_numpy,
)

__all__ = ["translate", "translate_header"]


# ---------------------------------------------------------------------------
# AttributeProto type ids (mirror ``onnx.AttributeProto.AttributeType``).
# ---------------------------------------------------------------------------
_ATTR_FLOAT = 1
_ATTR_INT = 2
_ATTR_STRING = 3
_ATTR_TENSOR = 4
_ATTR_GRAPH = 5
_ATTR_FLOATS = 6
_ATTR_INTS = 7
_ATTR_STRINGS = 8


# ---------------------------------------------------------------------------
# Value extraction (duck typed against ``TensorProto``)
# ---------------------------------------------------------------------------
def _dtype_expr(array: np.ndarray) -> str:
    """Returns a Python expression naming the numpy dtype of ``array``."""
    repl = {"bool": "bool_", "object": "object_", "str": "str_"}
    sdtype = str(array.dtype)
    sdtype = repl.get(sdtype, sdtype)
    return f"np.{sdtype}" if hasattr(np, sdtype) else f"ml_dtypes.{sdtype}"


def _array_expr(array: np.ndarray) -> str:
    """Returns ``np.array(<values>, dtype=<dtype>)`` for ``array``."""
    return f"np.array({array.tolist()!r}, dtype={_dtype_expr(array)})"


def _from_array_expr(tensor: Any, name: str) -> str:
    """Returns ``onh.from_array(np.array(...), name=...)`` for a ``TensorProto``."""
    array = _tensor_to_numpy(tensor)
    return f"onh.from_array({_array_expr(array)}, name={name!r})"


# ---------------------------------------------------------------------------
# Shapes and value infos
# ---------------------------------------------------------------------------
def _value_info_expr(name: str, elem_type: int, shape: tuple | None) -> str:
    """Returns an ``oh.make_tensor_value_info(...)`` expression."""
    tp = f"onnx.TensorProto.{_dtype_enum_name(elem_type)}"
    if elem_type and shape is not None:
        return f"oh.make_tensor_value_info({name!r}, {tp}, {shape!r})"
    if elem_type:
        return f"oh.make_tensor_value_info({name!r}, {tp}, [])"
    return f"oh.make_tensor_value_info({name!r}, onnx.TensorProto.UNDEFINED, [])"


# ---------------------------------------------------------------------------
# Attribute rendering
# ---------------------------------------------------------------------------
def _attr_value_expr(attr: Any) -> str:
    """Returns a Python expression for the value carried by ``attr``."""
    attr_type = int(getattr(attr, "type", 0) or 0)
    ref_attr_name = _s(getattr(attr, "ref_attr_name", "") or "")
    if ref_attr_name:
        return (
            f"oh.make_attribute_ref({_s(getattr(attr, 'name', ''))!r}, "
            f"{attr_type}, {ref_attr_name!r})"
        )
    if attr_type == _ATTR_INT:
        return str(int(getattr(attr, "i", 0) or 0))
    if attr_type == _ATTR_FLOAT:
        return repr(float(getattr(attr, "f", 0.0) or 0.0))
    if attr_type == _ATTR_STRING:
        return repr(_s(getattr(attr, "s", b"")))
    if attr_type == _ATTR_INTS:
        return repr([int(v) for v in _iter(getattr(attr, "ints", None))])
    if attr_type == _ATTR_FLOATS:
        return repr([float(v) for v in _iter(getattr(attr, "floats", None))])
    if attr_type == _ATTR_STRINGS:
        return repr([_s(v) for v in _iter(getattr(attr, "strings", None))])
    if attr_type == _ATTR_TENSOR:
        tensor = getattr(attr, "t", None)
        if tensor is None:
            raise NotImplementedError("Tensor attribute without a value.")
        return _from_array_expr(tensor, _s(getattr(tensor, "name", "")) or "")
    if attr_type == _ATTR_GRAPH:
        graph = getattr(attr, "g", None)
        if graph is None:
            raise NotImplementedError("Graph attribute without a value.")
        return _graph_expr(graph, indent="        ")
    raise NotImplementedError(
        f"Attribute {_s(getattr(attr, 'name', ''))!r} of type {attr_type} "
        f"cannot be translated yet."
    )


def _node_attributes(node: Any) -> list[tuple[str, str]]:
    """Returns ``[(name, value_expr), ...]`` for the attributes of ``node``."""
    result: list[tuple[str, str]] = []
    for attr in _iter(getattr(node, "attribute", None)):
        name = _s(getattr(attr, "name", ""))
        result.append((name, _attr_value_expr(attr)))
    return result


# ---------------------------------------------------------------------------
# Node rendering (shared: ``oh.make_node`` expression)
# ---------------------------------------------------------------------------
def _node_expr(node: Any) -> str:
    """Returns an ``oh.make_node(...)`` expression for ``node``."""
    op_type = _s(getattr(node, "op_type", ""))
    inputs = [_s(i) for i in _iter(getattr(node, "input", None))]
    outputs = [_s(o) for o in _iter(getattr(node, "output", None))]
    domain = _s(getattr(node, "domain", "") or "")
    args = [f"{op_type!r}", repr(inputs), repr(outputs)]
    if domain:
        args.append(f"domain={domain!r}")
    for name, value in _node_attributes(node):
        args.append(f"{name}={value}")
    return f"oh.make_node({', '.join(args)})"


def _builder_value_info_args(value_info: Any) -> list[str]:
    """Returns compact builder arguments for a value info."""
    return [
        repr(_s(getattr(value_info, "name", ""))),
        f"onnx.TensorProto.{_dtype_enum_name(_elem_type(value_info))}",
        repr(_shape_tuple(value_info)),
    ]


# ---------------------------------------------------------------------------
# Graph rendering (nested ``oh.make_graph`` expression, used for subgraphs)
# ---------------------------------------------------------------------------
def _graph_expr(graph: Any, indent: str) -> str:
    """Returns a nested ``oh.make_graph(...)`` expression for ``graph``."""
    inner = indent + "    "
    item = inner + "    "
    name = _s(getattr(graph, "name", "")) or "graph"

    nodes = [_node_expr(n) for n in _iter(getattr(graph, "node", None))]
    inputs = [
        _value_info_expr(_s(getattr(i, "name", "")), _elem_type(i), _shape_tuple(i))
        for i in _iter(getattr(graph, "input", None))
    ]
    outputs = [
        _value_info_expr(_s(getattr(o, "name", "")), _elem_type(o), _shape_tuple(o))
        for o in _iter(getattr(graph, "output", None))
    ]
    initializers = [
        _from_array_expr(init, _s(getattr(init, "name", "")))
        for init in _iter(getattr(graph, "initializer", None))
    ]

    def _block(items: list[str]) -> list[str]:
        return [inner + "[", *[f"{item}{it}," for it in items], inner + "],"]

    lines = [
        "oh.make_graph(",
        *_block(nodes),
        f"{inner}{name!r},",
        *_block(inputs),
        *_block(outputs),
    ]
    if initializers:
        lines.extend(_block(initializers))
    lines.append(indent + ")")
    return ("\n").join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def translate_header(api: str = "onnx-compact") -> str:
    """Returns the import header required by the code produced by :func:`translate`.

    Args:
        api: target flavour, ``"onnx-compact"`` or ``"builder"``.

    Returns:
        The import header as a string ending with a trailing newline.
    """
    if api == "onnx-compact":
        return textwrap.dedent("""\
            import numpy as np
            import ml_dtypes
            import onnx_light.onnx as onnx
            import onnx_light.onnx.helper as oh
            import onnx_light.onnx.numpy_helper as onh
            """)
    if api == "builder":
        return textwrap.dedent("""\
            import numpy as np
            import ml_dtypes
            import onnx_light.onnx as onnx
            import onnx_light.onnx.helper as oh
            import onnx_light.onnx.numpy_helper as onh
            from onnx_light.onnx_core.graph_builder import GraphBuilder
            """)
    raise ValueError(f"Unexpected value {api!r} for api.")


def _translate_compact(model: Any, graph: Any) -> str:
    """Translates a model/graph into the ``onnx-compact`` flavour."""
    name = _s(getattr(graph, "name", "")) or "graph"
    ir_version = int(getattr(model, "ir_version", 0) or 0)
    opsets = _opsets(model)

    nodes = [_node_expr(n) for n in _iter(getattr(graph, "node", None))]
    inputs = [
        _value_info_expr(_s(getattr(i, "name", "")), _elem_type(i), _shape_tuple(i))
        for i in _iter(getattr(graph, "input", None))
    ]
    outputs = [
        _value_info_expr(_s(getattr(o, "name", "")), _elem_type(o), _shape_tuple(o))
        for o in _iter(getattr(graph, "output", None))
    ]
    initializers = [
        _from_array_expr(init, _s(getattr(init, "name", "")))
        for init in _iter(getattr(graph, "initializer", None))
    ]

    lines = ["model = oh.make_model(", "    oh.make_graph("]
    lines.append("        [")
    for node in nodes:
        lines.append(f"            {node},")
    lines.append("        ],")
    lines.append(f"        {name!r},")
    lines.append("        [")
    for inp in inputs:
        lines.append(f"            {inp},")
    lines.append("        ],")
    lines.append("        [")
    for out in outputs:
        lines.append(f"            {out},")
    lines.append("        ],")
    if initializers:
        lines.append("        [")
        for init in initializers:
            lines.append(f"            {init},")
        lines.append("        ],")
    lines.append("    ),")
    opset_str = ", ".join(f"oh.make_opsetid({d!r}, {v})" for d, v in opsets)
    lines.append(f"    opset_imports=[{opset_str}],")
    if ir_version:
        lines.append(f"    ir_version={ir_version},")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _translate_builder(model: Any, graph: Any) -> str:
    """Translates a model/graph into the ``builder`` (GraphBuilder) flavour."""
    name = _s(getattr(graph, "name", "")) or "graph"
    ir_version = int(getattr(model, "ir_version", 0) or 0)
    opsets = _opsets(model)

    initializer_names = {
        _s(getattr(init, "name", "")) for init in _iter(getattr(graph, "initializer", None))
    }

    lines = [f"g = GraphBuilder({name!r})"]
    for domain, version in opsets:
        lines.append(f"g.set_opset_version({domain!r}, {version})")

    for inp in _iter(getattr(graph, "input", None)):
        input_name = _s(getattr(inp, "name", ""))
        if input_name in initializer_names:
            # An initializer that is also listed as a graph input; the builder
            # declares it once as an initializer below.
            continue
        args = _builder_value_info_args(inp)
        lines.append(f"g.inp({', '.join(args)})")

    for init in _iter(getattr(graph, "initializer", None)):
        array = _tensor_to_numpy(init)
        name = _s(getattr(init, "name", ""))
        lines.append(f"g.init({_array_expr(array)}, name={name!r})")

    for node in _iter(getattr(graph, "node", None)):
        op_type = _s(getattr(node, "op_type", ""))
        inputs = [_s(i) for i in _iter(getattr(node, "input", None))]
        outputs = [_s(o) for o in _iter(getattr(node, "output", None))]
        domain = _s(getattr(node, "domain", "") or "")
        args = [*[repr(value) for value in inputs], f"outputs={outputs!r}"]
        if domain:
            args.append(f"domain={domain!r}")
        attributes = _node_attributes(node)
        direct_attributes = []
        indirect_attributes = []
        for name, value in attributes:
            if (
                name.isidentifier()
                and not keyword.iskeyword(name)
                and name not in {"domain", "name", "outputs"}
            ):
                direct_attributes.append((name, value))
            else:
                indirect_attributes.append((name, value))
        args.extend(f"{name}={value}" for name, value in direct_attributes)
        if indirect_attributes:
            attrs = ", ".join(f"{name!r}: {value}" for name, value in indirect_attributes)
            args.append(f"**{{{attrs}}}")
        lines.append(f"g.op.{op_type}({', '.join(args)})")

    for out in _iter(getattr(graph, "output", None)):
        elem_type = _elem_type(out)
        if elem_type:
            lines.append(f"g.out({', '.join(_builder_value_info_args(out))})")
        else:
            lines.append(f"g.out({_s(getattr(out, 'name', ''))!r})")

    if ir_version:
        lines.append(f"model = g.to_onnx('model', ir_version={ir_version})")
    else:
        lines.append("model = g.to_onnx('model')")
    return "\n".join(lines) + "\n"


def translate(proto: Any, api: str = "onnx-compact") -> str:
    """Translates an ONNX model or graph into Python code that rebuilds it.

    Args:
        proto: a ``ModelProto`` or ``GraphProto`` (or a file path to load).
        api: target flavour, ``"onnx-compact"`` (default) or ``"builder"``.

    Returns:
        The generated Python code as a string (without the import header, see
        :func:`translate_header`).
    """
    if isinstance(proto, str):
        from ..onnx import load

        proto = load(proto)

    graph = _extract_graph(proto)
    model = proto

    if api == "onnx-compact":
        return _translate_compact(model, graph)
    if api == "builder":
        return _translate_builder(model, graph)
    raise ValueError(f"Unexpected value {api!r} for api.")
