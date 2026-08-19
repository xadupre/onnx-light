"""Translate an ONNX model or graph into Python code that rebuilds it.

Two output *flavours* (``api``) are supported:

* ``"onnx-compact"`` -- a single nested expression building the model with
  :mod:`onnx_light.onnx.helper` (``oh.make_model(oh.make_graph([...], ...))``),
  mirroring the *onnx-compact* API of
  `yet-another-onnx-builder <https://github.com/xadupre/yet-another-onnx-builder>`_.
* ``"builder"`` -- a plain Python script that rebuilds the same model with the
  incremental :class:`onnx_light.onnx_core.graph_builder.GraphBuilder`
  (``g.make_input(...)``, ``g.make_node(...)``, ``g.make_output(...)``,
  ``g.to_onnx(...)``).

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

import textwrap
from typing import Any

import ml_dtypes
import numpy as np

from ._proto_utils import _extract_graph, _iter, _s

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


# Mapping ``TensorProto.DataType`` -> numpy (or ``ml_dtypes``) scalar type.
_DTYPE_TO_NUMPY = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
    14: np.complex64,
    15: np.complex128,
    16: ml_dtypes.bfloat16,
    17: ml_dtypes.float8_e4m3fn,
    18: ml_dtypes.float8_e4m3fnuz,
    19: ml_dtypes.float8_e5m2,
    20: ml_dtypes.float8_e5m2fnuz,
    21: ml_dtypes.uint4,
    22: ml_dtypes.int4,
    23: ml_dtypes.float4_e2m1fn,
}

# ``TensorProto.DataType`` name for each id (mirror ``onnx.TensorProto``).
_DTYPE_ENUM_NAME = {
    0: "UNDEFINED",
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
    17: "FLOAT8E4M3FN",
    18: "FLOAT8E4M3FNUZ",
    19: "FLOAT8E5M2",
    20: "FLOAT8E5M2FNUZ",
    21: "UINT4",
    22: "INT4",
    23: "FLOAT4E2M1",
}

# Typed repeated field holding the values of a ``TensorProto`` (when the data
# is not stored in ``raw_data``).
_DTYPE_TO_FIELD = {
    1: "float_data",
    2: "int32_data",
    3: "int32_data",
    4: "int32_data",
    5: "int32_data",
    6: "int32_data",
    7: "int64_data",
    9: "int32_data",
    10: "int32_data",
    11: "double_data",
    12: "uint64_data",
    13: "uint64_data",
    14: "float_data",
    15: "double_data",
}


# ---------------------------------------------------------------------------
# Value extraction (duck typed against ``TensorProto``)
# ---------------------------------------------------------------------------
def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Returns the content of a ``TensorProto``-like object as a numpy array."""
    data_type = int(getattr(tensor, "data_type", 0) or 0)
    dims = [int(d) for d in _iter(getattr(tensor, "dims", None))]

    if data_type == 8:  # STRING
        strings = [_s(v) for v in _iter(getattr(tensor, "string_data", None))]
        array = np.array(strings, dtype=object)
        return array.reshape(dims) if dims else array.reshape(())

    if data_type not in _DTYPE_TO_NUMPY:
        raise NotImplementedError(f"Unable to translate a tensor with data_type={data_type}.")
    np_dtype = _DTYPE_TO_NUMPY[data_type]

    raw = getattr(tensor, "raw_data", b"") or b""
    if raw:
        array = np.frombuffer(raw, dtype=np_dtype)
        return array.reshape(dims) if dims else array.reshape(())

    field = _DTYPE_TO_FIELD.get(data_type)
    if field is None:
        raise NotImplementedError(
            f"Unable to translate a tensor with data_type={data_type} and no raw_data."
        )
    values = list(_iter(getattr(tensor, field, None)))
    if data_type in (10, 16, 17, 18, 19, 20, 21, 22, 23):
        # Small float / sub-byte types are packed as their raw bit pattern in
        # ``int32_data``; reinterpret the bits instead of casting the value.
        storage = {10: np.uint16, 16: np.uint16}.get(data_type, np.uint8)
        array = np.array(values, dtype=storage).view(np_dtype)
    else:
        array = np.array(values, dtype=np_dtype)
    return array.reshape(dims) if dims else array.reshape(())


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
def _shape_tuple(value_info: Any) -> tuple | None:
    """Returns the shape of a ``ValueInfoProto`` as a tuple, or ``None``."""
    type_proto = getattr(value_info, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None)
    if tensor_type is None:
        return None
    shape = getattr(tensor_type, "shape", None)
    if shape is None:
        return None
    dims = list(_iter(getattr(shape, "dim", None)))
    result: list = []
    for dim in dims:
        dim_param = _s(getattr(dim, "dim_param", "") or "")
        dim_value = int(getattr(dim, "dim_value", 0) or 0)
        if dim_param:
            result.append(dim_param)
        elif dim_value != 0:
            result.append(dim_value)
        else:
            result.append(None)
    return tuple(result)


def _elem_type(value_info: Any) -> int:
    """Returns the element type of a ``ValueInfoProto``."""
    type_proto = getattr(value_info, "type", None)
    tensor_type = getattr(type_proto, "tensor_type", None)
    return int(getattr(tensor_type, "elem_type", 0) or 0)


def _value_info_expr(name: str, elem_type: int, shape: tuple | None) -> str:
    """Returns an ``oh.make_tensor_value_info(...)`` expression."""
    tp = f"onnx.TensorProto.{_DTYPE_ENUM_NAME.get(elem_type, 'UNDEFINED')}"
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


def _opsets(model_or_graph: Any) -> list[tuple[str, int]]:
    """Returns the ``[(domain, version), ...]`` opset imports of a model."""
    result: list[tuple[str, int]] = []
    for opset in _iter(getattr(model_or_graph, "opset_import", None)):
        domain = _s(getattr(opset, "domain", "") or "")
        version = int(getattr(opset, "version", 0) or 0)
        result.append((domain, version))
    return result


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
        expr = _value_info_expr(input_name, _elem_type(inp), _shape_tuple(inp))
        lines.append(f"g.make_input({expr})")

    for init in _iter(getattr(graph, "initializer", None)):
        expr = _from_array_expr(init, _s(getattr(init, "name", "")))
        lines.append(f"g.make_initializer({expr})")

    for node in _iter(getattr(graph, "node", None)):
        op_type = _s(getattr(node, "op_type", ""))
        inputs = [_s(i) for i in _iter(getattr(node, "input", None))]
        outputs = [_s(o) for o in _iter(getattr(node, "output", None))]
        domain = _s(getattr(node, "domain", "") or "")
        args = [f"{op_type!r}", repr(inputs), f"outputs={outputs!r}"]
        if domain:
            args.append(f"domain={domain!r}")
        attributes = _node_attributes(node)
        if attributes:
            attrs = ", ".join(f"{k!r}: {v}" for k, v in attributes)
            args.append(f"attributes={{{attrs}}}")
        lines.append(f"g.make_node({', '.join(args)})")

    for out in _iter(getattr(graph, "output", None)):
        expr = _value_info_expr(_s(getattr(out, "name", "")), _elem_type(out), _shape_tuple(out))
        lines.append(f"g.make_output({expr})")

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
