"""ONNX → NNEF graph exporter.

The exporter walks an ONNX :class:`ModelProto` and emits a NNEF
``graph.nnef`` text file plus one ``*.dat`` file per initializer.
Operator translation is performed by small "converter" functions that
return a list of NNEF statements.  Users can extend the set of supported
operators with :func:`register_op_converter`.

The implementation is fully written in Python and only requires
:mod:`numpy`; it does not depend on the native :mod:`onnx_light` C
extensions nor on the upstream :mod:`onnx` package.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .tensor_io import write_nnef_tensor


class NNEFExportError(RuntimeError):
    """Raised when an ONNX construct cannot be expressed in NNEF."""


# ---------------------------------------------------------------------------
# ONNX attribute / tensor helpers (duck-typed on onnx and onnx_light protos).
# ---------------------------------------------------------------------------

# ONNX AttributeProto.AttributeType codes (stable enum values).
_ATTR_UNDEFINED = 0
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

# ONNX TensorProto data-type codes → numpy dtypes.
_ONNX_DTYPE_TO_NUMPY: dict[int, np.dtype] = {
    1: np.dtype(np.float32),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int8),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int16),
    6: np.dtype(np.int32),
    7: np.dtype(np.int64),
    9: np.dtype(np.bool_),
    10: np.dtype(np.float16),
    11: np.dtype(np.float64),
    12: np.dtype(np.uint32),
    13: np.dtype(np.uint64),
}


def _attr_value(attr: Any) -> Any:
    """Returns the Python value of an ONNX ``AttributeProto``."""
    t = getattr(attr, "type", _ATTR_UNDEFINED)
    if t == _ATTR_INT:
        return int(attr.i)
    if t == _ATTR_FLOAT:
        return float(attr.f)
    if t == _ATTR_STRING:
        s = attr.s
        return s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
    if t == _ATTR_INTS:
        return [int(v) for v in attr.ints]
    if t == _ATTR_FLOATS:
        return [float(v) for v in attr.floats]
    if t == _ATTR_STRINGS:
        return [
            v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            for v in attr.strings
        ]
    if t == _ATTR_TENSOR:
        return _tensor_to_numpy(attr.t)
    raise NNEFExportError(f"Unsupported ONNX attribute type: {t}")


def _attrs_to_dict(node: Any) -> dict[str, Any]:
    """Returns ``{attr_name: value}`` for ``node.attribute``."""
    return {a.name: _attr_value(a) for a in node.attribute}


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Converts an ONNX ``TensorProto`` to a :class:`numpy.ndarray`.

    Supports both ``raw_data`` and typed list fields, as well as
    ``external_data`` (only when the data is already materialised in
    ``raw_data`` by the caller — the exporter does not chase external
    files on disk).
    """
    dtype = _ONNX_DTYPE_TO_NUMPY.get(int(tensor.data_type))
    if dtype is None:
        raise NNEFExportError(f"Unsupported ONNX tensor data_type={tensor.data_type}")
    dims = tuple(int(d) for d in tensor.dims)
    raw = bytes(tensor.raw_data) if getattr(tensor, "raw_data", b"") else b""
    if raw:
        array = np.frombuffer(raw, dtype=dtype).copy()
    else:
        for field_name, np_dtype in (
            ("float_data", np.float32),
            ("int32_data", np.int32),
            ("int64_data", np.int64),
            ("double_data", np.float64),
            ("uint64_data", np.uint64),
        ):
            data = list(getattr(tensor, field_name, []))
            if data:
                array = np.asarray(data, dtype=np_dtype).astype(dtype, copy=False)
                break
        else:
            array = np.zeros(int(np.prod(dims)) if dims else 1, dtype=dtype)
    if dims:
        array = array.reshape(dims)
    else:
        array = array.reshape(())
    return array


# ---------------------------------------------------------------------------
# NNEF identifier and literal formatting.
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")
_RESERVED = {
    "graph",
    "fragment",
    "tensor",
    "integer",
    "scalar",
    "logical",
    "string",
    "true",
    "false",
    "if",
    "else",
    "for",
    "do",
    "while",
    "yield",
    "extension",
    "version",
    "external",
    "variable",
    "constant",
    "shape",
    "label",
    "dtype",
    "value",
}


def _to_identifier(name: str, used: dict[str, str]) -> str:
    """Returns a unique NNEF identifier mapped from ``name``."""
    if name in used:
        return used[name]
    cleaned = _IDENT_RE.sub("_", name) if name else "t"
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = "t_" + cleaned
    if cleaned in _RESERVED:
        cleaned = cleaned + "_"
    base = cleaned
    suffix = 0
    existing = set(used.values())
    while cleaned in existing:
        suffix += 1
        cleaned = f"{base}_{suffix}"
    used[name] = cleaned
    return cleaned


def _format_scalar(value: Any) -> str:
    """Formats a Python scalar as a NNEF literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise NNEFExportError(f"Non-finite literal {value!r} cannot be encoded in NNEF")
        return f"{float(value):g}"
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    raise NNEFExportError(f"Cannot format value {value!r} as NNEF literal")


def _format_list(values: Iterable[Any]) -> str:
    return "[" + ", ".join(_format_value(v) for v in values) + "]"


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return _format_list(value)
    return _format_scalar(value)


# ---------------------------------------------------------------------------
# Exporter context shared by op converters.
# ---------------------------------------------------------------------------


@dataclass
class _Tensor:
    """Metadata about a tensor used inside the graph."""

    onnx_name: str
    nnef_name: str
    shape: tuple[int, ...] | None = None
    dtype: np.dtype | None = None
    is_initializer: bool = False
    initializer_array: np.ndarray | None = None


@dataclass
class NNEFGraph:
    """In-memory representation of a NNEF graph ready to be serialised."""

    name: str
    inputs: list[str]
    outputs: list[str]
    statements: list[str] = field(default_factory=list)
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    version: tuple[int, int] = (1, 0)
    extensions: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Returns the textual ``graph.nnef`` representation."""
        major, minor = self.version
        lines = [f"version {major}.{minor};"]
        for ext in self.extensions:
            lines.append(f"extension {ext};")
        lines.append("")
        params = ", ".join(self.inputs) if self.inputs else ""
        results = ", ".join(self.outputs) if self.outputs else ""
        lines.append(f"graph {self.name}({params}) -> ({results})")
        lines.append("{")
        for stmt in self.statements:
            lines.append(f"    {stmt}")
        lines.append("}")
        return "\n".join(lines) + "\n"


class _ExportContext:
    """Mutable state threaded through op-converter invocations."""

    def __init__(self, graph: Any, name: str) -> None:
        self.graph = graph
        self.name = name
        self.name_map: dict[str, str] = {}
        self.tensors: dict[str, _Tensor] = {}
        self.statements: list[str] = []
        self.initializers: dict[str, np.ndarray] = {}
        self._tmp_counter = 0

    def map_name(self, onnx_name: str) -> str:
        return _to_identifier(onnx_name, self.name_map)

    def make_temp(self, base: str = "t") -> str:
        self._tmp_counter += 1
        cleaned = _IDENT_RE.sub("_", base) or "t"
        name = f"{cleaned}_tmp{self._tmp_counter}"
        return _to_identifier(name, self.name_map)

    def add_statement(self, stmt: str) -> None:
        self.statements.append(stmt)

    def get_initializer(self, onnx_name: str) -> np.ndarray | None:
        t = self.tensors.get(onnx_name)
        if t is None or not t.is_initializer:
            return None
        return t.initializer_array


# ---------------------------------------------------------------------------
# Operator converters.
# ---------------------------------------------------------------------------

#: Signature of a converter: ``f(ctx, node, attrs, inputs, outputs)``.
ConverterFn = Callable[["_ExportContext", Any, dict[str, Any], list[str], list[str]], None]

_CONVERTERS: dict[str, ConverterFn] = {}


def register_op_converter(op_type: str, converter: ConverterFn) -> None:
    """Registers (or overrides) the converter used for an ONNX op type.

    Args:
        op_type: ONNX operator name, e.g. ``"Conv"``.
        converter: Callable invoked with the export context, the ONNX
            node, the attribute dict and the mapped NNEF input/output
            identifiers.  The callable should append statements to
            ``ctx`` via :meth:`_ExportContext.add_statement`.
    """
    _CONVERTERS[op_type] = converter


def supported_ops() -> list[str]:
    """Returns the sorted list of ONNX op types with a builtin converter."""
    return sorted(_CONVERTERS.keys())


def _emit_assign(outputs: list[str], rhs: str) -> str:
    if len(outputs) == 1:
        return f"{outputs[0]} = {rhs};"
    lhs = ", ".join(outputs)
    return f"[{lhs}] = {rhs};"


def _emit_call(op: str, args: Sequence[str], kwargs: dict[str, Any] | None = None) -> str:
    parts = list(args)
    if kwargs:
        for k, v in kwargs.items():
            parts.append(f"{k} = {_format_value(v)}")
    return f"{op}({', '.join(parts)})"


# --- elementwise unary ----------------------------------------------------


def _make_unary(nnef_op: str) -> ConverterFn:
    def convert(ctx, node, attrs, inputs, outputs):
        ctx.add_statement(_emit_assign(outputs, _emit_call(nnef_op, [inputs[0]])))

    return convert


# --- elementwise binary ---------------------------------------------------


def _make_binary(nnef_op: str) -> ConverterFn:
    def convert(ctx, node, attrs, inputs, outputs):
        ctx.add_statement(_emit_assign(outputs, _emit_call(nnef_op, [inputs[0], inputs[1]])))

    return convert


# --- specific converters --------------------------------------------------


def _conv_padding(attrs: dict[str, Any], rank: int) -> list[list[int]]:
    pads = attrs.get("pads")
    if pads is None:
        auto_pad = attrs.get("auto_pad", "NOTSET")
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            return []  # NNEF understands an empty list as "auto/same"
        return [[0, 0]] * rank
    if len(pads) != 2 * rank:
        raise NNEFExportError(f"Expected {2 * rank} pad values, got {len(pads)}")
    return [[int(pads[i]), int(pads[i + rank])] for i in range(rank)]


def _conv_converter(ctx, node, attrs, inputs, outputs):
    if len(inputs) < 2:
        raise NNEFExportError("Conv requires at least input and weight")
    init_w = ctx.get_initializer(node.input[1])
    spatial_rank = 0
    if init_w is not None:
        spatial_rank = max(0, init_w.ndim - 2)
    elif "kernel_shape" in attrs:
        spatial_rank = len(attrs["kernel_shape"])
    bias = inputs[2] if len(inputs) >= 3 else "0.0"
    kwargs: dict[str, Any] = {}
    if "strides" in attrs:
        kwargs["stride"] = list(attrs["strides"])
    if "dilations" in attrs:
        kwargs["dilation"] = list(attrs["dilations"])
    if spatial_rank:
        kwargs["padding"] = _conv_padding(attrs, spatial_rank)
    if "group" in attrs and int(attrs["group"]) != 1:
        kwargs["groups"] = int(attrs["group"])
    args = [inputs[0], inputs[1], bias]
    ctx.add_statement(_emit_assign(outputs, _emit_call("conv", args, kwargs)))


def _pool_converter(nnef_op: str) -> ConverterFn:
    def convert(ctx, node, attrs, inputs, outputs):
        kernel = attrs.get("kernel_shape")
        if kernel is None:
            raise NNEFExportError(f"{node.op_type} requires kernel_shape")
        rank = len(kernel)
        # NNEF pool size includes batch and channel dims (=1 each).
        size = [1, 1, *list(kernel)]
        kwargs: dict[str, Any] = {"size": size}
        if "strides" in attrs:
            kwargs["stride"] = [1, 1, *list(attrs["strides"])]
        if "pads" in attrs or attrs.get("auto_pad", "NOTSET") != "NOTSET":
            kwargs["padding"] = [[0, 0], [0, 0], *_conv_padding(attrs, rank)]
        ctx.add_statement(_emit_assign(outputs, _emit_call(nnef_op, [inputs[0]], kwargs)))

    return convert


def _global_pool_converter(nnef_op: str) -> ConverterFn:
    def convert(ctx, node, attrs, inputs, outputs):
        # NNEF's mean_reduce / max_reduce on spatial axes implements
        # GlobalAveragePool / GlobalMaxPool for NCHW inputs.
        ctx.add_statement(
            _emit_assign(outputs, _emit_call(nnef_op, [inputs[0]], {"axes": [2, 3]}))
        )

    return convert


def _reshape_converter(ctx, node, attrs, inputs, outputs):
    init = ctx.get_initializer(node.input[1]) if len(node.input) >= 2 else None
    if init is None:
        raise NNEFExportError("Reshape requires a constant shape initializer to export to NNEF")
    shape = [int(v) for v in np.asarray(init).reshape(-1).tolist()]
    ctx.add_statement(_emit_assign(outputs, _emit_call("reshape", [inputs[0]], {"shape": shape})))


def _transpose_converter(ctx, node, attrs, inputs, outputs):
    perm = attrs.get("perm")
    kwargs: dict[str, Any] = {}
    if perm is not None:
        kwargs["axes"] = list(perm)
    ctx.add_statement(_emit_assign(outputs, _emit_call("transpose", [inputs[0]], kwargs)))


def _concat_converter(ctx, node, attrs, inputs, outputs):
    axis = int(attrs.get("axis", 0))
    rhs = f"concat([{', '.join(inputs)}], axis = {axis})"
    ctx.add_statement(_emit_assign(outputs, rhs))


def _softmax_converter(ctx, node, attrs, inputs, outputs):
    axis = int(attrs.get("axis", -1))
    ctx.add_statement(_emit_assign(outputs, _emit_call("softmax", [inputs[0]], {"axes": [axis]})))


def _flatten_converter(ctx, node, attrs, inputs, outputs):
    axis = int(attrs.get("axis", 1))
    if axis == 1:
        shape = [0, -1]
    else:
        # NNEF reshape supports 0=keep-dim and -1=infer; flatten with
        # arbitrary axis collapses the leading and trailing groups.
        shape = [0] * axis + [-1]
    ctx.add_statement(_emit_assign(outputs, _emit_call("reshape", [inputs[0]], {"shape": shape})))


def _gemm_converter(ctx, node, attrs, inputs, outputs):
    if len(inputs) < 2:
        raise NNEFExportError("Gemm requires at least A and B")
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    if alpha != 1.0 or beta != 1.0:
        raise NNEFExportError("Gemm export only supports alpha=beta=1")
    a = inputs[0]
    b = inputs[1]
    if trans_a:
        ta = ctx.make_temp("gemm_a")
        ctx.add_statement(_emit_assign([ta], _emit_call("transpose", [a])))
        a = ta
    if trans_b:
        tb = ctx.make_temp("gemm_b")
        ctx.add_statement(_emit_assign([tb], _emit_call("transpose", [b])))
        b = tb
    matmul_out = outputs[0] if len(inputs) < 3 else ctx.make_temp("gemm_mm")
    ctx.add_statement(_emit_assign([matmul_out], _emit_call("matmul", [a, b])))
    if len(inputs) >= 3:
        ctx.add_statement(_emit_assign(outputs, _emit_call("add", [matmul_out, inputs[2]])))


def _matmul_converter(ctx, node, attrs, inputs, outputs):
    ctx.add_statement(_emit_assign(outputs, _emit_call("matmul", [inputs[0], inputs[1]])))


def _batchnorm_converter(ctx, node, attrs, inputs, outputs):
    if len(inputs) < 5:
        raise NNEFExportError("BatchNormalization requires 5 inputs")
    eps = float(attrs.get("epsilon", 1e-5))
    # NNEF: batch_normalization(input, mean, variance, offset, scale, epsilon)
    args = [inputs[0], inputs[3], inputs[4], inputs[2], inputs[1]]
    ctx.add_statement(
        _emit_assign(outputs[:1], _emit_call("batch_normalization", args, {"epsilon": eps}))
    )


def _identity_converter(ctx, node, attrs, inputs, outputs):
    ctx.add_statement(_emit_assign(outputs, _emit_call("copy", [inputs[0]])))


def _clip_converter(ctx, node, attrs, inputs, outputs):
    # ONNX Clip has min/max as attributes (opset <11) or inputs (opset ≥11).
    lo = attrs.get("min")
    hi = attrs.get("max")
    extra: list[float] = []
    if lo is None and len(inputs) >= 2:
        init = ctx.get_initializer(node.input[1])
        if init is None:
            raise NNEFExportError("Clip min must be a constant initializer for NNEF export")
        lo = float(init.reshape(-1)[0])
    if hi is None and len(inputs) >= 3:
        init = ctx.get_initializer(node.input[2])
        if init is None:
            raise NNEFExportError("Clip max must be a constant initializer for NNEF export")
        hi = float(init.reshape(-1)[0])
    if lo is None:
        lo = float("-inf")
    if hi is None:
        hi = float("inf")
    if not np.isfinite(lo) and not np.isfinite(hi):
        # Pure identity – nothing to clip.
        ctx.add_statement(_emit_assign(outputs, _emit_call("copy", [inputs[0]])))
        return
    extra = [float(lo), float(hi)]
    ctx.add_statement(
        _emit_assign(
            outputs,
            _emit_call("clamp", [inputs[0], _format_scalar(extra[0]), _format_scalar(extra[1])]),
        )
    )


# Register the builtin converters.
for _onnx, _nnef in [
    ("Relu", "relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("Softplus", "softplus"),
    ("Exp", "exp"),
    ("Log", "log"),
    ("Sqrt", "sqrt"),
    ("Neg", "neg"),
    ("Abs", "abs"),
    ("Floor", "floor"),
    ("Ceil", "ceil"),
    ("Sin", "sin"),
    ("Cos", "cos"),
    ("Not", "not"),
]:
    register_op_converter(_onnx, _make_unary(_nnef))

for _onnx, _nnef in [
    ("Add", "add"),
    ("Sub", "sub"),
    ("Mul", "mul"),
    ("Div", "div"),
    ("Pow", "pow"),
    ("Min", "min"),
    ("Max", "max"),
    ("And", "and"),
    ("Or", "or"),
    ("Equal", "eq"),
    ("Less", "lt"),
    ("Greater", "gt"),
]:
    register_op_converter(_onnx, _make_binary(_nnef))

register_op_converter("Conv", _conv_converter)
register_op_converter("MaxPool", _pool_converter("max_pool"))
register_op_converter("AveragePool", _pool_converter("avg_pool"))
register_op_converter("GlobalAveragePool", _global_pool_converter("mean_reduce"))
register_op_converter("GlobalMaxPool", _global_pool_converter("max_reduce"))
register_op_converter("Reshape", _reshape_converter)
register_op_converter("Transpose", _transpose_converter)
register_op_converter("Concat", _concat_converter)
register_op_converter("Softmax", _softmax_converter)
register_op_converter("Flatten", _flatten_converter)
register_op_converter("Gemm", _gemm_converter)
register_op_converter("MatMul", _matmul_converter)
register_op_converter("BatchNormalization", _batchnorm_converter)
register_op_converter("Identity", _identity_converter)
register_op_converter("Clip", _clip_converter)


# ---------------------------------------------------------------------------
# Public exporter API.
# ---------------------------------------------------------------------------


def _shape_from_value_info(vi: Any) -> tuple[tuple[int, ...] | None, np.dtype | None]:
    tt = getattr(getattr(vi, "type", None), "tensor_type", None)
    if tt is None:
        return None, None
    dtype = _ONNX_DTYPE_TO_NUMPY.get(int(tt.elem_type)) if tt.elem_type else None
    dims: list[int] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") if hasattr(d, "HasField") else getattr(d, "dim_value", 0):
            dims.append(int(d.dim_value))
        else:
            dims.append(-1)
    return tuple(dims) if dims else None, dtype


def export_to_nnef(model: Any, *, graph_name: str | None = None) -> NNEFGraph:
    """Builds an in-memory :class:`NNEFGraph` from an ONNX ``ModelProto``.

    The result holds both the textual graph (accessible through
    :meth:`NNEFGraph.to_text`) and the parameter arrays that should be
    written to ``*.dat`` files.

    Args:
        model: An ONNX :class:`ModelProto` (from :mod:`onnx` or
            :mod:`onnx_light`).
        graph_name: Optional name to use for the NNEF ``graph`` block;
            defaults to ``model.graph.name`` or ``"main"``.

    Returns:
        A :class:`NNEFGraph` describing the converted model.

    Raises:
        NNEFExportError: when the model uses an operator without a
            registered converter, or when an attribute / tensor cannot
            be encoded in NNEF.
    """
    graph = model.graph
    name = graph_name or graph.name or "main"
    name = _IDENT_RE.sub("_", name) or "main"
    ctx = _ExportContext(graph, name)

    # Pre-populate name map with deterministic ids for I/O.
    input_names: list[str] = []
    output_names: list[str] = []
    initializer_names: set[str] = {init.name for init in graph.initializer}

    for vi in graph.input:
        if vi.name in initializer_names:
            # Initializer also listed as graph input → treat as constant.
            continue
        shape, dtype = _shape_from_value_info(vi)
        nnef_name = ctx.map_name(vi.name)
        ctx.tensors[vi.name] = _Tensor(vi.name, nnef_name, shape, dtype)
        input_names.append(nnef_name)

    for init in graph.initializer:
        array = _tensor_to_numpy(init)
        nnef_name = ctx.map_name(init.name)
        ctx.tensors[init.name] = _Tensor(
            init.name,
            nnef_name,
            tuple(array.shape),
            array.dtype,
            is_initializer=True,
            initializer_array=array,
        )
        ctx.initializers[nnef_name] = array

    for vi in graph.output:
        shape, dtype = _shape_from_value_info(vi)
        nnef_name = ctx.map_name(vi.name)
        if vi.name not in ctx.tensors:
            ctx.tensors[vi.name] = _Tensor(vi.name, nnef_name, shape, dtype)
        output_names.append(nnef_name)

    # Emit `external` statements for the graph inputs.
    for vi in graph.input:
        if vi.name in initializer_names:
            continue
        t = ctx.tensors[vi.name]
        kwargs: dict[str, Any] = {}
        if t.shape is not None:
            shape = [int(d) if d >= 0 else 1 for d in t.shape]
            kwargs["shape"] = shape
        ctx.add_statement(_emit_assign([t.nnef_name], _emit_call("external", [], kwargs)))

    # Emit `variable` statements for the initializers.
    for init in graph.initializer:
        t = ctx.tensors[init.name]
        kwargs = {"shape": list(t.initializer_array.shape), "label": init.name}
        ctx.add_statement(_emit_assign([t.nnef_name], _emit_call("variable", [], kwargs)))

    # Walk the nodes in topological order (ONNX graphs are stored sorted).
    for node in graph.node:
        op = node.op_type
        if op not in _CONVERTERS:
            raise NNEFExportError(
                f"No NNEF converter registered for ONNX op '{op}' (node '{node.name}'). "
                "Use onnx_light.nnef.register_op_converter to add one."
            )
        attrs = _attrs_to_dict(node)
        inputs = [ctx.map_name(n) if n else "0.0" for n in node.input]
        # Allocate fresh identifiers for outputs that are not already mapped.
        outputs = []
        for n in node.output:
            if not n:
                outputs.append(ctx.make_temp("unused"))
                continue
            nnef_name = ctx.map_name(n)
            if n not in ctx.tensors:
                ctx.tensors[n] = _Tensor(n, nnef_name)
            outputs.append(nnef_name)
        _CONVERTERS[op](ctx, node, attrs, inputs, outputs)

    nnef_graph = NNEFGraph(
        name=name,
        inputs=input_names,
        outputs=output_names,
        statements=ctx.statements,
        initializers=ctx.initializers,
    )
    return nnef_graph


def to_nnef_text(model: Any, *, graph_name: str | None = None) -> str:
    """Returns the ``graph.nnef`` text for ``model`` (initializers ignored)."""
    return export_to_nnef(model, graph_name=graph_name).to_text()


def save_nnef(
    model: Any,
    out_dir: str | os.PathLike[str],
    *,
    graph_name: str | None = None,
    overwrite: bool = True,
) -> str:
    """Writes ``model`` to ``out_dir`` as a NNEF model archive on disk.

    The destination directory is created if missing and will contain a
    ``graph.nnef`` file together with one ``<label>.dat`` file per
    initializer (where ``<label>`` is the safe-identifier mapping of
    the ONNX initializer name).

    Args:
        model: ONNX :class:`ModelProto` to export.
        out_dir: Destination directory.
        graph_name: Optional override for the NNEF graph name.
        overwrite: When ``False`` an existing non-empty directory raises
            :class:`FileExistsError`.

    Returns:
        The absolute path of ``out_dir``.

    Raises:
        NNEFExportError: when the model cannot be expressed in NNEF.
        FileExistsError: when ``overwrite`` is ``False`` and the output
            directory already contains files.
    """
    nnef = export_to_nnef(model, graph_name=graph_name)
    out_dir = os.fspath(out_dir)
    if os.path.isdir(out_dir):
        if not overwrite and os.listdir(out_dir):
            raise FileExistsError(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "graph.nnef"), "w", encoding="utf-8") as f:
        f.write(nnef.to_text())
    for nnef_name, array in nnef.initializers.items():
        write_nnef_tensor(os.path.join(out_dir, f"{nnef_name}.dat"), array)
    return os.path.abspath(out_dir)
