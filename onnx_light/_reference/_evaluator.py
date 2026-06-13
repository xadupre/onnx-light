# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of :class:`ReferenceEvaluator`.

The evaluator wraps the ``RunModel`` / ``RunGraph`` / ``RunFunction``
dispatcher exposed by :mod:`onnx_light.onnx_py._onnxpykernels` (re-exported
through the ``runtime`` submodule). All operator implementations come
from the C++ ``KernelDispatchTable``; this Python class only handles
input/output conversion between :class:`numpy.ndarray` and the runtime
``Tensor`` type and the bookkeeping required to expose an
``onnx.reference.ReferenceEvaluator``-compatible API.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from ..onnx_lib import FunctionProto, GraphProto, ModelProto, TensorProto, load, numpy_helper
from ..onnx_py._onnxpykernels import runtime as _runtime  # type: ignore[missing-import]

try:
    import ml_dtypes as _ml_dtypes  # type: ignore
except ImportError:  # pragma: no cover - ml_dtypes is a runtime dependency
    _ml_dtypes = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tensor / numpy conversion helpers
# ---------------------------------------------------------------------------

# Mapping ``TensorProto.DataType`` -> numpy dtype for fixed-width element
# types. Sub-byte types (INT4, UINT4, INT2, UINT2, FLOAT4E2M1) and the
# string type are handled out-of-band: see ``_cpp_tensor_to_proto``.
_DTYPE_TO_NP: dict[int, Any] = {
    int(TensorProto.FLOAT): np.float32,
    int(TensorProto.DOUBLE): np.float64,
    int(TensorProto.INT8): np.int8,
    int(TensorProto.INT16): np.int16,
    int(TensorProto.INT32): np.int32,
    int(TensorProto.INT64): np.int64,
    int(TensorProto.UINT8): np.uint8,
    int(TensorProto.UINT16): np.uint16,
    int(TensorProto.UINT32): np.uint32,
    int(TensorProto.UINT64): np.uint64,
    int(TensorProto.BOOL): np.bool_,
    int(TensorProto.FLOAT16): np.float16,
}

if _ml_dtypes is not None:
    _DTYPE_TO_NP.update(
        {
            int(TensorProto.BFLOAT16): _ml_dtypes.bfloat16,
            int(TensorProto.FLOAT8E4M3FN): _ml_dtypes.float8_e4m3fn,
            int(TensorProto.FLOAT8E4M3FNUZ): _ml_dtypes.float8_e4m3fnuz,
            int(TensorProto.FLOAT8E5M2): _ml_dtypes.float8_e5m2,
            int(TensorProto.FLOAT8E5M2FNUZ): _ml_dtypes.float8_e5m2fnuz,
        }
    )

# Reverse mapping: numpy dtype -> TensorProto.DataType int (for fast input path).
_NP_TO_DTYPE: dict[Any, int] = {v: k for k, v in _DTYPE_TO_NP.items()}

_IS_BIG_ENDIAN = sys.byteorder == "big"


def _cpp_tensor_to_proto(t: Any) -> TensorProto:
    """Materializes a :class:`TensorProto` from a runtime ``Tensor``.

    The returned proto carries an owned copy of the tensor bytes (or string
    data) and is suitable input for :func:`numpy_helper.to_array`.
    """
    tp = TensorProto()
    if t.name:
        tp.name = t.name
    tp.data_type = int(t.data_type)
    for d in t.shape:
        tp.dims.append(int(d))
    if int(t.data_type) == int(TensorProto.STRING):
        for s in t.string_data():
            tp.string_data.append(s.encode("utf-8") if isinstance(s, str) else s)
    else:
        tp.raw_data = bytes(t.raw_data())
    return tp


def _cpp_tensor_to_numpy(t: Any) -> np.ndarray:
    """Converts a runtime ``Tensor`` to a :class:`numpy.ndarray`.

    For standard fixed-width dtypes (float32, int64, etc.) the conversion
    is done directly via :func:`numpy.frombuffer` on the raw bytes,
    bypassing the intermediate :class:`TensorProto` construction.
    Sub-byte packed types and STRING tensors fall back to the full
    :func:`numpy_helper.to_array` path.
    """
    dt = int(t.data_type)
    np_dtype = _DTYPE_TO_NP.get(dt)
    if np_dtype is not None:
        raw = bytes(t.raw_data())
        if _IS_BIG_ENDIAN:  # pragma: no cover
            raw = np.frombuffer(raw, dtype=np_dtype).byteswap().tobytes()
        shape = t.shape
        return np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    # Fallback for sub-byte types (INT4/UINT4/INT2/UINT2/FLOAT4E2M1) and STRING.
    return numpy_helper.to_array(_cpp_tensor_to_proto(t))


def _numpy_to_cpp_tensor(name: str, arr: np.ndarray) -> Any:
    """Converts a :class:`numpy.ndarray` to a runtime ``Tensor``.

    For standard fixed-width dtypes, builds a minimal :class:`TensorProto`
    with only ``raw_data``, ``data_type`` and ``dims`` populated, avoiding
    the overhead of the full :func:`numpy_helper.from_array` path.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    onnx_dtype = _NP_TO_DTYPE.get(arr.dtype.type)
    if onnx_dtype is not None:
        tp = TensorProto()
        tp.name = name
        tp.data_type = onnx_dtype
        tp.dims.extend(arr.shape)
        tp.raw_data = arr.tobytes()
        return _runtime.tensor_from_proto(tp)
    # Fallback for strings, sub-byte types, and exotic dtypes.
    tp = numpy_helper.from_array(arr, name=name)
    return _runtime.tensor_from_proto(tp)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ReferenceEvaluator:
    """Evaluates an ONNX model using the C++ ``KernelDispatchTable``.

    The class is constructed from a ``ModelProto`` / ``GraphProto`` /
    ``FunctionProto`` (or the bytes / file path of a serialised
    ``ModelProto``). :meth:`run` then takes a ``{name: ndarray}``
    feed dictionary and returns the requested outputs as a list of
    NumPy arrays, mirroring the calling convention of
    :class:`onnx.reference.ReferenceEvaluator` (and
    :class:`onnxruntime.InferenceSession`).

    Example
    -------
    .. code-block:: python

        import numpy as np
        from onnx_light.onnx_lib import parser
        from onnx_light.reference import ReferenceEvaluator

        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["" : 18]>'
            'agraph (float[3] x) => (float[3] y) { y = Abs(x) }'
        )
        sess = ReferenceEvaluator(model)
        (y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.5], dtype=np.float32)})
        # y == np.array([1.0, 2.0, 3.5], dtype=np.float32)
    """

    def __init__(
        self,
        proto: ModelProto | GraphProto | FunctionProto | bytes | str | os.PathLike,
        *,
        events_enabled: bool = False,
        release_intermediates: bool = True,
    ) -> None:
        proto = self._load_proto(proto)
        self._model: ModelProto | None = None
        self._graph: GraphProto | None = None
        self._function: FunctionProto | None = None
        self._last_ctx: Any = None
        self._events_enabled = events_enabled
        self._release_intermediates = release_intermediates

        if isinstance(proto, ModelProto):
            self._model = proto
            self._graph = proto.graph
            graph_inputs = list(self._graph.input)
            outputs = [vi.name for vi in self._graph.output]
            initializers = {init.name for init in self._graph.initializer}
            self._opsets = {op.domain: op.version for op in proto.opset_import}
        elif isinstance(proto, GraphProto):
            self._graph = proto
            graph_inputs = list(proto.input)
            outputs = [vi.name for vi in proto.output]
            initializers = {init.name for init in proto.initializer}
            self._opsets = {}
        elif isinstance(proto, FunctionProto):
            self._function = proto
            graph_inputs = None
            outputs = list(proto.output)
            initializers = set()
            self._opsets = {op.domain: op.version for op in proto.opset_import}
        else:
            raise TypeError(
                f"Unsupported proto type for ReferenceEvaluator: {type(proto).__name__}."
            )

        # Map-typed graph inputs (``map(K, V)``) are consumed by the C++ kernels
        # for ``ai.onnx.ml::DictVectorizer`` and ``ai.onnx.ml::CastMap`` through a
        # two-tensor naming convention: ``<name>_keys`` / ``<name>_values``.
        # Expand these inputs so callers feed (and the runtime receives) the tensors
        # directly by those names, keeping all map-handling logic in C++.
        if graph_inputs is not None:
            inputs: list[str] = []
            for vi in graph_inputs:
                if vi.name in initializers:
                    continue
                if vi.type.has_map_type():
                    inputs.append(f"{vi.name}_keys")
                    inputs.append(f"{vi.name}_values")
                else:
                    inputs.append(vi.name)
        else:
            assert self._function is not None
            inputs = list(self._function.input)

        self._input_names: list[str] = inputs
        self._input_names_set: frozenset[str] = frozenset(self._input_names)
        self._output_names: list[str] = list(outputs)

        # Pre-compute the opset version and KernelContext once at construction
        # time instead of rebuilding them on every run() call.
        version: int = int(self._opsets.get("", self._opsets.get("ai.onnx", 0)) or 0)
        if version == 0 and self._opsets:
            version = int(max(self._opsets.values()))
        self._kernel_ctx = _runtime.KernelContext(_runtime.default_opset(version))

        # Mapping ``"<domain>:<op_type>" -> low-level callback``. A
        # low-level callback has the signature
        # ``fn(node: NodeProto, ctx: RuntimeContext) -> None`` and is
        # registered as-is on every freshly constructed RuntimeContext
        # in :meth:`run`. The higher-level :meth:`register_custom_kernel`
        # API installs a numpy-friendly wrapper that delegates to the
        # user-provided callable.
        self._custom_kernels: dict[str, Any] = {}

    # -- custom kernels -----------------------------------------------------

    def register_custom_kernel(self, domain: str, op_type: str, fn: Any) -> None:
        """Registers a Python custom kernel for ``(domain, op_type)``.

        The kernel is invoked on every :meth:`run` call whenever a node
        matches the registered ``(domain, op_type)`` pair. Custom
        kernels override any built-in onnx-light kernel with the same
        key (model-local functions and the built-in control-flow
        operators ``If`` / ``Loop`` / ``Scan`` / ``SequenceMap`` still
        take precedence).

        Parameters
        ----------
        domain:
            Operator domain. The empty string is treated as
            ``ai.onnx``.
        op_type:
            Operator name (``NodeProto.op_type``).
        fn:
            Python callable invoked as ``fn(node, *inputs)`` where
            ``node`` is the matching :class:`NodeProto` and ``inputs``
            are the input tensors converted to :class:`numpy.ndarray`.
            The callable must return either a single
            :class:`numpy.ndarray` (for single-output kernels) or a
            tuple / list of arrays (for multi-output kernels), in the
            same order as the node's declared outputs.

        Examples
        --------
        .. code-block:: python

            def square(node, x):
                return x * x

            sess.register_custom_kernel("my.domain", "Square", square)
        """

        def _wrapper(node: Any, ctx: Any) -> None:
            inputs: list[Any] = []
            for raw_name in node.input:
                name = str(raw_name)
                if not name:
                    inputs.append(None)
                else:
                    inputs.append(_cpp_tensor_to_numpy(ctx.get(name)))
            result = fn(node, *inputs)
            if isinstance(result, (list, tuple)):
                outputs = list(result)
            else:
                outputs = [result]
            output_names = [str(n) for n in node.output]
            expected = len(output_names)
            if len(outputs) != expected:
                raise ValueError(
                    f"Custom kernel for {domain!r}:{op_type!r} returned "
                    f"{len(outputs)} output(s) but the node declares {expected} "
                    f"output(s)."
                )
            for i, value in enumerate(outputs):
                name = output_names[i]
                if not name:
                    continue
                ctx.put(name, _numpy_to_cpp_tensor(name, value), "output")

        self._custom_kernels[f"{domain or 'ai.onnx'}:{op_type}"] = (domain, op_type, _wrapper)

    # -- proto loading ------------------------------------------------------

    @staticmethod
    def _load_proto(
        proto: ModelProto | GraphProto | FunctionProto | bytes | str | os.PathLike,
    ) -> ModelProto | GraphProto | FunctionProto:
        if isinstance(proto, (ModelProto, GraphProto, FunctionProto)):
            return proto
        if isinstance(proto, (bytes, bytearray)):
            model = ModelProto()
            model.ParseFromString(bytes(proto))
            return model
        if isinstance(proto, (str, os.PathLike)):
            return load(os.fspath(proto))
        raise TypeError(f"Unsupported input type for ReferenceEvaluator: {type(proto).__name__}.")

    # -- public properties --------------------------------------------------

    @property
    def input_names(self) -> list[str]:
        """Names of the graph (or function) inputs, in declaration order.

        Inputs that are also listed in the graph initializers are
        omitted, since the caller does not have to supply them.
        """
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        """Names of the graph (or function) outputs, in declaration order."""
        return list(self._output_names)

    @property
    def opsets(self) -> dict[str, int]:
        """Mapping ``domain -> version`` extracted from ``opset_import``.

        Empty for evaluators built from a bare :class:`GraphProto`.
        """
        return dict(self._opsets)

    # -- evaluation ---------------------------------------------------------

    def events(self) -> list[Any]:
        """Returns the event log from the most recent :meth:`run` call.

        Each entry is a ``TensorEvent`` object with an :meth:`as_dict` method
        that returns a dictionary with the keys ``"action"``, ``"kind"``,
        ``"name"``, ``"data_type"``, ``"shape"``, ``"value_count"``,
        ``"values"`` and ``"string_values"``.

        Returns an empty list if :meth:`run` has not been called yet.
        """
        if self._last_ctx is None:
            return []
        return self._last_ctx.events()

    def run(
        self, output_names: list[str] | None, feed_inputs: dict[str, Any]
    ) -> list[np.ndarray | list]:
        """Executes the wrapped graph / model / function.

        Parameters
        ----------
        output_names:
            Names of the outputs to return. ``None`` is shorthand for
            "every declared output, in declaration order".
        feed_inputs:
            Mapping of input name to NumPy array. Every name listed by
            :attr:`input_names` must be present.

        Returns
        -------
        list of :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            One entry per name in ``output_names`` (defaults to
            :attr:`output_names`), in the requested order. Tensor-typed
            outputs are returned as :class:`numpy.ndarray`; sequence-typed
            outputs are returned as a ``list`` of :class:`numpy.ndarray`
            (one array per sequence element).
        """
        if output_names is None:
            output_names = self._output_names

        if not isinstance(feed_inputs, dict):
            raise TypeError(
                "feed_inputs must be a dict[name -> numpy.ndarray], not "
                f"{type(feed_inputs).__name__}."
            )

        # Fast missing-input check using frozenset difference.
        missing = self._input_names_set - feed_inputs.keys()
        if missing:
            raise ValueError(
                f"Missing input(s) for ReferenceEvaluator.run: {sorted(missing)}. "
                f"Expected: {self._input_names}, got: {list(feed_inputs)}."
            )

        ctx = _runtime.RuntimeContext(self._kernel_ctx)
        ctx.events_enabled = self._events_enabled
        # Releasing intermediates would drop any requested output that is
        # not a declared graph/function output before the caller can fetch
        # it. Disable the per-run release in that case so callers can still
        # observe arbitrary intermediate values via ``run([name], ...)``.
        declared_outputs = frozenset(self._output_names)
        release = self._release_intermediates and all(
            name in declared_outputs for name in output_names
        )
        ctx.release_intermediates = release

        for domain, op_type, wrapper in self._custom_kernels.values():
            ctx.register_custom_kernel(domain, op_type, wrapper)

        for name, value in feed_inputs.items():
            ctx.set(name, _numpy_to_cpp_tensor(name, value))

        if self._model is not None:
            _runtime.run_model(self._model, ctx)
        elif self._function is not None:
            _runtime.run_function(self._function, ctx)
        else:
            _runtime.run_graph(self._graph, ctx)

        self._last_ctx = ctx

        results: list[np.ndarray | list] = []
        for name in output_names:
            if ctx.has(name):
                results.append(_cpp_tensor_to_numpy(ctx.get(name)))
            elif ctx.has_sequence(name):
                results.append([_cpp_tensor_to_numpy(t) for t in ctx.get_sequence(name)])
            else:
                all_names = sorted(ctx.names() + ctx.sequence_names())
                raise RuntimeError(
                    f"Output {name!r} was not produced by the graph. "
                    f"Available names after execution: {all_names}."
                )
        return results
