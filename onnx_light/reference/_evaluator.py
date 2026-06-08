# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of :class:`ReferenceEvaluator`.

The evaluator wraps the ``RunModel`` / ``RunGraph`` / ``RunFunction``
dispatcher exposed by :mod:`onnx_light.onnx_py._onnxkernels` (re-exported
through the ``runtime`` submodule). All operator implementations come
from the C++ ``KernelDispatchTable``; this Python class only handles
input/output conversion between :class:`numpy.ndarray` and the runtime
``Tensor`` type and the bookkeeping required to expose an
``onnx.reference.ReferenceEvaluator``-compatible API.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..onnx_lib import (
    FunctionProto,
    GraphProto,
    ModelProto,
    TensorProto,
    load,
    numpy_helper,
)
from ..onnx_py._onnxkernels import runtime as _runtime  # type: ignore[missing-import]

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

    Delegates to :func:`numpy_helper.to_array`, which already handles
    sub-byte packed types (INT4/UINT4/INT2/UINT2/FLOAT4E2M1) and string
    tensors.
    """
    return numpy_helper.to_array(_cpp_tensor_to_proto(t))


def _numpy_to_cpp_tensor(name: str, arr: np.ndarray) -> Any:
    """Converts a :class:`numpy.ndarray` to a runtime ``Tensor``."""
    if isinstance(arr, np.ndarray):
        np_arr = arr
    else:
        np_arr = np.asarray(arr)
    tp = numpy_helper.from_array(np_arr, name=name)
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
    ) -> None:
        proto = self._load_proto(proto)
        self._model: ModelProto | None = None
        self._graph: GraphProto | None = None
        self._function: FunctionProto | None = None

        if isinstance(proto, ModelProto):
            self._model = proto
            self._graph = proto.graph
            inputs = [vi.name for vi in self._graph.input]
            outputs = [vi.name for vi in self._graph.output]
            initializers = {init.name for init in self._graph.initializer}
            self._opsets = {op.domain: op.version for op in proto.opset_import}
        elif isinstance(proto, GraphProto):
            self._graph = proto
            inputs = [vi.name for vi in proto.input]
            outputs = [vi.name for vi in proto.output]
            initializers = {init.name for init in proto.initializer}
            self._opsets = {}
        elif isinstance(proto, FunctionProto):
            self._function = proto
            inputs = list(proto.input)
            outputs = list(proto.output)
            initializers = set()
            self._opsets = {op.domain: op.version for op in proto.opset_import}
        else:
            raise TypeError(
                f"Unsupported proto type for ReferenceEvaluator: {type(proto).__name__}."
            )

        # Inputs that are also initializers (a legal but uncommon pattern)
        # do not need to be supplied by the caller.
        self._input_names: list[str] = [n for n in inputs if n not in initializers]
        self._output_names: list[str] = list(outputs)

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
        raise TypeError(
            f"Unsupported input type for ReferenceEvaluator: {type(proto).__name__}."
        )

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

    def run(
        self,
        output_names: list[str] | None,
        feed_inputs: dict[str, Any],
    ) -> list[np.ndarray]:
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
        list of :class:`numpy.ndarray`
            One array per name in ``output_names`` (defaults to
            :attr:`output_names`), in the requested order.
        """
        if output_names is None:
            output_names = list(self._output_names)

        if not isinstance(feed_inputs, dict):
            raise TypeError(
                "feed_inputs must be a dict[name -> numpy.ndarray], not "
                f"{type(feed_inputs).__name__}."
            )

        missing = [n for n in self._input_names if n not in feed_inputs]
        if missing:
            raise ValueError(
                f"Missing input(s) for ReferenceEvaluator.run: {missing}. "
                f"Expected: {self._input_names}, got: {list(feed_inputs)}."
            )

        # Pick the opset version of the default ai.onnx domain (falls back
        # to the highest version declared otherwise, then to 0).
        version: int = int(self._opsets.get("", self._opsets.get("ai.onnx", 0)) or 0)
        if version == 0 and self._opsets:
            version = int(max(self._opsets.values()))
        ctx = _runtime.RuntimeContext(
            _runtime.KernelContext(_runtime.default_opset(version))
        )

        for name, value in feed_inputs.items():
            ctx.set(name, _numpy_to_cpp_tensor(name, value))

        if self._model is not None:
            _runtime.run_model(self._model, ctx)
        elif self._function is not None:
            _runtime.run_function(self._function, ctx)
        else:
            assert self._graph is not None
            _runtime.run_graph(self._graph, ctx)

        results: list[np.ndarray] = []
        for name in output_names:
            if not ctx.has(name):
                raise RuntimeError(
                    f"Output {name!r} was not produced by the graph. "
                    f"Available names after execution: {sorted(ctx.names())}."
                )
            results.append(_cpp_tensor_to_numpy(ctx.get(name)))
        return results
