# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of :class:`ReferenceEvaluator`.

The evaluator wraps the ``RuntimeSession`` / ``ExecutionPlan`` execution
machinery exposed by :mod:`onnx_light.onnx_py._onnxpykernels` (re-exported
through the ``runtime`` submodule). All operator implementations come from
the C++ ``KernelDispatchTable``; this Python class only handles input/output
conversion between :class:`numpy.ndarray` and the runtime ``Tensor`` type and
the bookkeeping required to expose an
``onnx.reference.ReferenceEvaluator``-compatible API.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from ..onnx_lib import FunctionProto, GraphProto, ModelProto, TensorProto, load, numpy_helper

try:
    from ..onnx_py._onnxpykernels import runtime as _runtime  # type: ignore[missing-import]
except ImportError as exc:  # pragma: no cover - exercised only in reduced builds
    raise ImportError(
        "onnx-light was built without the operator-kernel runtime "
        "(ONNX_LIGHT_BUILD_KERNELS=OFF); install the full build to use "
        "ReferenceEvaluator."
    ) from exc

try:
    import ml_dtypes as _ml_dtypes  # type: ignore
except ImportError:  # pragma: no cover - ml_dtypes is a runtime dependency
    _ml_dtypes = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tensor / numpy conversion helpers
# ---------------------------------------------------------------------------

# Mapping ``TensorProto.DataType`` -> numpy dtype for fixed-width element
# types. Sub-byte types (INT4, UINT4, INT2, UINT2, FLOAT4E2M1) and the
# string type are handled out-of-band: see ``_runtime.tensor_to_proto``.
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

# ``TensorProto.DataType`` values backed by a stock NumPy dtype, which
# therefore round-trip through the DLPack exchange protocol
# (``Tensor.__dlpack__`` / :func:`numpy.from_dlpack`). Captured before the
# ml_dtypes entries are added below: bfloat16/float8 have no stock NumPy dtype
# and, like the sub-byte packed and string types, use the raw-reinterpret or
# proto fallbacks instead.
_DLPACK_DTYPES: frozenset[int] = frozenset(_DTYPE_TO_NP)

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


def _cpp_tensor_to_numpy(t: Any) -> np.ndarray:
    """Converts a runtime ``Tensor`` to a :class:`numpy.ndarray`.

    For standard fixed-width dtypes (float32, int64, etc.) the array is a
    zero-copy view imported through the DLPack exchange protocol
    (``Tensor.__dlpack__`` / :func:`numpy.from_dlpack`); the DLPack capsule
    keeps the source tensor alive for the lifetime of the returned array so the
    borrowed view never dangles. bfloat16/float8 tensors are reinterpreted from
    the raw byte view returned by :func:`_runtime.tensor_to_numpy` (DLPack has
    no stock NumPy dtype for them), and sub-byte packed types and STRING
    tensors fall back to the full :func:`numpy_helper.to_array` path.
    """
    dt = int(t.data_type)
    if not _IS_BIG_ENDIAN and dt in _DLPACK_DTYPES:
        # Zero-copy import via the DLPack protocol; ``from_dlpack`` yields a
        # correctly shaped, native-dtype array sharing the tensor's buffer.
        return np.from_dlpack(t)
    np_dtype = _DTYPE_TO_NP.get(dt)
    if np_dtype is not None:
        # ``tensor_to_numpy`` returns a 1-D uint8 view borrowing the tensor's
        # bytes (no copy); reinterpret it as ``np_dtype`` and reshape. Also
        # used on big-endian hosts, where the little-endian buffer must be
        # byte-swapped after reinterpretation.
        raw = _runtime.tensor_to_numpy(t)
        arr = raw.view(np_dtype)
        if _IS_BIG_ENDIAN:  # pragma: no cover
            arr = arr.byteswap()
        return arr.reshape(t.shape)
    # Fallback for sub-byte types (INT4/UINT4/INT2/UINT2/FLOAT4E2M1) and STRING.
    return numpy_helper.to_array(_runtime.tensor_to_proto(t))


def _numpy_to_cpp_tensor(name: str, arr: np.ndarray) -> Any:
    """Converts a :class:`numpy.ndarray` to a runtime ``Tensor``.

    For standard fixed-width dtypes, copies the array's raw bytes directly
    into the C++ ``Tensor`` via :func:`_runtime.tensor_from_numpy`, avoiding
    the triple-copy overhead of serializing through a ``TensorProto``
    intermediate.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    onnx_dtype = _NP_TO_DTYPE.get(arr.dtype.type)
    if onnx_dtype is not None:
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        raw = arr.view(np.uint8).ravel()
        return _runtime.tensor_from_numpy(name, onnx_dtype, list(arr.shape), raw)
    # Fallback for strings, sub-byte types, and exotic dtypes.
    tp = numpy_helper.from_array(arr, name=name)
    return _runtime.tensor_from_proto(tp)


def _run_via_session(
    graph_or_function: Any, ctx: Any, sessions: dict[int, Any] | None = None
) -> None:
    """Runs ``graph_or_function`` by building (or reusing) its cached
    :class:`~onnx_light.onnx_py._onnxpykernels.runtime.ExecutionPlan` and
    driving it through a (reused)
    :class:`~onnx_light.onnx_py._onnxpykernels.runtime.RuntimeSession`.

    When ``graph_or_function`` is a ``GraphProto``, every declared
    initializer is seeded into ``ctx`` first (names ``ctx`` already carries
    are left as-is). ``FunctionProto`` has no initializers, so nothing is
    seeded in that case. For a full model, call
    :func:`_runtime.register_model_functions` first so nodes referring to
    model-local functions resolve, then call this on ``model.graph``.

    ``sessions`` is an optional cache mapping a plan's identity to the
    ``(plan, RuntimeSession)`` pair built for it. Reusing the session across
    calls keeps the per-node kernel resolution that :cpp:func:`RuntimeSession::Run`
    performs on its first run instead of rediscovering (and rebuilding) every
    node's kernel on each call. When ``sessions`` is ``None`` a fresh session
    is built for this single run.
    """
    initializers = getattr(graph_or_function, "initializer", None)
    if initializers is not None:
        for init in initializers:
            if not ctx.has(init.name):
                ctx.set(init.name, _runtime.tensor_from_proto(init), "initializer")
    plan = ctx.get_execution_plan(graph_or_function)
    if sessions is None:
        _runtime.RuntimeSession(plan).run(ctx)
        return
    entry = sessions.get(id(plan))
    if entry is None:
        # Cache the plan alongside the session so the plan (and therefore its
        # identity) stays alive for as long as the session that wraps it.
        entry = (plan, _runtime.RuntimeSession(plan))
        sessions[id(plan)] = entry
    entry[1].run(ctx)


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

    Parameters
    ----------
    verbose:
        Verbosity level forwarded to the runtime. ``0`` disables progress
        printing; positive values print one line per dispatched node while the
        model is executing.
    events_enabled:
        When ``True``, the runtime records a :class:`RuntimeEvent` for every
        tensor map mutation and node dispatch, retrievable through
        :meth:`events` after :meth:`run`.
    release_intermediates:
        When ``True`` (the default), the runtime frees each intermediate tensor
        as soon as its last consumer has run.
    allocator:
        Optional :class:`SimpleRawBufferAllocator` (or any
        ``RawBufferAllocator``) attached to the internal
        :class:`RuntimeContext`. When provided together with
        ``events_enabled=True``, every recorded :class:`RuntimeEvent` carries
        the allocator's live (``allocated_bytes``) and peak (``peak_bytes``)
        memory, so the event log doubles as a per-node memory profile. The
        allocator must have enough slot ``capacity`` for the number of buffers
        alive at the same time; the caller retains ownership.

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
        verbose: int = 0,
        events_enabled: bool = False,
        release_intermediates: bool = True,
        allocator: Any = None,
    ) -> None:
        proto = self._load_proto(proto)
        if not isinstance(verbose, int):
            raise TypeError(f"verbose must be an integer, not {type(verbose).__name__}.")
        if verbose < 0:
            raise ValueError(f"verbose must be non-negative, got {verbose}.")
        self._model: ModelProto | None = None
        self._graph: GraphProto | None = None
        self._function: FunctionProto | None = None
        self._last_ctx: Any = None
        self._verbose = verbose
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

        # ``_map_inputs`` records the names of graph inputs declared as ``map(K, V)``.
        # Map-typed inputs are fed as a Python ``dict``; :meth:`run` stores them
        # via ``ctx.put_map`` so the runtime can retrieve them by the original name.
        self._map_inputs: set[str] = set()
        # ``_sequence_inputs`` records graph inputs declared as ``seq(T)``. Such
        # inputs are fed as a Python ``list``/``tuple`` of arrays (one per
        # sequence element) and handed to the runtime via ``put_sequence``
        # instead of the single-tensor ``set`` path.
        self._sequence_inputs: set[str] = set()
        # ``_optional_sequence_inputs`` records graph inputs declared as
        # ``optional(seq(T))``. The runtime models present optionals as a
        # passthrough of the underlying value, so these inputs are fed through
        # the same ``put_sequence`` path as plain ``seq(T)`` inputs.
        self._optional_sequence_inputs: set[str] = set()
        if graph_inputs is not None:
            inputs: list[str] = []
            for vi in graph_inputs:
                if vi.name in initializers:
                    continue
                if vi.type.has_map_type():
                    # Map-typed inputs are stored in the runtime's map store
                    # under the original graph-input name; the caller feeds a
                    # Python ``dict`` and :meth:`run` converts it to a Map.
                    inputs.append(vi.name)
                    self._map_inputs.add(vi.name)
                else:
                    if vi.type.has_sequence_type():
                        self._sequence_inputs.add(vi.name)
                    elif (
                        vi.type.has_optional_type()
                        and vi.type.optional_type.has_elem_type()
                        and vi.type.optional_type.elem_type.has_sequence_type()
                        and (
                            vi.type.optional_type.elem_type.sequence_type.elem_type.has_tensor_type()
                        )
                    ):
                        self._optional_sequence_inputs.add(vi.name)
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

        # Build the RuntimeContext once at construction time and reuse it for
        # every :meth:`run` call. Keeping a single context alive across runs
        # amortises the per-model ExecutionPlan analysis (cached inside the
        # context, keyed by graph/function address) instead of rebuilding it on
        # every call. The per-invocation tensor / sequence / event state is
        # reset via ``RuntimeContext.clear`` at the start of each :meth:`run`.
        self._ctx = _runtime.RuntimeContext(
            self._kernel_ctx,
            verbose=self._verbose,
            events_enabled=self._events_enabled,
            allocator=allocator,
        )
        # Keep a Python reference to the allocator so it outlives the context
        # (the binding also keeps it alive) and attach it so the runtime routes
        # buffer storage through it and records its live / peak memory on every
        # event.
        self._allocator = allocator

        # Mapping ``"<domain>:<op_type>" -> low-level callback``. A
        # low-level callback has the signature
        # ``fn(node: NodeProto, ctx: RuntimeContext) -> None`` and is
        # registered directly on the persistent RuntimeContext when the
        # higher-level :meth:`register_custom_kernel` API is called. That
        # API installs a numpy-friendly wrapper that delegates to the
        # user-provided callable.
        self._custom_kernels: dict[str, Any] = {}

        # Cache of ``id(plan) -> (plan, RuntimeSession)`` so the RuntimeSession
        # is reused across :meth:`run` calls. The first run of a session
        # resolves and builds every node's kernel once; reusing the session
        # keeps that work instead of redoing the per-node dispatch on every
        # call. Registering a custom kernel clears the cache so the next run
        # rebuilds the sessions and picks the new kernel up.
        self._sessions: dict[int, Any] = {}

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
        # Register the wrapper directly on the persistent RuntimeContext. A
        # later registration for the same (domain, op_type) overwrites the
        # previous one, matching the dict-based bookkeeping above.
        self._ctx.register_custom_kernel(domain, op_type, _wrapper)
        # Drop any cached RuntimeSession: its kernels were resolved before this
        # custom kernel existed, so the next run must rebuild them to pick it up.
        self._sessions.clear()

    def unregister_custom_kernel(self, domain: str, op_type: str) -> bool:
        """Removes a custom kernel previously registered for ``(domain, op_type)``.

        Custom kernels are consulted before the built-in onnx-light
        dispatch table, so unregistering one restores the original
        built-in kernel for that ``(domain, op_type)`` when there is one
        (a subsequent :meth:`run` dispatches to it again). If no built-in
        kernel exists for the pair, running a graph that uses it fails
        with an ``unsupported op_type`` error, as it would before any
        custom kernel was registered.

        Parameters
        ----------
        domain:
            Operator domain. The empty string is treated as
            ``ai.onnx``.
        op_type:
            Operator name (``NodeProto.op_type``).

        Returns
        -------
        bool
            ``True`` when a custom kernel was removed, ``False`` when no
            custom kernel was registered for ``(domain, op_type)``.

        Examples
        --------
        .. code-block:: python

            sess.register_custom_kernel("", "Abs", lambda node, x: -x)
            sess.unregister_custom_kernel("", "Abs")  # restores built-in Abs
        """
        key = f"{domain or 'ai.onnx'}:{op_type}"
        if key not in self._custom_kernels:
            return False
        del self._custom_kernels[key]
        # Remove the wrapper from the persistent RuntimeContext so the next
        # run falls back to the built-in kernel dispatch table (the original
        # kernel) for this (domain, op_type).
        self._ctx.unregister_custom_kernel(domain, op_type)
        # Drop any cached RuntimeSession: its kernels were resolved while the
        # custom kernel existed, so the next run must rebuild them to pick up
        # the restored built-in kernel.
        self._sessions.clear()
        return True

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

        Map-typed (``map(K, V)``) inputs are listed under their original
        graph-input name.  These are fed as a Python ``dict`` (e.g.
        ``{"x": {10: 1.5, 30: 2.5}}``) when calling :meth:`run`.
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

        Each entry is a ``RuntimeEvent`` object with an :meth:`as_dict` method
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
    ) -> list[np.ndarray | list[np.ndarray]]:
        """Executes the wrapped graph / model / function.

        Parameters
        ----------
        output_names:
            Names of the outputs to return. ``None`` is shorthand for
            "every declared output, in declaration order".
        feed_inputs:
            Mapping of input name to value. Tensor inputs are fed as a
            :class:`numpy.ndarray`; ``seq(T)`` inputs are fed as a ``list``
            (or ``tuple``) of arrays, one per sequence element; ``map(K, V)``
            inputs are fed as a Python ``dict`` (e.g. ``{"x": {10: 1.5}}``).
            Every name listed by :attr:`input_names` must be present.

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

        ctx = self._ctx
        # Reset the per-invocation tensor / sequence / event state from any
        # previous run while preserving the cached execution plans, registered
        # custom kernels, kernel context and the ``events_enabled`` setting.
        ctx.clear()
        # Releasing intermediates would drop any requested output that is
        # not a declared graph/function output before the caller can fetch
        # it. Disable the per-run release in that case so callers can still
        # observe arbitrary intermediate values via ``run([name], ...)``.
        declared_outputs = frozenset(self._output_names)
        release = self._release_intermediates and all(
            name in declared_outputs for name in output_names
        )
        ctx.release_intermediates = release

        for name, value in feed_inputs.items():
            is_optional_sequence_input = name in self._optional_sequence_inputs
            if name in self._map_inputs:
                # ``map(K, V)`` inputs are fed as a Python ``dict`` and stored
                # in the runtime's map store via ``put_map``.  A size-1 numpy
                # object array wrapping the dict (e.g. produced by
                # ``np.asarray(some_dict, dtype=object)``) is unwrapped first.
                if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
                    unwrapped = value.item()
                    if isinstance(unwrapped, dict):
                        value = unwrapped
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Map input {name!r} must be fed as a Python dict, "
                        f"not {type(value).__name__}."
                    )
                ctx.put_map(name, value)
            elif name in self._sequence_inputs or is_optional_sequence_input:
                # ``seq(T)`` graph inputs are fed as a list/tuple of arrays (one
                # per sequence element) and stored through ``put_sequence``.
                if not isinstance(value, (list, tuple)):
                    input_type_description = (
                        "optional sequence" if is_optional_sequence_input else "sequence"
                    )
                    raise TypeError(
                        f"{input_type_description.capitalize()} input {name!r} must be fed as a "
                        f"list/tuple of arrays, not {type(value).__name__}."
                    )
                elements = [
                    _numpy_to_cpp_tensor(f"{name}_{i}", element)
                    for i, element in enumerate(value)
                ]
                ctx.put_sequence(name, elements)
            else:
                ctx.set(name, _numpy_to_cpp_tensor(name, value))

        if self._model is not None:
            _runtime.register_model_functions(self._model, ctx)
            _run_via_session(self._model.graph, ctx, self._sessions)
        elif self._function is not None:
            _run_via_session(self._function, ctx, self._sessions)
        else:
            _run_via_session(self._graph, ctx, self._sessions)

        self._last_ctx = ctx

        results: list[np.ndarray | list[np.ndarray]] = []
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
