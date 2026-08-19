# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of :class:`ReferenceEvaluator`.

The evaluator delegates input conversion, execution, and output conversion to
the native ``ReferenceEvaluatorRunner`` exposed by
:mod:`onnx_light.onnx_py._onnxpykernels`. All operator implementations come
from the C++ ``KernelDispatchTable``; this Python class provides the
``onnx.reference.ReferenceEvaluator``-compatible API and custom-kernel
wrapping.
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


_CPU_EXECUTION_KEYS = {
    "num_threads",
    "spin_policy",
    "spin_budget",
    "affinity_policy",
    "cpu_set",
    "allow_nested_parallelism",
}
_CPU_SPIN_POLICIES = {
    "adaptive": _runtime.CpuSpinPolicy.ADAPTIVE,
    "fixed_iterations": _runtime.CpuSpinPolicy.FIXED_ITERATIONS,
    "fixed_duration": _runtime.CpuSpinPolicy.FIXED_DURATION,
    "park_immediately": _runtime.CpuSpinPolicy.PARK_IMMEDIATELY,
}
_CPU_AFFINITY_POLICIES = {
    "none": _runtime.CpuAffinityPolicy.NONE,
    "physical_cores": _runtime.CpuAffinityPolicy.PHYSICAL_CORES,
    "performance_cores": _runtime.CpuAffinityPolicy.PERFORMANCE_CORES,
    "physical_then_smt": _runtime.CpuAffinityPolicy.PHYSICAL_THEN_SMT,
    "explicit": _runtime.CpuAffinityPolicy.EXPLICIT,
}


def _parse_cpu_execution(cpu_execution: Any) -> Any:
    """Validates and converts a CPU execution policy."""
    if cpu_execution is None:
        return None
    if isinstance(cpu_execution, _runtime.CpuExecutionPolicy):
        _runtime.resolve_cpu_execution_policy(cpu_execution)
        return cpu_execution
    if not isinstance(cpu_execution, dict):
        raise TypeError(
            "cpu_execution must be a dict, CpuExecutionPolicy, or None, "
            f"not {type(cpu_execution).__name__}."
        )
    unknown = set(cpu_execution) - _CPU_EXECUTION_KEYS
    if unknown:
        raise ValueError(f"Unknown cpu_execution keys: {', '.join(sorted(unknown))}.")

    policy = _runtime.CpuExecutionPolicy()
    if "num_threads" in cpu_execution:
        value = cpu_execution["num_threads"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("cpu_execution['num_threads'] must be an integer.")
        policy.num_threads = value
    if "spin_policy" in cpu_execution:
        value = cpu_execution["spin_policy"]
        if isinstance(value, str):
            if value not in _CPU_SPIN_POLICIES:
                raise ValueError(f"Unknown CPU spin policy {value!r}.")
            value = _CPU_SPIN_POLICIES[value]
        policy.spin_policy = value
    if "spin_budget" in cpu_execution:
        value = cpu_execution["spin_budget"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("cpu_execution['spin_budget'] must be a non-negative integer.")
        policy.spin_budget = value
    if "affinity_policy" in cpu_execution:
        value = cpu_execution["affinity_policy"]
        if isinstance(value, str):
            if value not in _CPU_AFFINITY_POLICIES:
                raise ValueError(f"Unknown CPU affinity policy {value!r}.")
            value = _CPU_AFFINITY_POLICIES[value]
        policy.affinity_policy = value
    if "cpu_set" in cpu_execution:
        processors = []
        for value in cpu_execution["cpu_set"]:
            if isinstance(value, _runtime.CpuLogicalProcessor):
                processors.append(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                processors.append(_runtime.CpuLogicalProcessor(value))
            elif isinstance(value, dict):
                unknown_processor_keys = set(value) - {"id", "group"}
                if unknown_processor_keys or "id" not in value:
                    raise ValueError(
                        "Each cpu_set mapping must contain 'id' and may contain only 'group'."
                    )
                processors.append(
                    _runtime.CpuLogicalProcessor(value["id"], value.get("group", 0))
                )
            else:
                raise TypeError(
                    "cpu_execution['cpu_set'] entries must be integers, mappings, "
                    "or CpuLogicalProcessor objects."
                )
        policy.cpu_set = processors
    if "allow_nested_parallelism" in cpu_execution:
        value = cpu_execution["allow_nested_parallelism"]
        if not isinstance(value, bool):
            raise TypeError("cpu_execution['allow_nested_parallelism'] must be a bool.")
        policy.allow_nested_parallelism = value

    _runtime.resolve_cpu_execution_policy(policy)
    return policy


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
_DEFAULT_ARENA_CAPACITY = 4096
_DEFAULT_ARENA_EXTRA_SLOTS = 4


def _cpp_tensor_to_numpy(t: Any, steal: bool = False) -> np.ndarray:
    """Converts a runtime ``Tensor`` to a :class:`numpy.ndarray`.

    For standard fixed-width dtypes (float32, int64, etc.) the array is a
    zero-copy view imported through the DLPack exchange protocol
    (``Tensor.__dlpack__`` / :func:`numpy.from_dlpack`); the DLPack capsule
    keeps the source tensor alive for the lifetime of the returned array so the
    borrowed view never dangles. bfloat16/float8 tensors are reinterpreted from
    the raw byte view returned by :func:`_runtime.tensor_to_numpy` (DLPack has
    no stock NumPy dtype for them), and sub-byte packed types and STRING
    tensors fall back to the full :func:`numpy_helper.to_array` path.

    When ``steal`` is ``True`` and ``t`` owns its bytes inline (the case for a
    graph output produced without an allocator), the raw-byte path transfers
    the buffer's ownership to NumPy instead of borrowing it, so the source
    tensor can be released while the array lives on. ``steal`` must only be set
    for terminal tensors that are not consumed elsewhere.
    """
    dt = int(t.data_type)
    if not _IS_BIG_ENDIAN and dt in _DLPACK_DTYPES:
        # Zero-copy import via the DLPack protocol; ``from_dlpack`` yields a
        # correctly shaped, native-dtype array sharing the tensor's buffer.
        return np.from_dlpack(t)
    np_dtype = _DTYPE_TO_NP.get(dt)
    if np_dtype is not None:
        # ``tensor_to_numpy`` returns a 1-D uint8 array over the tensor's bytes;
        # reinterpret it as ``np_dtype`` and reshape. With ``steal`` the array
        # owns the (moved-out) bytes; otherwise it borrows them. Also used on
        # big-endian hosts, where the little-endian buffer must be byte-swapped
        # after reinterpretation.
        shape = t.shape
        raw = _runtime.tensor_to_numpy(t, steal)
        arr = raw.view(np_dtype)
        if _IS_BIG_ENDIAN:  # pragma: no cover
            arr = arr.byteswap()
        return arr.reshape(shape)
    # Fallback for sub-byte types (INT4/UINT4/INT2/UINT2/FLOAT4E2M1) and STRING.
    return numpy_helper.to_array(_runtime.tensor_to_proto(t))


def _numpy_to_cpp_tensor(name: str, arr: np.ndarray, copy: bool = True) -> Any:
    """Converts a :class:`numpy.ndarray` to a runtime ``Tensor``.

    For standard fixed-width dtypes, copies the array's raw bytes directly
    into the C++ ``Tensor`` via :func:`_runtime.tensor_from_numpy`, avoiding
    the triple-copy overhead of serializing through a ``TensorProto``
    intermediate.
    """
    is_ndarray = isinstance(arr, np.ndarray)
    if not is_ndarray:
        arr = np.asarray(arr)
    onnx_dtype = _NP_TO_DTYPE.get(arr.dtype.type)
    if onnx_dtype is not None:
        needs_copy = copy or not is_ndarray
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            needs_copy = True
        if arr.ndim == 0:
            # Some kernels read scalar bytes from owned tensor storage directly.
            # Keep scalar inputs on the owned-copy path to preserve that contract.
            needs_copy = True
            raw = arr.reshape((1,)).view(np.uint8)
        else:
            raw = arr.view(np.uint8).ravel()
        return _runtime.tensor_from_numpy(name, onnx_dtype, list(arr.shape), raw, copy=needs_copy)
    # Fallback for strings, sub-byte types, and exotic dtypes.
    tp = numpy_helper.from_array(arr, name=name)
    return _runtime.tensor_from_proto(tp)


def _make_numpy_custom_kernel(domain: str, op_type: str, fn: Any) -> Any:
    """Wraps a numpy-friendly custom kernel ``fn`` as a low-level callback.

    The returned callable follows the ``fn(node, ctx)`` contract expected by
    the ``register_custom_kernel`` bindings: it reads the node inputs from
    ``ctx`` as :class:`numpy.ndarray`, invokes ``fn(node, *inputs)`` and writes
    the returned array(s) back into ``ctx``. Shared by
    :meth:`ReferenceEvaluator.register_custom_kernel` (per-session) and
    :meth:`ReferenceEvaluator.register_custom_kernel_global` (process-wide).
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

    return _wrapper


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ReferenceEvaluator:
    """Evaluates an ONNX model using the C++ ``KernelDispatchTable``.

    The class is constructed from a ``ModelProto`` / ``GraphProto`` /
    ``FunctionProto`` (or the bytes / file path of a serialised
    ``ModelProto``). :meth:`run` then takes a feed dictionary whose tensor
    values may be NumPy arrays or runtime ``Tensor`` objects and returns the
    requested outputs as a list of NumPy arrays, mirroring the calling
    convention of :class:`onnx.reference.ReferenceEvaluator` (and
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
    io_allocator:
        Optional :class:`IOArena` (or any ``RawBufferAllocator``) dedicated to
        declared graph outputs. When neither allocator is provided, the
        evaluator creates persistent execution and I/O arenas and reuses them
        across runs. Passing only one allocator creates a default persistent
        arena for the other lifetime domain; pass the same allocator as both
        ``allocator`` and ``io_allocator`` to request single-allocator
        behaviour explicitly.

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
        io_allocator: Any = None,
        cpu_execution: Any = None,
        cpu_execution_counters: bool = False,
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
        self._cpu_execution = _parse_cpu_execution(cpu_execution)
        if not isinstance(cpu_execution_counters, bool):
            raise TypeError("cpu_execution_counters must be a bool.")

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
        create_execution_arena = allocator is None
        create_io_arena = io_allocator is None
        if create_execution_arena or create_io_arena:
            if self._graph is not None:
                node_output_slots = sum(len(node.output) for node in self._graph.node)
                initializer_slots = len(self._graph.initializer)
            elif self._function is not None:
                node_output_slots = sum(len(node.output) for node in self._function.node)
                initializer_slots = 0
            else:
                node_output_slots = 0
                initializer_slots = 0
            arena_capacity = max(
                _DEFAULT_ARENA_CAPACITY,
                len(self._input_names)
                + len(self._output_names)
                + initializer_slots
                + node_output_slots
                + _DEFAULT_ARENA_EXTRA_SLOTS,
            )
            if create_execution_arena:
                allocator = _runtime.ExecutionArena(arena_capacity)
            if create_io_arena:
                io_allocator = _runtime.IOArena(arena_capacity)

        self._ctx = _runtime.RuntimeContext(
            self._kernel_ctx,
            verbose=self._verbose,
            events_enabled=self._events_enabled,
            allocator=allocator,
            io_allocator=io_allocator,
        )
        if self._model is not None:
            _runtime.register_model_functions(self._model, self._ctx)

        # Mapping ``"<domain>:<op_type>" -> low-level callback``. A
        # low-level callback has the signature
        # ``fn(node: NodeProto, ctx: RuntimeContext) -> None`` and is
        # registered directly on the persistent RuntimeContext when the
        # higher-level :meth:`register_custom_kernel` API is called. That
        # API installs a numpy-friendly wrapper that delegates to the
        # user-provided callable.
        self._custom_kernels: dict[str, Any] = {}

        if self._model is not None:
            execution_root = self._model.graph
        elif self._function is not None:
            execution_root = self._function
        else:
            execution_root = self._graph
        self._runner = _runtime.ReferenceEvaluatorRunner(
            execution_root,
            self._input_names,
            self._map_inputs,
            self._sequence_inputs | self._optional_sequence_inputs,
            self._output_names,
            _runtime.RuntimeSessionOptions(
                cpu_execution=self._cpu_execution, cpu_execution_counters=cpu_execution_counters
            ),
        )
        self._last_ctx = self._ctx

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
        _wrapper = _make_numpy_custom_kernel(domain, op_type, fn)

        self._custom_kernels[f"{domain or 'ai.onnx'}:{op_type}"] = (domain, op_type, _wrapper)
        # Register the wrapper directly on the persistent RuntimeContext. A
        # later registration for the same (domain, op_type) overwrites the
        # previous one, matching the dict-based bookkeeping above.
        self._ctx.register_custom_kernel(domain, op_type, _wrapper)
        # Drop any cached RuntimeSession: its kernels were resolved before this
        # custom kernel existed, so the next run must rebuild them to pick it up.
        self._runner.reset()

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
        self._runner.reset()
        return True

    @staticmethod
    def register_custom_kernel_global(domain: str, op_type: str, fn: Any) -> None:
        """Registers a process-wide (global) numpy custom kernel.

        Unlike :meth:`register_custom_kernel`, which only affects the
        evaluator it is called on, a global kernel is picked up by every
        :class:`ReferenceEvaluator` (and any other runtime context). Register
        the kernel *before* running an evaluator, since an evaluator caches its
        runtime sessions on first run and only rebuilds them when its own
        (per-session) registrations change.

        ``fn`` follows the same ``fn(node, *inputs)`` numpy contract as
        :meth:`register_custom_kernel`. A per-session registration for the same
        ``(domain, op_type)`` overrides the global one.

        Parameters
        ----------
        domain:
            Operator domain. The empty string is treated as ``ai.onnx``.
        op_type:
            Operator name (``NodeProto.op_type``).
        fn:
            Python callable invoked as ``fn(node, *inputs)``; see
            :meth:`register_custom_kernel`.

        Examples
        --------
        .. code-block:: python

            def square(node, x):
                return x * x

            ReferenceEvaluator.register_custom_kernel_global("my.domain", "Square", square)
        """
        _wrapper = _make_numpy_custom_kernel(domain, op_type, fn)
        _runtime.register_custom_kernel(domain, op_type, _wrapper)

    @staticmethod
    def unregister_custom_kernel_global(domain: str, op_type: str) -> bool:
        """Removes a process-wide custom kernel registered by
        :meth:`register_custom_kernel_global`.

        The empty domain is normalised to ``ai.onnx``. Returns ``True`` when a
        global custom kernel was removed, ``False`` otherwise. Note that
        evaluators which already built (and cached) their runtime sessions keep
        dispatching to the previously resolved kernel until their sessions are
        rebuilt.

        Parameters
        ----------
        domain:
            Operator domain. The empty string is treated as ``ai.onnx``.
        op_type:
            Operator name (``NodeProto.op_type``).

        Returns
        -------
        bool
            ``True`` when a global custom kernel was removed.
        """
        return bool(_runtime.unregister_custom_kernel(domain, op_type))

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

    def used_kernels(self) -> list[str]:
        """Returns the kernel identifiers used by the evaluator.

        Identifiers use the normalized ``"<domain>:<op_type>"`` form and
        follow execution order. Repeated operators are preserved because each
        node has its own kernel instance. Returns an empty list before the first
        :meth:`run`.
        """
        return self._runner.used_kernels()

    @property
    def cpu_execution_policy(self) -> Any:
        """Returns the requested CPU execution policy."""
        return self._runner.cpu_execution_policy

    @property
    def cpu_execution_resolution(self) -> Any:
        """Returns the immutable CPU policy resolved for this evaluator."""
        return self._runner.cpu_execution_resolution

    @property
    def cpu_execution_identity(self) -> Any:
        """Returns the sharing key of this evaluator's leased executor."""
        return self._runner.cpu_execution_identity

    @property
    def cpu_execution_counters(self) -> Any:
        """Returns cumulative counters for the shared executor.

        Compatible evaluators lease the same executor, so enabling counters on
        any lease enables them for that executor and the snapshot includes
        dispatches from every compatible leaseholder.
        """
        return self._runner.cpu_execution_counters

    @property
    def cpu_executor_instance_id(self) -> int:
        """Returns the process-local identity of the leased CPU executor.

        Compatible evaluators that lease the same executor return the same
        value. This diagnostic identity is not stable across processes and
        must not be used as a tuning-cache key.
        """
        return self._runner.cpu_executor_instance_id

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
            :class:`numpy.ndarray` or a runtime ``Tensor``; both forms may be
            mixed in one call. ``seq(T)`` inputs are fed as a ``list`` (or
            ``tuple``) of arrays or runtime tensors, one per sequence element;
            ``map(K, V)`` inputs are fed as a Python ``dict`` (e.g.
            ``{"x": {10: 1.5}}``). Every name listed by :attr:`input_names`
            must be present.

        Returns
        -------
        list of :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            One entry per name in ``output_names`` (defaults to
            :attr:`output_names`), in the requested order. Tensor-typed
            outputs are returned as :class:`numpy.ndarray`; sequence-typed
            outputs are returned as a ``list`` of :class:`numpy.ndarray`
            (one array per sequence element).
        """
        return self._runner.run(self._ctx, output_names, feed_inputs, self._release_intermediates)
