runtime_session.h
=================

Sessions, including subgraphs, read directly usable initializer storage from
the immutable source graph. No ownership mode is required. The graph and its
backing buffers must outlive the session and its initializer views, including
when supplied through ``SetInitializers``.

Host initializers are not copied into the execution allocator for CPU execution
(``kCPU`` or the default ``kUndefined`` device). Raw storage and native float,
double, int32, int64 and uint64 fields can be borrowed; representations requiring
decoding, including strings, still use normal tensor conversion. This host
borrowing rule does not implement transfers to another device.

The context can supply a model lifetime token for selected retained results;
ownerless externally borrowed buffers cannot be retained. Nonpersistent graph
outputs still materialize borrowed storage normally so they can outlive the model.

Kernel preparation
------------------

The first ``Run`` resolves every planned kernel and invokes preparation hooks
for immutable inputs. The root session creates one ``KernelPreparationStore``;
``If``, ``Loop``, ``Scan``, ``SequenceMap``, and model-local function sessions
inherit the same store. A kernel binds a stable integer slot and retains it for
the session lifetime. Subsequent runs read the slot directly, so ordinary
``ExecutionPlan`` replay does not wait on readiness, acquire a preparation
mutex, or repeat a string-keyed lookup.

``Gemm`` currently uses this path for a constant ``B`` initializer. An
initializer that is also a graph input, or a value supplied by the caller
before initialization, remains overridable and is not prepared.
``RuntimeSession::prepared_bytes`` reports the allocations owned by the shared
hierarchy store, so the root session's value includes preparations created by
its nested sessions.

This synchronous store is not ``PreparedObjectStore``: asynchronous external
data loading, dependency scheduling, generations, cache persistence, and
eviction remain responsibilities of ``PreparedExecutionPlan`` and
``PreparedExecutionState``.

.. doxygenfile:: onnx_core/runtime/runtime_session.h
   :project: onnx-light
