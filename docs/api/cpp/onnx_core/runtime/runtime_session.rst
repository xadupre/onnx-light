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

Initializer views do not own the source model. C++ callers keep it alive through
all such views, including returned or retained results. Ownerless externally
borrowed buffers cannot be retained. Python bindings retain the source model
when exporting initializer views.

Kernel preparation
------------------

Construction fixes a non-null ``ValueStore`` owner, creating a private store
unless ``RuntimeSessionOptions::value_store`` explicitly supplies one.
The first ``Run`` resolves every planned kernel and invokes preparation hooks
for immutable inputs. Each run installs the session's store into the
``RuntimeContext``; it never adopts a context's previous store. ``If``,
``Loop``, ``Scan``, ``SequenceMap``, and model-local function sessions explicitly
inherit the parent's owner. A kernel binds a store-specific ``ValueStore::Slot``
with a stable integer index during initialization. After publication,
``BindForRead`` validates it once and returns a store-pointer/index binding.
``Gemm`` retains this binding for direct reads on subsequent runs, without
rechecking ownership or readiness, so ordinary
``ExecutionPlan`` replay does not wait on readiness, acquire a preparation
mutex, or repeat a string-keyed lookup.

``Gemm`` currently uses this path for a constant ``B`` initializer. An
initializer that is also a graph input, or a value supplied by the caller
before initialization, remains overridable and is not prepared.
``RuntimeSession::prepared_bytes`` reports live buffer sizes in the static
store's allocators, including allocated but unpublished values and preparations
created by nested sessions. It excludes retained free capacity, execution/I/O
arenas, and dynamic prepared-object residency. Explicitly sharing a store
between otherwise independent sessions also shares this accounting: it is
not a per-session partition. All sessions sharing one store must serialize
their runs. Contexts may keep the owner alive after a session is destroyed.

``value_store()`` returns a reference, available even before the first run.
Use ``value_store_owner()`` when passing shared ownership to options or a
nested session. Const session/context access returns ``const ValueStore&``
or ``std::shared_ptr<const ValueStore>``; initialization-time mutation and
sharing a mutable owner require a non-const session/context.

This synchronous store is not ``PreparedObjectStore``: asynchronous external
data loading, dependency scheduling, generations, cache persistence, and
eviction remain responsibilities of ``PreparedExecutionPlan`` and
``PreparedExecutionState``.
Dynamic preparation does not automatically publish values into the static
store, and its admission/residency budgets do not constrain static allocations.

.. doxygenfile:: onnx_core/runtime/runtime_session.h
   :project: onnx-light
