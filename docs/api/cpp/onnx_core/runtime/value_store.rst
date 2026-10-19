value_store.h
==========================

``ValueStore`` owns immutable values prepared synchronously while
a ``RuntimeSession`` hierarchy initializes its kernels. ``RuntimeContext``
always exposes a store, and child contexts inherit the same shared instance.
Each session owns a non-null store from construction and installs it into
the context before initializing kernels. Nested graph/function sessions
explicitly receive their parent's owner. Independent sessions do not adopt
the store of a reused context; cross-session sharing requires
``RuntimeSessionOptions::value_store``.
Kernels bind a stable ``uint32_t`` slot during initialization and read that
slot directly during execution, without a map lookup, mutex, readiness wait,
or scheduler dispatch on the hot path. All sessions sharing a store must
serialize their runs. Its allocations remain alive until the last shared
owner, including any context, is released.

The location under ``runtime/kernels/`` reflects its kernel-preparation role:
it is not the invocation value directory. "Static" means publication once per
slot followed by immutable replay, not a global C++ object or mutation-free
initialization. ``value_store()`` returns a reference;
``value_store_owner()`` returns shared ownership. Both session and context
accessors expose only read-only views through const objects.

This store is deliberately narrower than ``PreparedObjectStore``. It does not
schedule asynchronous loading, track generations, evict values, or persist
prepared payloads. Those responsibilities remain in
``PreparedExecutionState`` and ``PreparedExecutionPlan``.
Publishing through that dynamic path does not populate ``ValueStore``.

``prepared_bytes()`` sums live buffer sizes in this static store's allocators,
including allocated but unpublished values. It excludes retained free-buffer
capacity, execution/I/O arenas, and dynamic objects. There is no static-store
admission budget or eviction policy; dynamic scheduler budgets do not apply.

.. doxygenfile:: onnx_core/runtime/kernels/value_store.h
   :project: onnx-light
