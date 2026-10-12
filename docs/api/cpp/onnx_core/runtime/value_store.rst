value_store.h
==========================

``ValueStore`` owns immutable values prepared synchronously while
a ``RuntimeSession`` hierarchy initializes its kernels. ``RuntimeContext``
always exposes a store, and child contexts inherit the same shared instance.
The first ``RuntimeSession::Run`` retains that store for the session hierarchy.
Kernels bind a stable ``uint32_t`` slot during initialization and read that
slot directly during execution, without a map lookup, mutex, readiness wait,
or scheduler dispatch on the hot path.

This store is deliberately narrower than ``PreparedObjectStore``. It does not
schedule asynchronous loading, track generations, evict values, or persist
prepared payloads. Those responsibilities remain in
``PreparedExecutionState`` and ``PreparedExecutionPlan``.

.. doxygenfile:: onnx_core/runtime/kernels/value_store.h
   :project: onnx-light
