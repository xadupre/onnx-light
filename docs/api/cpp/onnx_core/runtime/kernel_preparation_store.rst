kernel_preparation_store.h
==========================

``KernelPreparationStore`` owns immutable values prepared synchronously while
a ``RuntimeSession`` hierarchy initializes its kernels. A root session creates
one store; control-flow subgraphs and model-local functions inherit it. Kernels
bind a stable integer slot during initialization and read that slot directly
during execution, without a map lookup, mutex, readiness wait, or scheduler
dispatch on the hot path.

This store is deliberately narrower than ``PreparedObjectStore``. It does not
schedule asynchronous loading, track generations, evict values, or persist
prepared payloads. Those responsibilities remain in
``PreparedExecutionState`` and ``PreparedExecutionPlan``.

.. doxygenfile:: onnx_core/runtime/kernels/kernel_preparation_store.h
   :project: onnx-light
