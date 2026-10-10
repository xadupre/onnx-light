execution_plan.h
================

``ExecutionPlan`` is the immutable, per-graph action schedule replayed by
``RuntimeSession``. It records ordered allocation, shape, node-execution,
release, lock, and unlock actions derived from graph topology and metadata.
It does not own runtime tensor values, resolved kernel instances, or prepared
kernel buffers.

The ownership boundaries are:

* ``ExecutionPlan`` owns the reusable action and node sequence.
* ``RuntimeSession`` resolves and retains one kernel instance per planned node.
* ``RuntimeContext`` owns invocation inputs, outputs, intermediates, and
  execution allocations.
* ``KernelPreparationStore`` owns immutable, synchronously prepared kernel
  values for the root session and its nested sessions.
* ``PreparedExecutionPlan`` is the separate scope-aware dependency graph used
  when loading, preparation, publication, and invocation work must overlap.

Consequently, direct ``ExecutionPlan`` replay remains a synchronous hot path:
kernel initialization prepares constant inputs once, then each run reads the
kernel's stable preparation slot and replays only the ordinary execution
actions. The general prepared-execution scheduler remains available for
asynchronous loading, priorities, generations, cancellation, persistence, and
eviction; it is not consulted by this direct replay path.

.. doxygenfile:: onnx_core/compute/execution_plan.h
   :project: onnx-light
