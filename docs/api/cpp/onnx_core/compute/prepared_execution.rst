prepared_execution.h
====================

``PrepareAsync`` and ``RunAsync`` take their task callback by value and move
it into the submission. Temporary lambdas, named lambdas, and temporary
``PreparedTaskExecutor`` conversions are safe: the caller need not keep the
original callable alive. An lvalue ``std::function`` is copied, so replacing
or destroying that original function does not change the submitted callback.

This ownership does not extend to objects captured by reference or through
raw pointers. The caller must keep those objects, the ``PreparedExecutionState``,
and any supplied ``CpuExecutor`` alive until completion. ``RunAsync`` also
borrows its plan; ``PrepareAsync`` retains its own session-only plan.
Destroying or replacing a live ``ExecutionHandle`` waits for its owned work
to finish. Cancellation and error reporting retain their existing semantics.

.. doxygenfile:: onnx_core/compute/prepared_execution.h
