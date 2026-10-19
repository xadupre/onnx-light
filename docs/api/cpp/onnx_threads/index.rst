Standalone threading
=====================

The implementation is shared by the existing proto/runtime facades. It does
not require ONNX model types; see :ref:`l-design-threading`.

Configuration and visibility
-----------------------------

.. doxygenfile:: onnx_threads/config.h
   :project: onnx-light

I/O batch adapter
------------------

.. doxygenfile:: onnx_threads/thread_pool.h
   :project: onnx-light

Queued workers
---------------

.. doxygenfile:: onnx_threads/worker_pool.h
   :project: onnx-light

Parallel regions
-----------------

.. doxygenfile:: onnx_threads/parallel_for.h
   :project: onnx-light

CPU descriptor
---------------

.. doxygenfile:: onnx_threads/cpu_descriptor.h
   :project: onnx-light

CPU execution policy
---------------------

.. doxygenfile:: onnx_threads/cpu_execution_policy.h
   :project: onnx-light

CPU executor
-------------

.. doxygenfile:: onnx_threads/cpu_executor.h
   :project: onnx-light

Parallel-region diagnostics
---------------------------

.. doxygenfile:: onnx_threads/parallel_region_collector.h
   :project: onnx-light

Completion state
-----------------

.. doxygenfile:: onnx_threads/task_completion.h
   :project: onnx-light

Bounded fork/join
------------------

.. doxygenfile:: onnx_threads/fork_join.h
   :project: onnx-light
