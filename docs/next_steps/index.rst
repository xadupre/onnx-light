
.. _l-next-steps:

Next Steps
==========

:Date: 2026-08

Recommended implementation order
--------------------------------

Runtime execution work should proceed in dependency order:

1. implement :ref:`l-next-steps-session-execution-pools` PR01--PR03 so session
   parameters control a real, inspectable executor;
2. add tuning/Python inspection in Pool PR04, then migrate registered
   ``onnx-light-cpu`` kernels in Pool PR05;
3. build :ref:`l-next-steps-parallel-for-profiling` on the truthful executor
   and connect it to calibration;
4. integrate that executor with :ref:`l-next-steps-prepared-execution` rather
   than creating another scheduler pool;
5. run the cross-repository Pool PR07 performance gate before tuning kernels
   against the new execution identity.

Model-format work such as custom types, quantization, compiled tensors, and
model resolution may proceed independently until it reaches prepared
execution. Within the runtime track, the order above is mandatory: profiling
or tuning a hidden global pool would produce profiles that a session cannot
reproduce.

For large-model startup, begin with the benchmark and direct parser work in
:ref:`l-next-steps-model-loading`, then follow its dependencies through model
resolution, compiled tensors, and prepared execution.

.. toctree::
    :maxdepth: 1
    :caption: In progress

    2026-08_graph_builder_optimization
    2026-08_session_execution_pools

.. toctree::
    :maxdepth: 1
    :caption: Discussion

    2026-08_custom_types
    2026-08_proto_inheritance
    2026-08_quantization
    2026-08_graph_builder_quantized_tensor
    2026-08_mutable_cache
    2026-08_compiled_tensor
    2026-08_parallel_for_profiling
    2026-08_model_loading
    2026-08_model_resolution
    2026-08_prepared_execution
    2026-08_split_wheels

.. toctree::
    :maxdepth: 1
    :caption: Completed

    2025-07_onnx_proto
    2026-06_lib_onnx
    2026-06_kernels_backend_tests
    2026-06_gradient
    2026-07_onnxruntime_onnx_light
    2026-08_proto_binary_size
    2026-08_processor_aware_kernel_tuning
    2026-08_buffer_reuse_arena
