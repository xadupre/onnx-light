
.. _l-next-steps:

Next Steps
==========

:Date: 2026-08

Recommended implementation order
--------------------------------

Runtime execution work should proceed in dependency order:

1. implement Profile PR01 from
   :ref:`l-next-steps-parallel-for-profiling`, starting with portable bounded
   events and a disabled path with no instrumentation work;
2. add process CPU time, inspection, hardware counters, and calibration
   diagnostics only after that event contract is stable;
3. integrate the same executor with :ref:`l-next-steps-prepared-execution`
   rather than creating another scheduler pool.

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
    :caption: Ready to implement

    2026/2026-08_parallel_for_profiling
    2026/2026-08_model_loading

.. toctree::
    :maxdepth: 1
    :caption: Discussion

    2026/2026-08_custom_types
    2026/2026-08_proto_inheritance
    2026/2026-08_quantization
    2026/2026-08_graph_builder_quantized_tensor
    2026/2026-08_mutable_cache
    2026/2026-08_compiled_tensor
    2026/2026-08_model_resolution
    2026/2026-08_prepared_execution
    2026/2026-08_split_wheels

.. toctree::
    :maxdepth: 1
    :caption: Completed

    2025-07_onnx_proto
    2026/2026-06_lib_onnx
    2026/2026-06_kernels_backend_tests
    2026/2026-06_gradient
    2026/2026-07_onnxruntime_onnx_light
    2026/2026-08_proto_binary_size
    2026/2026-08_processor_aware_kernel_tuning
    2026/2026-08_buffer_reuse_arena
    2026/2026-08_graph_builder_optimization
    2026/2026-08_session_execution_pools
