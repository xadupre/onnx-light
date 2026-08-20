
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
3. use that same executor when the fast-loading sequence reaches
   :ref:`l-next-steps-prepared-execution`, rather than creating another
   scheduler pool.

Model-format work such as custom types, quantization, compiled tensors, and
model resolution may proceed independently until it reaches prepared
execution. Within the runtime track, the order above is mandatory: profiling
or tuning a hidden global pool would produce profiles that a session cannot
reproduce.

Large-model startup follows one explicit four-document sequence:

1. fix existing parser, external-data, and initializer-materialization defects
   in :ref:`l-next-steps-model-loading-bug-fixes`;
2. implement :ref:`l-next-steps-prepared-execution`;
3. complete the ownership-aware cross-repository work in
   :ref:`l-next-steps-model-loading`;
4. connect adaptive I/O, model resolution, prepared tensors, and first-token
   overlap in :ref:`l-next-steps-native-fast-loading-completion`.

Parallel-for profiling may proceed alongside step 1, but its executor
instrumentation must be stable before step 2 begins.

Assignable issue sequence:

.. list-table::
    :header-rows: 1
    :widths: 12 24 64

    * - Step
      - Issues
      - Order
    * - 1. Bug fixes
      - #4608--#4610
      - `#4608 <https://github.com/xadupre/onnx-light/issues/4608>`_ ->
        `#4609 <https://github.com/xadupre/onnx-light/issues/4609>`_ ->
        `#4610 <https://github.com/xadupre/onnx-light/issues/4610>`_
    * - 2. Prepared execution
      - #4613--#4617
      - `#4613 <https://github.com/xadupre/onnx-light/issues/4613>`_ ->
        `#4614 <https://github.com/xadupre/onnx-light/issues/4614>`_ ->
        `#4615 <https://github.com/xadupre/onnx-light/issues/4615>`_ ->
        `#4616 <https://github.com/xadupre/onnx-light/issues/4616>`_ ->
        `#4617 <https://github.com/xadupre/onnx-light/issues/4617>`_
    * - 3. onnxruntime
      - #4611--#4612
      - `#4611 <https://github.com/xadupre/onnx-light/issues/4611>`_ ->
        `#4612 <https://github.com/xadupre/onnx-light/issues/4612>`_. #4612 is
        coordination for a PR in ``microsoft/onnxruntime`` and must not be
        assigned to an agent working only in ``xadupre/onnx-light``.
    * - 4. Native completion
      - #4618--#4623
      - `#4618 <https://github.com/xadupre/onnx-light/issues/4618>`_ ->
        `#4619 <https://github.com/xadupre/onnx-light/issues/4619>`_ ->
        `#4620 <https://github.com/xadupre/onnx-light/issues/4620>`_ ->
        `#4621 <https://github.com/xadupre/onnx-light/issues/4621>`_ ->
        `#4622 <https://github.com/xadupre/onnx-light/issues/4622>`_ ->
        `#4623 <https://github.com/xadupre/onnx-light/issues/4623>`_

.. toctree::
    :maxdepth: 1
    :caption: Fast-loading implementation sequence

    2026/2026-08_model_loading_bug_fixes
    2026/2026-08_prepared_execution
    2026/2026-08_onnxruntime_fast_model_loading
    2026/2026-08_native_fast_loading_completion

.. toctree::
    :maxdepth: 1
    :caption: Ready to implement

    2026/2026-08_parallel_for_profiling

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
    2026/2026-08_split_wheels

.. toctree::
    :maxdepth: 1
    :caption: Completed

    2025/2025-07_onnx_proto
    2026/2026-06_lib_onnx
    2026/2026-06_kernels_backend_tests
    2026/2026-06_gradient
    2026/2026-07_onnxruntime_onnx_light
    2026/2026-08_proto_binary_size
    2026/2026-08_processor_aware_kernel_tuning
    2026/2026-08_buffer_reuse_arena
    2026/2026-08_graph_builder_optimization
    2026/2026-08_session_execution_pools
