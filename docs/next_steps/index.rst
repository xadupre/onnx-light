.. _l-next-steps:

Next Steps
==========

:Date: 2026-08

.. toctree::
    :maxdepth: 1
    :hidden:

    2026/2026-08_kernel_parallelization
    2026/2026-08_onnxruntime_fast_model_loading
    2026/2026-08_custom_types
    2026/2026-08_proto_inheritance
    2026/2026-08_quantization
    2026/2026-08_graph_builder_quantized_tensor
    2026/2026-08_mutable_cache
    2026/2026-08_compiled_tensor
    2026/2026-08_model_resolution
    2026/2026-08_split_wheels
    2026/2026-08_fast_loading_sequence
    2026/2026-08_model_loading_bug_fixes
    2026/2026-08_prepared_execution
    2026/2026-08_native_fast_loading_completion
    2026/2026-08_parallel_for_profiling
    2026/2026-08_graph_builder_authoring
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

Started
-------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Remaining work
    * - :ref:`l-next-steps-kernel-parallelization`
      - Publish the ARM64 baseline and calibration reports, compare them with
        the x86-64 results, calibrate the remaining ``Gemm`` parameters, decide
        which values are safe portable defaults, and complete cross-platform
        acceptance and ORT attribution.
    * - :ref:`l-next-steps-model-loading`
      - Implement issue #4612 in an ONNX Runtime fork: retain mapped-payload
        owners in ``SessionState``, use direct reads for ineligible tensors,
        run the four-configuration benchmark, and submit the upstream PR. All
        native dependencies through #4623 are complete.

Completed
---------

2025
^^^^

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :ref:`l-next-steps-onnx-proto`
      - Supplies the protobuf-free model representation used by startup and
        ONNX Runtime.

2026
^^^^

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :ref:`l-next-steps-fast-loading-sequence`
      - Orders the four startup plans: bug fixes, prepared execution, native
        completion, then final ONNX Runtime integration.
    * - :ref:`l-next-steps-model-loading-bug-fixes`
      - Makes parsing, external data, and initializer materialization reliable
        before asynchronous work starts.
    * - :ref:`l-next-steps-prepared-execution`
      - Provides the dependency graph, bounded scheduling, kernel creation, and
        prepacking needed by parallel startup.
    * - :ref:`l-next-steps-native-fast-loading-completion`
      - Connects adaptive I/O, model resolution, prepared tensors, and
        first-token overlap before any new work in ONNX Runtime.
    * - :ref:`l-next-steps-parallel-for-profiling`
      - Measures work decomposition, utilization, and hardware counters so
        tuning decisions have evidence.
    * - :ref:`l-next-steps-graph-builder-authoring`
      - Supplies reproducible models and workflows used to exercise the
        roadmap.
    * - :ref:`l-next-steps-lib-onnx`
      - Supplies ONNX validation and transformation without ``libprotobuf``.
    * - :ref:`l-next-steps-kernels-backend-tests`
      - Provides the native kernels and correctness corpus required before
        parallel variants are accepted.
    * - :ref:`l-next-steps-gradient`
      - Supports training graphs independently from the runtime roadmap.
    * - :ref:`l-next-steps-ort-onnx-light`
      - Establishes the ONNX Runtime build-time integration extended by the
        optimized startup contract.
    * - :ref:`l-next-steps-proto-binary-size`
      - Keeps the library embedded by ONNX Runtime small.
    * - :ref:`l-next-steps-processor-aware-kernel-tuning`
      - Provides schemas, defaults, calibration, user overrides, immutable
        snapshots, and persistent machine profiles.
    * - :ref:`l-next-steps-buffer-reuse-arena`
      - Supplies ownership-aware reusable storage for startup buffers and
        persistent state.
    * - :ref:`l-next-steps-graph-builder-optimization`
      - Finalizes graph rewrites before live payload selection and ONNX Runtime
        handoff.
    * - :ref:`l-next-steps-session-execution-pools`
      - Supplies the shared executor used by parallel kernels and startup
        tasks.

Discussion
----------

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: sphinx-datatable

    * - Plan
      - Contribution
    * - :ref:`l-next-steps-custom-types`
      - Describes structured byte buffers used by packed weights and persistent
        state.
    * - :ref:`l-next-steps-proto-inheritance`
      - Reuses common schema fields without changing the flat wire format.
    * - :ref:`l-next-steps-quantization`
      - Defines quantized layouts consumed by kernels, prepared weights, and
        persistent cache pages.
    * - :ref:`l-next-steps-graph-builder-quantized-tensor`
      - Preserves quantized initializers until preparation or persistent-state
        allocation.
    * - :ref:`l-next-steps-mutable-cache`
      - Defines in-place KV-cache updates, aliasing, capacity, and persistence.
    * - :ref:`l-next-steps-compiled-tensor`
      - Persists packed weights so startup and ONNX Runtime integration can
        avoid repeated prepacking.
    * - :ref:`l-next-steps-model-resolution`
      - Determines the final graph and live payloads before parallel reads or
        ONNX Runtime handoff.
    * - :ref:`l-next-steps-split-wheels`
      - Packages runtime capabilities independently from their execution order.
