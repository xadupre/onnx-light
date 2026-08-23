
.. _l-next-steps:

Next Steps
==========

:Date: 2026-08


Recommended implementation order
--------------------------------

The roadmap has four objectives:

1. **Parallelize and tune kernels on multiple machines.** Define a safe default
   for every tunable kernel, then let each user calibrate that kernel and store
   the best parameters for their machine. See
   :ref:`l-next-steps-kernel-parallelization` and
   :ref:`l-next-steps-processor-aware-kernel-tuning`.
2. **Parallelize model startup.** Read and transform the model concurrently,
   then create kernels and prepack their weights through one memory-bounded
   execution plan. See :ref:`l-next-steps-fast-loading-sequence` and
   :ref:`l-next-steps-prepared-execution`.
3. **Manage persistent state.** Use ownership-aware arenas for KV caches and
   other state that survives one graph invocation, without copying or
   accidentally recycling live storage. See :ref:`l-next-steps-mutable-cache`
   and :ref:`l-next-steps-buffer-reuse-arena`.
4. **Integrate the completed startup path with ONNX Runtime.** Only after the
   native loader is complete, let ORT consume ``onnx-light`` payload ownership,
   prepared data, and direct-read contracts. See
   :ref:`l-next-steps-model-loading`.

The execution order is **1 -> 2 -> native loading completion -> 4**. Objective
3 branches from objective 2 because persistent state reuses its arena and
lifetime rules; it does not depend on the final ORT integration. Objective
numbers identify themes, not permission to start cross-repository work early.

All Next Steps
--------------

.. toctree::
    :maxdepth: 1
    :hidden:

    2026/2026-08_fast_loading_sequence
    2026/2026-08_parallel_for_profiling
    2026/2026-08_kernel_parallelization
    2026/2026-08_custom_types
    2026/2026-08_proto_inheritance
    2026/2026-08_quantization
    2026/2026-08_graph_builder_quantized_tensor
    2026/2026-08_graph_builder_authoring
    2026/2026-08_mutable_cache
    2026/2026-08_compiled_tensor
    2026/2026-08_model_resolution
    2026/2026-08_split_wheels
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

Use the search field to filter by status or text, and select a column heading
to sort the table.

``1``--``4`` refer to the four objectives above. ``Foundation`` supplies a
shared prerequisite. ``Independent`` is useful work but is not required by
these four objectives.

.. list-table::
    :header-rows: 1
    :widths: 12 12 28 48
    :class: sphinx-datatable

    * - Status
      - Objective
      - Plan
      - Contribution
    * - Started
      - 2
      - :ref:`l-next-steps-fast-loading-sequence`
      - Orders the four startup plans: bug fixes, prepared execution, native
        completion, then final ORT integration.
    * - Completed
      - 2
      - :ref:`l-next-steps-model-loading-bug-fixes`
      - Makes parsing, external data, and initializer materialization reliable
        before asynchronous work starts.
    * - Completed
      - 2
      - :ref:`l-next-steps-prepared-execution`
      - Provides the dependency graph, bounded scheduling, kernel creation, and
        prepacking needed by parallel startup.
    * - Planned
      - 2
      - :ref:`l-next-steps-native-fast-loading-completion`
      - Connects adaptive I/O, model resolution, prepared tensors, and
        first-token overlap before any new work in onnxruntime.
    * - Blocked
      - 4
      - :ref:`l-next-steps-model-loading`
      - Final cross-repository integration; blocked until every native loading
        issue through #4623 is closed.
    * - Completed
      - 1
      - :ref:`l-next-steps-parallel-for-profiling`
      - Measures work decomposition, utilization, and hardware counters so
        tuning decisions have evidence.
    * - Started
      - 1
      - :ref:`l-next-steps-kernel-parallelization`
      - Inventories kernel parallel coverage, publishes cross-machine
        baselines, then migrates measured kernel families through the tuning
        and calibration APIs.
    * - Discussed
      - 2, 3
      - :ref:`l-next-steps-custom-types`
      - Describes structured byte buffers used by packed weights and persistent
        state.
    * - Discussed
      - Foundation
      - :ref:`l-next-steps-proto-inheritance`
      - Reuses common schema fields without changing the flat wire format.
    * - Discussed
      - 1, 2, 3
      - :ref:`l-next-steps-quantization`
      - Defines quantized layouts consumed by kernels, prepared weights, and
        persistent cache pages.
    * - Discussed
      - 2, 3
      - :ref:`l-next-steps-graph-builder-quantized-tensor`
      - Preserves quantized initializers until preparation or persistent-state
        allocation.
    * - Discussed
      - Foundation
      - :ref:`l-next-steps-graph-builder-authoring`
      - Supplies reproducible models and workflows used to exercise the four
        objectives.
    * - Discussed
      - 3
      - :ref:`l-next-steps-mutable-cache`
      - Defines in-place KV-cache updates, aliasing, capacity, and persistence.
    * - Discussed
      - 2, 4
      - :ref:`l-next-steps-compiled-tensor`
      - Persists packed weights so startup and ORT integration can avoid
        repeated prepacking.
    * - Discussed
      - 2, 4
      - :ref:`l-next-steps-model-resolution`
      - Determines the final graph and live payloads before parallel reads or
        ORT handoff.
    * - Discussed
      - Foundation
      - :ref:`l-next-steps-split-wheels`
      - Packages runtime capabilities independently; it does not change their
        execution order.
    * - Completed
      - Foundation
      - :ref:`l-next-steps-onnx-proto`
      - Supplies the protobuf-free model representation used by startup and ORT.
    * - Completed
      - Foundation
      - :ref:`l-next-steps-lib-onnx`
      - Supplies ONNX validation and transformation without ``libprotobuf``.
    * - Completed
      - 1
      - :ref:`l-next-steps-kernels-backend-tests`
      - Provides the native kernels and correctness corpus required before
        parallel variants are accepted.
    * - Completed
      - Independent
      - :ref:`l-next-steps-gradient`
      - Supports training graphs but is not a prerequisite for the four
        objectives.
    * - Completed
      - 4
      - :ref:`l-next-steps-ort-onnx-light`
      - Establishes the existing ORT build-time integration that the optimized
        startup contract extends.
    * - Completed
      - 4
      - :ref:`l-next-steps-proto-binary-size`
      - Keeps the library embedded by ORT small.
    * - Completed
      - 1
      - :ref:`l-next-steps-processor-aware-kernel-tuning`
      - Provides schemas, defaults, calibration, user overrides, immutable
        snapshots, and persistent machine profiles.
    * - Completed
      - 2, 3
      - :ref:`l-next-steps-buffer-reuse-arena`
      - Supplies ownership-aware reusable storage for startup buffers and
        persistent state.
    * - Completed
      - 2, 4
      - :ref:`l-next-steps-graph-builder-optimization`
      - Finalizes graph rewrites before live payload selection and ORT handoff.
    * - Completed
      - 1, 2
      - :ref:`l-next-steps-session-execution-pools`
      - Supplies the shared executor used by parallel kernels and startup tasks.
