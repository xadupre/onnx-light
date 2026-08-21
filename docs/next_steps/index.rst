
.. _l-next-steps:

Next Steps
==========

:Date: 2026-08


Recommended implementation order
--------------------------------

Runtime execution work should proceed in dependency order:

1. implement Profile PR01 from
   :ref:`l-next-steps-parallel-for-profiling`: first record a fixed maximum
   number of platform-independent profiling events; when profiling is disabled,
   the runtime must not collect data or perform any instrumentation work;
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

Large-model startup follows the four-plan sequence documented in
:ref:`l-next-steps-fast-loading-sequence`.

All Next Steps
--------------

.. toctree::
    :maxdepth: 1
    :hidden:

    2026/2026-08_fast_loading_sequence
    2026/2026-08_parallel_for_profiling
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

.. list-table::
    :header-rows: 1
    :widths: 12 24 32 32
    :class: sphinx-datatable

    * - Status
      - Next step
      - Planned work
      - Why
    * - Started
      - :ref:`l-next-steps-fast-loading-sequence`
      - Orchestrate the four-step large-model startup roadmap.
      - Define one dependency order for fast loading.
    * - Started
      - :ref:`l-next-steps-parallel-for-profiling`
      - Add bounded, opt-in ``ParallelFor`` diagnostics and hardware counters.
      - Explain CPU under-utilization before tuning prepared execution.
    * - Discussed
      - :ref:`l-next-steps-custom-types`
      - Define structured byte-buffer types for custom formats.
      - ``TypeProto.Opaque`` does not describe serialized layouts.
    * - Discussed
      - :ref:`l-next-steps-proto-inheritance`
      - Add schema inheritance while retaining flat wire encoding.
      - Reuse common fields without duplicating them across proto families.
    * - Discussed
      - :ref:`l-next-steps-quantization`
      - Describe quantized data families and their proto mappings.
      - Represent quantization consistently as structured custom types.
    * - Discussed
      - :ref:`l-next-steps-graph-builder-quantized-tensor`
      - Preserve quantized initializers in graph storage.
      - Avoid implicit dequantization or rewriting by ``GraphBuilder``.
    * - Discussed
      - :ref:`l-next-steps-graph-builder-authoring`
      - Add compact graph authoring and non-gallery runtime walkthroughs.
      - Make models easier to build, inspect, optimize, and execute.
    * - Discussed
      - :ref:`l-next-steps-mutable-cache`
      - Support in-place KV-cache updates with controlled aliasing.
      - Avoid duplicating large caches on every update.
    * - Discussed
      - :ref:`l-next-steps-compiled-tensor`
      - Persist packed tensor representations as caches.
      - Avoid repeating expensive prepacking when a model is reloaded.
    * - Discussed
      - :ref:`l-next-steps-model-resolution`
      - Resolve the final graph and required payloads before I/O.
      - Load weights only after transformations and liveness analysis.
    * - Discussed
      - :ref:`l-next-steps-split-wheels`
      - Split public features into composable Python wheels.
      - Let users install only the components they need.
    * - Completed
      - :ref:`l-next-steps-onnx-proto`
      - Build the protobuf-free ONNX message layer.
      - Provide the project's independent base schema layer.
    * - Completed
      - :ref:`l-next-steps-lib-onnx`
      - Port the ONNX C++ library to ``onnx_proto``.
      - Run the upstream library without ``libprotobuf``.
    * - Completed
      - :ref:`l-next-steps-kernels-backend-tests`
      - Provide native kernels and backend tests in C++.
      - Validate the runtime natively without depending on Python.
    * - Completed
      - :ref:`l-next-steps-gradient`
      - Generate backward-pass graphs symbolically.
      - Support training with a native graph-based gradient pass.
    * - Completed
      - :ref:`l-next-steps-ort-onnx-light`
      - Route onnxruntime protobuf usage through ``onnx-light``.
      - Provide a build-time alternative to protobuf in onnxruntime.
    * - Completed
      - :ref:`l-next-steps-proto-binary-size`
      - Reduce the ``lib_onnx_proto`` shared-library footprint.
      - Avoid shipping unused wrapper overhead.
    * - Completed
      - :ref:`l-next-steps-processor-aware-kernel-tuning`
      - Make kernel thresholds processor-specific and persistent.
      - Adapt thresholds to hardware instead of fixed constants.
    * - Completed
      - :ref:`l-next-steps-buffer-reuse-arena`
      - Reuse execution and I/O buffers safely.
      - Reduce allocations without breaking NumPy ownership.
    * - Completed
      - :ref:`l-next-steps-graph-builder-optimization`
      - Rewrite local graph patterns to cheaper equivalents.
      - Add the native optimization engine missing from ``GraphBuilder``.
    * - Completed
      - :ref:`l-next-steps-session-execution-pools`
      - Manage CPU policies and shared executor pools.
      - Give sessions deterministic, shareable execution resources.
