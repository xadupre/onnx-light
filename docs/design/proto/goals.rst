Goals
=====

``onnx-light`` provides a protobuf-free, ONNX-compatible model representation
and a modular C++ runtime. The reference kernels, shape inference, graph
transformations, and backend tests now provide the foundation for four current
goals.

Tune and parallelize kernels
++++++++++++++++++++++++++++

Kernel implementations must scale across processors and machines without
hard-coded, machine-specific decisions. Parallel kernels share the session CPU
executor and expose thresholds, grain sizes, algorithms, and participant limits
through the processor-aware tuning API. Conservative compiled defaults keep
every kernel usable when no calibrated machine profile is available. See
:ref:`l-next-steps-kernel-parallelization` and
:ref:`l-next-steps-processor-aware-kernel-tuning`.

Parallelize model startup
+++++++++++++++++++++++++

Loading a large model must overlap independent parsing, reading, tensor
preparation, and kernel creation while keeping CPU, I/O, and memory use bounded.
The implementation first establishes reliable model resolution and ownership,
then schedules prepared execution and overlaps useful work with the first
inference. See :ref:`l-next-steps-fast-loading-sequence` and
:ref:`l-next-steps-prepared-execution`.

Integrate with ONNX Runtime
+++++++++++++++++++++++++++

ONNX Runtime can build against ``onnx-light`` instead of protobuf. The next
integration step is to carry the native ownership and prepared-payload contracts
into ONNX Runtime so mapped tensors, parallel loading, and first-token overlap
retain their benefits at the consumer boundary. See
:ref:`l-next-steps-ort-onnx-light` and :ref:`l-next-steps-model-loading`.

Persist reusable runtime state
++++++++++++++++++++++++++++++

Repeated sessions and inference steps should reuse expensive state instead of
copying or rebuilding it. This includes packed weights, bounded reusable arenas,
and mutable KV-caches with guaranteed in-place updates and explicit ownership.
Persistent state must remain compatible with model identity, processor
capabilities, and execution policy. See
:ref:`l-next-steps-buffer-reuse-arena` and
:ref:`l-next-steps-mutable-cache`.
