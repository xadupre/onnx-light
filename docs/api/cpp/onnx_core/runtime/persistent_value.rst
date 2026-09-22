persistent_value
================

``PersistentTensor`` owns a retained tensor value and its optional append capacity
by composition. Its move-only ``AppendLease`` grants one tail write for a specific
invocation; ordinary ``Tensor`` copies and borrowed views carry no such permission.
``PersistentValue`` preserves whole structured and encoded feedback values, using
``PersistentTensor`` for tensor leaves.

.. doxygenfile:: onnx_core/runtime/persistent_value.h
   :project: onnx-light
