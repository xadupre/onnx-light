persistent_value
================

``PersistentTensor`` owns a retained tensor value and its optional append capacity
by composition. Its move-only ``AppendLease`` grants one tail write for a specific
invocation; ordinary ``Tensor`` copies and borrowed views carry no such permission.
``PersistentValue`` preserves whole structured and encoded feedback values, using
``PersistentTensor`` for tensor leaves.

``PersistentStorageStatistics`` describes reported allocations, copied bytes and
storage reuse, independently of the consuming operator. ``PersistentStorageCounters``
provides synchronized accumulation and snapshots shared by feedback invocations
and their child contexts. It belongs to the persistence layer, not to
``RuntimeContext`` or a particular kernel.

.. doxygenfile:: onnx_core/runtime/persistent_value.h
   :project: onnx-light
