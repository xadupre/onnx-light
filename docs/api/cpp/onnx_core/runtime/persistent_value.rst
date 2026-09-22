persistent_value
================

``PersistentTensor`` owns a retained tensor value and its optional append capacity
by composition. Its move-only ``AppendLease`` reserves writable tails, grants at
most one in-place extension per invocation, and owns the contiguous growth policy.
``AppendReservation`` exposes only the new region: the kernel initializes it
directly, then seals the candidate with ``Commit``. Sealing does not publish
feedback state. Ordinary ``Tensor`` copies and borrowed views carry no such permission.
``PersistentValue`` preserves whole structured and encoded feedback values, using
``PersistentTensor`` for tensor leaves.

``PersistentStorageStatistics`` describes the allocations, copied bytes and reuse
reported by one storage event, independently of the consuming operator.
Reservations report this work only when a callback is supplied. The runtime
supplies that callback only when its existing ``events_enabled`` option is true;
reports then appear in ``RuntimeContext::events()`` as ``kPersistentStorage`` events.
There are no always-on counters or auditing mutexes. Explicit consumers can add
event payloads with ``operator+=`` to calculate totals.

.. doxygenfile:: onnx_core/runtime/persistent_value.h
   :project: onnx-light
