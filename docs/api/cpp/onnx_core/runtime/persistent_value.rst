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

Reservations record allocations, copied bytes and reuse directly through the
optional ``RuntimeContext`` supplied to ``Reserve`` when its existing
``events_enabled`` option is true. Reports appear in the shared
``RuntimeContext::events()`` log as ``kPersistentStorage`` events.
Their ``storage_*`` fields describe the work for that event, independently of
the consuming operator. There is no separate statistics object or cumulative
counter state; consumers sum fields from the event list when totals are needed.

.. doxygenfile:: onnx_core/runtime/persistent_value.h
   :project: onnx-light
