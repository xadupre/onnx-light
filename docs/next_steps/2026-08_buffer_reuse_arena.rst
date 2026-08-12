.. _l-next-steps-buffer-reuse-arena:

Buffer-reuse arena allocator
============================

:Date: 2026-08

**discussion**

Objective
+++++++++

The objective is to remove the per-run allocation cost onnx-light pays for
large outputs. When a model is run repeatedly on similarly shaped inputs, the
result buffers should reuse already-materialized memory pages instead of being
returned to the system allocator and re-faulted on the next run.

Problem
+++++++

Both onnx-light and ONNX Runtime expose their outputs to NumPy without copying
the payload. ONNX Runtime builds the ``numpy.ndarray`` directly over
``tensor.DataRaw()`` and attaches a capsule that keeps the ``OrtValue`` alive,
so only the small handle is copied, not the underlying bytes. onnx-light is
zero-copy as well: :cpp:func:`RawBuffer::release` hands the owned
:cpp:type:`RawByteBuffer` to a capsule so the NumPy array keeps the storage
alive. A copy happens only for a non-owning CPU tensor, a GPU→CPU transfer, or
strings.

The difference is what happens to the buffer once the output array is
destroyed:

* ONNX Runtime returns the freed buffer to its **arena**, which keeps the
  pages mapped and materialized and recycles them on the next run.
* onnx-light returns the buffer to the **system allocator**. The next run
  re-allocates and re-faults every page.

Concretely, :cpp:class:`SimpleRawBufferAllocator` already pools ``RawBuffer``
*slots* (O(1) allocate/free, stable slot addresses), but on
:cpp:func:`SimpleRawBufferAllocator::Free` it assigns ``RawBuffer{}`` to the
slot, which destroys the backing ``std::vector`` and releases its bytes to the
system allocator. The next :cpp:func:`SimpleRawBufferAllocator::Allocate`
calls ``resize(n_bytes)`` on the now-empty vector, forcing a fresh allocation.

.. code-block:: cpp

    void SimpleRawBufferAllocator::Free(RawBuffer *buf) {
      // ...
      buffers_[i] = RawBuffer{}; // releases the bytes to the system allocator
      // ...
    }

Cost model
++++++++++

The cost is not a data copy. The ``malloc`` / ``free`` calls themselves are
close to fixed, but memory *materialization* is proportional to the touched
size. ``malloc(400 MB)`` mostly reserves a virtual-address range; physical
pages are attached lazily on first access (demand paging):

1. the operator kernel writes into a fresh page;
2. the CPU raises a minor page fault;
3. the OS kernel allocates a physical page and zeroes it;
4. the page table is updated and execution resumes.

For 400 MB this is roughly 100000 four-kilobyte pages faulted, attributed and
zeroed on every run. An arena keeps those pages mapped after a logical free and
recycles them; without an arena a large allocation can be handed back to the OS
with ``munmap``, so the next run repeats the whole cycle.

For ``100M`` ``float32`` values (400 MB), onnx-light therefore pays the
allocation, the release, and the page faults on every iteration. In a
benchmark that does not keep the output alive, the buffer is released
immediately after each measurement and re-faulted at the next call.

Design
++++++

Add a **buffer-reuse arena** that recycles ``RawBuffer`` storage across
allocations instead of releasing it. The arena implements the existing
:cpp:class:`RawBufferAllocator` interface, so no call site changes.

Key ideas:

* On :cpp:func:`RawBufferAllocator::Free`, retain the buffer's capacity in a
  free list keyed by (bucketed) size rather than destroying the storage.
* On :cpp:func:`RawBufferAllocator::Allocate`, pop a retained buffer whose
  capacity is large enough and :cpp:func:`RawBuffer::resize` it (the
  :cpp:class:`DefaultInitAllocator` skips zero-filling, so no ``memset`` is
  paid, and shrinking never releases capacity). Only allocate new storage when
  no reusable buffer is available.
* Because the reused buffer keeps its capacity, its pages stay mapped and
  materialized; the second and later runs avoid the page faults entirely.

Retention policy
++++++++++++++++

The arena trades resident memory for speed, so retention must be bounded:

* **Size bucketing.** Group retained buffers by rounded-up capacity (for
  example next power of two, or a fixed set of size classes) so a request finds
  a reusable buffer in O(1) without scanning.
* **Capacity cap.** Bound the total retained bytes. When the cap is exceeded,
  release the least recently used retained buffers back to the system
  allocator. This keeps steady-state runs fast while letting a spike shrink.
* **Explicit trim.** Provide a ``Trim`` / ``Shrink`` entry point so a caller
  that is done with large runs can return retained pages to the OS.

Accounting
++++++++++

The existing counters keep their current meaning: ``TotalAllocatedSize`` and
``PeakAllocatedSize`` count only *live* (allocated, not-yet-freed) bytes so
peak-memory tests are unaffected. Retained-but-free capacity is reported
separately (for example ``RetainedSize``) so it does not inflate the peak and
so the arena's memory footprint remains observable.

Interaction with zero-copy output
++++++++++++++++++++++++++++++++++

When an output is exported to NumPy, :cpp:func:`RawBuffer::release` moves the
storage out of the arena-owned slot, so the arena must treat that buffer as
gone rather than retained — the pages now belong to the NumPy capsule and are
freed when the array dies, exactly as today. Only buffers freed *back to the
arena* (dead intermediate results, and outputs the caller did not keep) are
eligible for reuse. This preserves the invariant that a live output array
always pins its own storage, so re-running the session with different inputs
never overwrites a result the caller still holds.

Correctness note
++++++++++++++++

Reuse is safe only for buffers the arena still owns. A buffer whose bytes were
moved out (released to a capsule) or adopted from foreign storage must not be
placed on the free list. The arena keeps ownership identity per slot, mirroring
the existing ``index_map_`` check in
:cpp:func:`SimpleRawBufferAllocator::Free`.

Implementation order
++++++++++++++++++++

1. Introduce a ``ReusingRawBufferAllocator`` (or an opt-in mode on
   :cpp:class:`SimpleRawBufferAllocator`) that keeps freed storage instead of
   destroying it, with size-bucketed free lists.
2. Add a bounded retention policy (capacity cap + LRU eviction) and a ``Trim``
   entry point.
3. Report retained capacity separately from live/peak size.
4. Ensure released-to-capsule buffers are excluded from reuse.
5. Benchmark a repeated large-output model and confirm the second and later
   runs no longer pay per-iteration page faults while peak memory is unchanged.
