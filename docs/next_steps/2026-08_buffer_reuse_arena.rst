.. _l-next-steps-buffer-reuse-arena:

Buffer-reuse arenas
===================

:Date: 2026-08

**implementation in progress**

Objective
+++++++++

The objective is to remove repeated allocation and page-fault costs without
weakening the ownership guarantees of zero-copy NumPy outputs.

Two different buffer lifetimes must be handled:

* **execution buffers** hold intermediate node results. They can be reused as
  soon as the execution plan reaches their last use;
* **I/O buffers** cross the runtime boundary. In particular, an output exposed
  as a NumPy array cannot be reused until that array is destroyed.

These lifetimes require two arenas with separate ownership, retention policies,
and accounting. Treating both categories as one free list obscures when a
buffer is actually reusable and can lead either to dangling NumPy arrays or to
unnecessarily pinned execution memory.

Current behaviour
+++++++++++++++++

:cpp:class:`SimpleRawBufferAllocator` pools stable :cpp:struct:`RawBuffer`
slots, but it does not retain their byte storage:

.. code-block:: cpp

    void SimpleRawBufferAllocator::Free(RawBuffer *buf) {
      // ...
      buffers_[i] = RawBuffer{}; // releases the bytes
      // ...
    }

Consequently, an intermediate result released by the execution plan loses its
capacity, and the next similarly sized result allocates and materializes fresh
pages.

Allocator-backed Python outputs have a separate lifetime problem. Inline-owned
outputs are moved into a capsule, but allocator-backed outputs remain owned by
the :cpp:class:`RuntimeContext`; their NumPy arrays keep a reference to that
context. A later :cpp:func:`RuntimeContext::Clear` destroys the tensors and
returns their allocations even if an array from the previous run is still
alive. Keeping the context alive is therefore not sufficient: each exported
array must pin its own allocation independently of the mutable contents of the
context.

Inputs normally borrow NumPy storage and need no arena allocation. An input
requires I/O-owned storage only when it must be copied, converted, transferred
from another device, or supplied through an explicit preallocated-I/O API.

Cost model
++++++++++

The main cost is not copying the result. A large allocation commonly reserves
virtual address space first and materializes physical pages when kernels write
to it:

1. a kernel writes into a fresh page;
2. the CPU raises a minor page fault;
3. the operating system allocates and zeroes a physical page;
4. the page table is updated and execution resumes.

For 400 MB this represents roughly 100000 four-kilobyte pages. If freeing the
buffer causes the system allocator to unmap them, the same work is repeated on
the next run. Retaining free buffers in an arena keeps those pages available
for similarly sized allocations.

Design
+++++

Introduce two arenas behind a common allocation-handle abstraction:

``ExecutionArena``
  Allocates node intermediates and other run-local temporary results. The
  execution plan returns a buffer at its last use, after which the arena may
  immediately reuse it.

``IOArena``
  Allocates graph outputs and any owned input staging buffers. An output
  allocation remains live while Python, another API consumer, or an explicit
  I/O binding holds it. It returns to the I/O arena only when the last external
  owner releases it.

Both arenas may implement the existing :cpp:class:`RawBufferAllocator`
operations internally, but a bare ``RawBuffer *`` is not a sufficient
cross-boundary ownership token. Introduce a movable allocation handle that
contains:

* the buffer pointer;
* its owning arena;
* its logical size and retained capacity;
* an explicit operation for returning the allocation exactly once.

A :cpp:class:`Tensor` owns this handle while the value is internal. Moving a
tensor moves the handle. Destroying or replacing the tensor returns the handle
to its arena unless ownership has been transferred to an external consumer.

The arenas are session-level objects, not per-run objects. Their retained
storage therefore survives :cpp:func:`RuntimeContext::Clear` and repeated
calls to ``Run``. The I/O arena state must itself be reference-counted by
exported leases so that destroying the runtime before an older NumPy array does
not leave the capsule with a dangling arena pointer.

Allocation routing
++++++++++++++++++

The runtime must choose the arena from the value's role, not merely from the
operator that creates it:

* graph outputs are allocated from the I/O arena;
* intermediate node outputs are allocated from the execution arena;
* temporary kernel workspaces are allocated from the execution arena;
* borrowed inputs allocate nothing;
* copied or converted inputs are allocated from the I/O arena.

The execution plan already knows which value names are graph outputs. Extend
the output-allocation path so it accepts the output name or an allocation role,
rather than passing one undifferentiated allocator to every
:cpp:func:`MakeOutputTensor` call. This preserves zero-copy output creation:
the final operator writes directly into an I/O allocation, with no promotion
copy after execution.

Subgraphs and functions follow the same rule relative to their caller. Values
that remain internal use the child execution arena. A value crossing the child
boundary must be returned through an I/O-style handle or transferred into the
parent's appropriate arena without copying.

Export to NumPy
+++++++++++++++

Exporting an allocator-backed output transfers its allocation handle out of
the tensor and into the NumPy owner capsule:

.. code-block:: text

    IOArena allocation
          |
          v
    output Tensor --transfer--> NumPy capsule
                                      |
                                      v
                            return to IOArena on destruction

The capsule owns the allocation itself, not the whole
:cpp:class:`RuntimeContext`. Therefore:

* :cpp:func:`RuntimeContext::Clear` may remove the tensor entry without
  invalidating an older NumPy array;
* a subsequent run cannot overwrite a buffer still referenced by Python;
* destroying the array returns the buffer to the I/O arena for a later run;
* multiple arrays from different runs may coexist safely.

Inline-owned outputs may use the same capsule abstraction by adopting their
:cpp:type:`RawByteBuffer` into the I/O arena, or retain the existing
standalone capsule path when pooling them is not required.

Reuse policy
++++++++++++

Each arena maintains its own retained free lists:

* use bucketed capacities so allocation does not scan every free buffer;
* choose the smallest available bucket that satisfies the request;
* preserve capacity when resizing a reused buffer;
* allocate new storage only when no suitable free buffer exists;
* bound retained capacity independently for each arena;
* evict least-recently-used free buffers when a cap is exceeded;
* expose ``Trim`` / ``Shrink`` independently on both arenas.

Separate caps are important. A burst of externally retained outputs must not
evict useful execution buffers, and a large workspace spike must not consume
the memory budget intended for repeated outputs.

Performance requirements
+++++++++++++++++++++++++

The two-arena design must not add work proportional to tensor size. In a
steady-state workload with repeated shapes:

* allocating and freeing a buffer performs no system ``malloc`` / ``free``;
* returning a NumPy output performs no system deallocation while the I/O
  arena remains below its retention cap;
* exporting an output performs no payload copy;
* moving an allocation handle between a tensor and a capsule is O(1);
* allocation routing and free-list lookup are O(1) for a fixed set of size
  classes;
* pages materialized during warm-up remain available to later runs;
* arena metadata is allocated during arena growth or initialization, not for
  every tensor allocation.

The common path should therefore be:

.. code-block:: text

    warm-up: system allocation -> page materialization -> arena allocation
    later runs: retained buffer -> kernel write -> external lease -> retained buffer

System deallocation is reserved for explicit trimming, cap-driven eviction,
arena destruction after the last lease, or an allocation size that cannot be
retained.

Accounting
++++++++++

Report memory by arena and by state:

``LiveExecutionSize``
  Bytes currently owned by live intermediate results and workspaces.

``RetainedExecutionSize``
  Capacity of free buffers retained by the execution arena.

``LiveIOSize``
  Bytes owned by live graph outputs, exported arrays, and owned input staging
  buffers.

``RetainedIOSize``
  Capacity of free buffers retained by the I/O arena.

Peak counters should exist for both live categories. A combined process-level
view may be reported in addition, but retained capacity must not be presented
as live tensor memory.

Correctness invariants
++++++++++++++++++++++

The implementation must preserve the following invariants:

1. A buffer belongs to exactly one arena.
2. A live allocation is owned by exactly one tensor, binding, or external
   lease.
3. A buffer appears on a free list only after its last owner releases it.
4. Clearing a runtime context cannot invalidate an exported output.
5. A new run cannot reuse storage pinned by an output from an older run.
6. Borrowed input memory is never inserted into an arena free list.
7. Transferring an allocation between owners does not move or copy its bytes.

Implementation order
++++++++++++++++++++

1. Add tests demonstrating that a NumPy output remains valid after
   :cpp:func:`RuntimeContext::Clear` and after subsequent runs.
2. Introduce the movable allocation handle and use it for allocator-backed
   :cpp:class:`Tensor` storage.
3. Implement ``ExecutionArena`` with capacity-preserving, size-bucketed reuse
   for intermediates and temporary workspaces.
4. Implement ``IOArena`` and make its allocation handle suitable for ownership
   by a NumPy capsule.
5. Extend output allocation with an execution/I/O role and route declared graph
   outputs directly to the I/O arena.
6. Transfer each exported output handle to its NumPy capsule; remove the
   dependency on keeping the mutable :cpp:class:`RuntimeContext` as the data
   owner.
7. Add independent retention caps, LRU eviction, trimming, and accounting for
   both arenas.
8. Benchmark repeated large intermediate and large-output models separately.
   Confirm that later runs reuse materialized pages, that retained NumPy
   outputs remain unchanged, and that peak live-memory accounting remains
   accurate.

Benchmarks
++++++++++

At minimum, measure these scenarios:

* repeated runs where outputs are destroyed before the next run;
* repeated runs while every previous output remains alive;
* a model dominated by large intermediates but with a small output;
* alternating output shapes and sizes;
* explicit trimming after a large one-off run.

After one warm-up iteration with stable shapes, acceptance requires no
payload-sized copy, no system allocation or deallocation for arena-managed
buffers, and no new minor page faults attributable to rematerializing those
buffers. Holding an output from an older run may require one additional I/O
allocation, but it must not disturb execution-arena reuse.

Free buffers are reused only within their own lifetime domain, while buffers
still visible outside the runtime remain pinned and untouched.
