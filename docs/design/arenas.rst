.. _l-design-arenas:

Buffer-reuse arenas
===================

The runtime allocates tensor storage through two buffer-reuse arenas instead of
calling the system allocator for every result. The arenas remove repeated
allocation and page-fault costs without weakening the ownership guarantees of
zero-copy NumPy outputs. The detailed implementation plan and the pull requests
that delivered it are recorded in :ref:`l-next-steps-buffer-reuse-arena`.

Two buffer lifetimes
++++++++++++++++++++

A value produced during a run has one of two lifetimes, and each lifetime maps
to its own arena:

* **execution buffers** hold intermediate node results and kernel workspaces.
  They can be reused as soon as the execution plan reaches their last use;
* **I/O buffers** cross the runtime boundary. An output exposed as a NumPy array
  cannot be reused until that array is destroyed.

Treating both categories as one free list obscures when a buffer is actually
reusable and can lead either to dangling NumPy arrays or to unnecessarily pinned
execution memory. The two lifetimes therefore require separate ownership,
retention policies, and accounting.

The two arenas
++++++++++++++

:cpp:class:`ExecutionArena`
  Allocates node intermediates and other run-local temporary results. The
  execution plan returns a buffer at its last use, after which the arena may
  immediately reuse it.

:cpp:class:`IOArena`
  Allocates graph outputs and any owned input staging buffers. An output
  allocation remains live while Python, another API consumer, or an explicit
  I/O binding holds it. It returns to the I/O arena only when the last external
  owner releases it.

Both arenas are session-level objects, not per-run objects. Their retained
storage survives :cpp:func:`RuntimeContext::Clear` and repeated calls to
``Run``, so pages materialized during warm-up remain available to later runs.

Movable allocation handle
+++++++++++++++++++++++++

A bare ``RawBuffer *`` is not a sufficient cross-boundary ownership token, so the
arenas hand out an :cpp:class:`AllocationHandle` that carries:

* the buffer pointer;
* its owning arena;
* its logical size and retained capacity;
* an explicit operation for returning the allocation exactly once.

A :cpp:class:`Tensor` owns this handle while the value is internal. Moving a
tensor moves the handle. Destroying or replacing the tensor returns the handle
to its arena unless ownership has been transferred to an external consumer.

An I/O allocation exported for external ownership is pinned by a
reference-counted :cpp:class:`IOLease`. The lease keeps the buffer live and the
:cpp:class:`IOArena` alive on its own, so destroying the runtime before an older
NumPy array does not leave a capsule with a dangling arena pointer.

Allocation routing
++++++++++++++++++

The runtime chooses the arena from the value's role, not merely from the
operator that creates it:

* graph outputs are allocated from the I/O arena;
* intermediate node outputs are allocated from the execution arena;
* temporary kernel workspaces are allocated from the execution arena;
* borrowed inputs allocate nothing;
* copied or converted inputs are allocated from the I/O arena.

The kernel does not decide whether one of its outputs is final. That information
belongs to the graph/session layer. Before each node's kernel runs,
:cpp:class:`RuntimeSession` records the allocation role of each output *slot* by
comparing ``node.output(slot)`` with the names declared by
``GraphProto::output``. A kernel then requests storage for a slot through the
slot-aware :cpp:func:`RuntimeContext::MakeOutputTensor` overload, and
:cpp:func:`RuntimeContext::AllocatorForOutput` resolves that slot to its final
arena. A kernel workspace has the opposite requirement to a declared output: it
must stay in the execution arena even when the node is routed to the I/O
allocator, so :cpp:func:`RuntimeContext::MakeTemporaryTensor` always allocates
from :cpp:func:`RuntimeContext::execution_allocator`.

Because each output is materialized directly in its final arena, a mixed-output
node — one that produces at least one declared output alongside an intermediate
— needs no promotion copy: its declared outputs go to the I/O arena and its
intermediates to the execution arena without migration.

.. code-block:: text

    GraphProto::output names
             |
             v
    ExecutionPlan: (node, output slot) -> allocation role
             |
             v
    kernel asks for output slot N
             |
             +-- execution role --> ExecutionArena
             |
             `-- I/O role -------> IOArena

Subgraphs and functions follow the same rule relative to their caller. Values
that remain internal use the child execution arena; a value crossing the child
boundary is returned through an I/O-style handle or transferred into the
parent's appropriate arena without copying.

Export to NumPy
+++++++++++++++

Exporting an allocator-backed output transfers its allocation handle out of the
tensor and into the NumPy owner capsule. :cpp:func:`IOArena::ExportHandle` turns
a live buffer into an :cpp:class:`AllocationHandle` backed by an
:cpp:class:`IOLease`, so the capsule owns the allocation itself, not the whole
:cpp:class:`RuntimeContext`. Therefore:

* :cpp:func:`RuntimeContext::Clear` may remove the tensor entry without
  invalidating an older NumPy array;
* a subsequent run cannot overwrite a buffer still referenced by Python;
* destroying the array returns the buffer to the I/O arena for a later run;
* multiple arrays from different runs may coexist safely.

Reuse and retention policy
++++++++++++++++++++++++++

Each arena maintains its own retained free lists:

* it uses bucketed capacities so allocation does not scan every free buffer;
* it chooses the smallest available bucket that satisfies the request;
* it preserves capacity when resizing a reused buffer;
* it allocates new storage only when no suitable free buffer exists;
* it bounds retained capacity through a per-arena retention cap and evicts the
  least-recently-freed buffers when the cap is exceeded;
* it exposes ``Trim`` to release retained free-buffer storage on demand.

Separate caps matter: a burst of externally retained outputs must not evict
useful execution buffers, and a large workspace spike must not consume the
memory budget intended for repeated outputs. Live and leased buffers are never
evicted, so trimming and eviction only give back capacity that is currently
idle.

Accounting
++++++++++

Memory is reported by arena and by state:

``LiveExecutionSize``
  Bytes currently owned by live intermediate results and workspaces.

``RetainedExecutionSize``
  Capacity of free buffers retained by the execution arena.

``LiveIOSize``
  Bytes owned by live graph outputs, exported arrays, and owned input staging
  buffers.

``RetainedIOSize``
  Capacity of free buffers retained by the I/O arena.

Peak counters exist for both live categories. Retained capacity is never
presented as live tensor memory.

Correctness invariants
++++++++++++++++++++++

The design preserves the following invariants:

1. A buffer belongs to exactly one arena.
2. A live allocation is owned by exactly one tensor, binding, or external lease.
3. A buffer appears on a free list only after its last owner releases it.
4. Clearing a runtime context cannot invalidate an exported output.
5. A new run cannot reuse storage pinned by an output from an older run.
6. Borrowed input memory is never inserted into an arena free list.
7. Transferring an allocation between owners does not move or copy its bytes.
