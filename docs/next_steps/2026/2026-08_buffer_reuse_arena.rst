.. _l-next-steps-buffer-reuse-arena:

Buffer-reuse arenas
===================

:Date: 2026-08

**implementation complete**

The resulting arena design is described in :ref:`l-design-arenas`.

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

Progress
++++++++

The implementation is split into focused pull requests:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Pull request
     - Step
     - Result
   * - `PR #4430 <https://github.com/xadupre/onnx-light/pull/4430>`_
     - Allocator-output lifetime characterization
     - Moves this plan into active implementation and adds two expected-failure
       tests covering ``RuntimeContext::Clear`` and a subsequent run while an
       older allocator-backed NumPy output remains alive.
   * - `PR #4431 <https://github.com/xadupre/onnx-light/pull/4431>`_
     - Movable allocation handle
     - Gives allocator-backed tensors move-only ownership that returns each
       allocation exactly once and fixes the output lifetime tests from step 1.
   * - `PR #4436 <https://github.com/xadupre/onnx-light/pull/4436>`_
     - ``ExecutionArena``
     - Adds capacity-preserving, best-fit reuse for execution buffers, live and
       retained accounting, runtime integration, and Python access.
   * - `PR #4444 <https://github.com/xadupre/onnx-light/pull/4444>`_
     - ``IOArena``
     - Adds the second arena: capacity-preserving reuse for I/O buffers plus a
       reference-counted ``IOLease`` that pins an exported allocation and keeps
       the arena alive, making the allocation handle suitable for ownership by a
       NumPy capsule.
   * - `PR #4447 <https://github.com/xadupre/onnx-light/pull/4447>`_
     - Output allocation routing
     - Gives :cpp:class:`RuntimeContext` a dedicated I/O allocator alongside its
       execution allocator and has :cpp:class:`RuntimeSession` route a node's
       kernel invocation through it whenever that node produces a declared
       graph output, falling back to the execution allocator otherwise.
   * - `PR #4454 <https://github.com/xadupre/onnx-light/pull/4454>`_
     - Self-owning exported allocation handle
     - Lets an ``IOArena`` export a live buffer as an ``AllocationHandle`` backed
       by an ``IOLease`` (``ExportHandle``). The handle keeps its arena alive on
       its own and returns the buffer exactly once, so an exported graph output
       can be transferred to a NumPy capsule without keeping the mutable
       :cpp:class:`RuntimeContext` as the data owner.
   * - `PR #4457 <https://github.com/xadupre/onnx-light/pull/4457>`_
     - Capsule ownership wiring for exported outputs
     - Updates NumPy export so allocator-backed graph outputs transfer their
       released ``AllocationHandle`` into an ``IOArena`` lease-backed handle and
       store that handle directly in the NumPy capsule, removing the dependency
       on keeping the mutable :cpp:class:`RuntimeContext` alive as the data owner.
   * - `PR #4465 <https://github.com/xadupre/onnx-light/pull/4465>`_
     - Arena trimming
     - Adds ``Trim`` to both :cpp:class:`ExecutionArena` and
       :cpp:class:`IOArena`. It releases the storage held on the retained free
       lists, returns the freed slots to the unused pool, and reports the number
       of bytes released. Live and (for the I/O arena) leased buffers are left
       untouched, so trimming only gives back capacity that is currently idle.
   * - `PR #4469 <https://github.com/xadupre/onnx-light/pull/4469>`_
     - Retention caps and LRU eviction
     - Gives both arenas a per-arena retention cap that bounds the total
       capacity kept on the free lists. When freeing (or returning a lease)
       pushes the retained capacity above the cap, the arena evicts the
       least-recently-freed buffers until it fits again. ``SetRetentionCap``
       lowers the cap and evicts immediately; the cap defaults to unbounded so
       existing behaviour is unchanged. Live and leased buffers are never
       evicted.
   * - `PR #4480 <https://github.com/xadupre/onnx-light/pull/4480>`_
     - Python runtime activation
     - Exposes ``IOArena`` and its accounting controls in the Python runtime
       bindings, adds the ``io_allocator`` argument to ``RuntimeContext``, and
       makes ``ReferenceEvaluator`` create persistent execution and I/O arenas
       by default while still accepting caller-provided allocators.
   * - `PR #4493 <https://github.com/xadupre/onnx-light/pull/4493>`_
     - Output-slot routing
     - Completes step 5: :cpp:func:`RuntimeSession::VerifyOutputAllocators`
       resolves the allocation role of each output *slot* instead of the node as
       a whole, so a node that produces both a declared graph output and an
       intermediate keeps the declared output in the I/O arena and its
       intermediate in the execution arena.
   * - `PR #4497 <https://github.com/xadupre/onnx-light/pull/4497>`_
     - Slot-aware output allocation API
     - Adds :cpp:func:`RuntimeContext::AllocatorForOutput` and the slot-aware
       :cpp:func:`RuntimeContext::MakeOutputTensor` overload. Before each node's
       kernel runs, :cpp:class:`RuntimeSession` records which output slots carry
       a declared graph output, so a kernel can materialize each output directly
       in its final arena. A mixed-output node then needs no promotion copy: its
       declared outputs go to the I/O arena and its intermediates to the
       execution arena without the migration copy that
       :cpp:func:`RuntimeSession::VerifyOutputAllocators` would otherwise make.
   * - `PR #4506 <https://github.com/xadupre/onnx-light/pull/4506>`_
     - Slot-aware temporary allocation API
     - Adds :cpp:func:`RuntimeContext::MakeTemporaryTensor`, the
       ``AllocateTemporary`` half of the slot-aware allocation facade. It always
       allocates a kernel workspace from the execution arena, even while
       :cpp:class:`RuntimeSession` routes a declared-output node through the I/O
       allocator, so scratch buffers never enter the I/O arena's retention
       budget. Completes the "plus a kernel workspace" test coverage the plan
       requires alongside the mixed-output slot routing.
   * - `PR #4511 <https://github.com/xadupre/onnx-light/pull/4511>`_
     - Built-in kernel allocation migration
     - Converts built-in kernels to allocate each result through the slot-aware
       :cpp:func:`RuntimeContext::MakeOutputTensor` API and routes their
       scratch/workspace storage through the execution arena. Mixed-output
       kernels therefore materialize every result directly in its final arena
       without relying on post-hoc migration.

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
++++++

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

The kernel does not decide whether one of its outputs is final. That information
belongs to the graph/session layer.

Current implementation
----------------------

Kernels still allocate their outputs with code equivalent to:

.. code-block:: cpp

    Tensor y = MakeOutputTensor(dtype, shape, n_bytes,
                                rt != nullptr ? rt->allocator() : nullptr);

Here ``rt->allocator()`` is the allocator that
``RuntimeSession::Run`` selected *before invoking the kernel*.
``RuntimeSession::ProducesDeclaredOutput`` checks whether any name in
``node.output()`` is also present in ``GraphProto::output``. If so, it makes the
I/O allocator active for the entire kernel invocation; otherwise it makes the
execution allocator active. The kernel remains unaware of the graph-output
classification, but every allocation it performs during that invocation uses
the same selected arena.

This node-scoped selection is sufficient for a single-output node and keeps the
declared-output path zero-copy: a node that produces a declared output has the
I/O allocator active, so its declared output is materialized directly in the I/O
arena. It is not by itself the complete two-arena design, because a mixed-output
node would allocate *all* of its outputs from the I/O arena.

Output-slot routing closes that gap after the kernel runs.
``RuntimeSession::VerifyOutputAllocators`` now resolves the allocation role of
each output *slot* individually: a declared graph output belongs to the I/O
arena and every other (intermediate) output belongs to the execution arena. A
declared output produced by a node routed to the I/O allocator is already in the
right arena (no copy). Only the intermediates of a *mixed* node — a node that
produces at least one declared output alongside an intermediate — are migrated
back to the execution arena, so a rarely occurring mixed node no longer pins its
intermediates in the I/O arena. Nodes with a uniform role (all declared, all
intermediate) never trigger a migration.

A remaining limitation of output-slot routing alone is that migrating an
intermediate copies its bytes once. The slot-aware allocation API removes that
copy for kernels that adopt it: :cpp:func:`RuntimeContext::AllocatorForOutput`
resolves an output slot's arena from the per-slot roles
:cpp:class:`RuntimeSession` records before the kernel runs, and the slot-aware
:cpp:func:`RuntimeContext::MakeOutputTensor` overload allocates each output
directly in that arena. A kernel that produces its outputs through this overload
therefore writes each result straight into its final arena, so
:cpp:func:`RuntimeSession::VerifyOutputAllocators` finds every slot already in
place and performs no migration. Kernels that still use the node-scoped
``rt->allocator()`` path keep the previous behaviour: the runtime migrates a
mixed node's intermediates back to the execution arena with a single copy, and a
temporary workspace allocated through ``rt->allocator()`` uses the node-scoped
arena.

A kernel workspace has the opposite requirement to a declared output: it must
stay in the execution arena even when the node is routed to the I/O allocator.
:cpp:func:`RuntimeContext::MakeTemporaryTensor` is the ``AllocateTemporary`` half
of the facade: it always allocates from :cpp:func:`RuntimeContext::execution_allocator`
regardless of which allocator is currently active, so scratch buffers that a
declared-output kernel needs never enter the I/O arena's retention budget.
`PR #4511 <https://github.com/xadupre/onnx-light/pull/4511>`_ converts the
built-in kernels to the slot-aware
:cpp:func:`RuntimeContext::MakeOutputTensor` and
:cpp:func:`RuntimeContext::MakeTemporaryTensor` allocation paths.

Target output-slot contract
---------------------------

The complete design requires the following path:

1. When the session is built, it records the names declared by
   ``GraphProto::output``.
2. For every node output slot, the session compares
   ``node.output(slot)`` with that set and records an ``execution`` or ``I/O``
   allocation role in the execution plan.
3. During execution, the kernel requests storage for an output *slot*. It
   supplies the element type, shape and byte size, but not the lifetime role.
4. The runtime resolves the slot's precomputed role and calls either
   ``ExecutionArena::Allocate`` or ``IOArena::Allocate``.
5. The kernel writes directly into that buffer. No result is first allocated
   in the execution arena and then copied or promoted to the I/O arena.

Conceptually, the allocation path is:

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

The important API distinction is between asking for anonymous bytes and asking
for a node output. ``MakeOutputTensor(dtype, shape, bytes, allocator)`` alone
cannot make the decision because neither the allocator nor the kernel knows
which graph value the bytes will represent. The output-allocation API must
therefore carry a node/output-slot identity, for example through
``RuntimeContext::MakeOutputTensor(node, slot, dtype, shape, bytes)`` or an
equivalent pre-resolved output-allocation object. The kernel identifies the
slot it is producing; the runtime, not the kernel, translates that slot into an
arena.

This does require changing kernel allocation calls. A direct form is
``rt->MakeOutputTensor(slot, dtype, shape, bytes)``. A less intrusive form is
to pass an ``OutputAllocator`` facade into the kernel, with
``AllocateOutput(slot, ...)`` for results and ``AllocateTemporary(...)`` for
workspaces. Retaining the current undifferentiated ``rt->allocator()`` API
cannot implement correct mixed-output routing. Both halves of the direct facade
now exist: ``RuntimeContext::MakeOutputTensor(slot, ...)`` is
``AllocateOutput`` and ``RuntimeContext::MakeTemporaryTensor(...)`` is
``AllocateTemporary`` (it always allocates from the execution arena).

The migration can preserve the existing plain ``MakeOutputTensor`` overload for
standalone kernel calls and tests that explicitly supply an allocator. Runtime
dispatch, however, must use the slot-aware path. Tests must cover both mixed
orders (final/intermediate and intermediate/final), plus a kernel workspace,
and verify each allocation's owning arena.

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
   :cpp:func:`RuntimeContext::Clear` and after subsequent runs
   (`PR #4430 <https://github.com/xadupre/onnx-light/pull/4430>`_). The tests
   are expected failures until step 2 introduces independent allocation
   ownership. They assert allocator live counts before reading an older array,
   so the known dangling pointer is never dereferenced.
2. Introduce the movable allocation handle and use it for allocator-backed
   :cpp:class:`Tensor` storage (`PR #4431
   <https://github.com/xadupre/onnx-light/pull/4431>`_). This step removes both
   expected-failure markers from step 1.
3. Implement ``ExecutionArena`` with capacity-preserving, size-bucketed reuse
   for intermediates and temporary workspaces (`PR #4436
   <https://github.com/xadupre/onnx-light/pull/4436>`_).
4. Implement ``IOArena`` and make its allocation handle suitable for ownership
   by a NumPy capsule (`PR #4444
   <https://github.com/xadupre/onnx-light/pull/4444>`_). The arena reuses I/O
   buffers like ``ExecutionArena`` and exports each allocation through a
   reference-counted ``IOLease`` that pins the buffer and keeps the arena alive
   until the last external owner releases it.
5. Extend output allocation with an execution/I/O role and route declared graph
   outputs directly to the I/O arena. `PR #4447
   <https://github.com/xadupre/onnx-light/pull/4447>`_ adds the dedicated I/O
   allocator and the initial node-scoped routing: ``RuntimeSession::Run``
   switches the active allocator for a kernel invocation when the node produces
   a declared graph output. `PR #4493
   <https://github.com/xadupre/onnx-light/pull/4493>`_ completes this step with
   output-slot routing: ``RuntimeSession::VerifyOutputAllocators`` resolves the
   arena of each output slot individually, so a mixed-output node keeps its
   declared graph outputs in the I/O arena and its intermediate outputs in the
   execution arena. `PR #4497
   <https://github.com/xadupre/onnx-light/pull/4497>`_ adds the slot-aware
   allocation API that makes this zero-copy: ``RuntimeContext::AllocatorForOutput``
   and the slot-aware ``RuntimeContext::MakeOutputTensor`` overload let a kernel
   materialize each output directly in its final arena, so a mixed-output node
   produced through that overload needs no migration copy. `PR #4506
   <https://github.com/xadupre/onnx-light/pull/4506>`_ adds the complementary
   ``RuntimeContext::MakeTemporaryTensor`` workspace overload, which always
   allocates from the execution arena so a declared-output kernel's scratch
   buffers never enter the I/O arena. `PR #4511
   <https://github.com/xadupre/onnx-light/pull/4511>`_ completes the
   kernel-facing migration by assigning each built-in output allocation its
   ONNX output slot and routing every built-in workspace through the execution
   arena.
6. Transfer each exported output handle to its NumPy capsule; remove the
   dependency on keeping the mutable :cpp:class:`RuntimeContext` as the data
   owner. The enabling mechanism lands first (`PR #4454
   <https://github.com/xadupre/onnx-light/pull/4454>`_): ``IOArena::ExportHandle``
   turns a live buffer into an ``AllocationHandle`` backed by an ``IOLease``, so
   the handle keeps its arena alive on its own and can be owned by a capsule
   independently of the context. The NumPy capsule wiring then lands in `PR #4457
   <https://github.com/xadupre/onnx-light/pull/4457>`_.
7. Activate both arenas through the Python runtime path (`PR #4480
   <https://github.com/xadupre/onnx-light/pull/4480>`_): ``IOArena`` and its
   accounting and retention controls are exposed in the Python bindings,
   ``RuntimeContext`` accepts an ``io_allocator`` argument, and
   ``ReferenceEvaluator`` creates or accepts persistent execution and I/O
   arenas. Python runs now exercise the routing and lease mechanisms from
   steps 5 and 6 end to end.
8. Add independent retention caps, LRU eviction, trimming, and accounting for
   both arenas. Trimming lands first (`PR #4465
   <https://github.com/xadupre/onnx-light/pull/4465>`_): ``ExecutionArena::Trim``
   and ``IOArena::Trim`` release every retained free buffer's storage, return the
   slots to the unused pool, and report the bytes released, without touching live
   or leased buffers. Retention caps and LRU eviction follow (`PR #4469
   <https://github.com/xadupre/onnx-light/pull/4469>`_): each arena bounds the
   total capacity kept on its free lists and, when a free (or lease return) would
   exceed the cap, evicts the least-recently-freed buffers until the retained
   capacity fits. ``SetRetentionCap`` lowers the cap and evicts immediately; the
   cap defaults to unbounded and live or leased buffers are never evicted.
9. Benchmark repeated large intermediate and large-output models separately.
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

Pull requests
+++++++++++++

* `PR #4430 <https://github.com/xadupre/onnx-light/pull/4430>`_: allocator-backed
  NumPy output lifetime characterization.
* `PR #4431 <https://github.com/xadupre/onnx-light/pull/4431>`_: movable
  allocation ownership for allocator-backed tensors.
* `PR #4436 <https://github.com/xadupre/onnx-light/pull/4436>`_:
  capacity-preserving ``ExecutionArena`` reuse.
* `PR #4444 <https://github.com/xadupre/onnx-light/pull/4444>`_: capacity-preserving
  ``IOArena`` reuse with a reference-counted ``IOLease`` for exported I/O buffers.
* `PR #4447 <https://github.com/xadupre/onnx-light/pull/4447>`_: routes declared
  graph outputs to a dedicated I/O allocator via an execution/I/O allocation role.
* `PR #4454 <https://github.com/xadupre/onnx-light/pull/4454>`_: adds
  ``IOArena::ExportHandle`` and an ``IOLease``-backed ``AllocationHandle`` so an
  exported output can outlive the :cpp:class:`RuntimeContext` that produced it.
* `PR #4457 <https://github.com/xadupre/onnx-light/pull/4457>`_: wires NumPy
  output export to store an ``IOLease``-backed ``AllocationHandle`` directly in
  the owner capsule instead of keeping :cpp:class:`RuntimeContext` alive.
* `PR #4465 <https://github.com/xadupre/onnx-light/pull/4465>`_: adds ``Trim`` to
  both arenas to release retained free-buffer storage on demand.
* `PR #4469 <https://github.com/xadupre/onnx-light/pull/4469>`_: adds a per-arena
  retention cap with least-recently-freed eviction to both arenas.
* `PR #4480 <https://github.com/xadupre/onnx-light/pull/4480>`_: activates the
  two-arena design in the Python runtime by exposing ``IOArena``, wiring
  ``RuntimeContext.io_allocator``, and giving ``ReferenceEvaluator`` persistent
  execution and I/O arenas.
* `PR #4493 <https://github.com/xadupre/onnx-light/pull/4493>`_: completes
  output-slot routing so ``RuntimeSession`` resolves each output slot's arena
  individually, keeping a mixed-output node's declared outputs in the I/O arena
  and its intermediates in the execution arena.
* `PR #4497 <https://github.com/xadupre/onnx-light/pull/4497>`_: adds the
  slot-aware output allocation API (``RuntimeContext::AllocatorForOutput`` and the
  slot-aware ``RuntimeContext::MakeOutputTensor`` overload) so a kernel
  materializes each output directly in its final arena, removing the migration
  copy for a mixed-output node.
* `PR #4506 <https://github.com/xadupre/onnx-light/pull/4506>`_: adds the
  slot-aware temporary allocation API
  (``RuntimeContext::MakeTemporaryTensor``) so a declared-output kernel allocates
  its scratch/workspace buffers from the execution arena, keeping them out of the
  I/O arena's retention budget.
* `PR #4511 <https://github.com/xadupre/onnx-light/pull/4511>`_: migrates
  built-in kernels to the slot-aware output and temporary allocation APIs so
  mixed-output kernels avoid migration copies and workspaces never consume the
  I/O arena retention budget.
