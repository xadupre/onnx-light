.. _l-next-steps-prepared-values-and-persistent-state:

Prepared values, custom representations, and persistent state
================================================================================

:Date: 2026-09

**planned**

Objective and consolidation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Define one runtime contract for reusable prepared weights and request-owned
persistent state, with a representation system that supports both declarative
quantization and fully custom structures. The first end-to-end consumer is
Qwen: shared packed projection weights plus independent mutable KV caches
for successive decode requests.

This roadmap replaces the independent implementation sequences in:

* :ref:`l-next-steps-custom-types`;
* :ref:`l-next-steps-quantization`;
* :ref:`l-next-steps-graph-builder-quantized-tensor`;
* :ref:`l-next-steps-compiled-tensor`;
* :ref:`l-next-steps-mutable-cache`.

Those pages remain design history and format examples. Where their proposals
conflict, this page is authoritative. In particular, specialized quantization
protos must not introduce a second payload/type hierarchy, and a compiled
cache entry is not the same thing as a quantized source value.

The completed prepared-execution, native fast-loading, allocator, and session
executor work remains the foundation. This plan extends their contracts; it
does not rebuild their schedulers or reopen their completed implementation
sequences. :ref:`l-next-steps-proto-inheritance` is independent and is not a
prerequisite.

Existing foundations and missing integration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The implementation already supplies:

* ``PreparedKey``, session/invocation ``TaskScope`` and explicit task
  dependencies in ``onnx_core/compute/prepared_task.h``;
* ``PreparedObjectStore``, immutable publication, generation tracking,
  consumer pins, residency budgets, eviction and materialization recipes in
  ``onnx_core/compute/prepared_execution.h``;
* ``PreparedTensorCache`` with digest, ISA, runtime, layout and format checks,
  diagnosed misses and atomic background persistence in
  ``onnx_core/compute/prepared_tensor_cache.h``;
* ordinary ``Tensor`` storage owners, borrowed views and allocation handles
  in ``onnx_core/runtime/memory/simple_tensor.h``.

These are reusable facilities, not yet a unified graph-visible structured
value system. The proposed ``StructTypeProto``, ``StructProto`` and
``CompiledTensorProto`` are not existing serialized contracts. Prepared
objects currently expose a raw-buffer view, while ``RuntimeSession`` retains
ordinary initializers and kernel instances. ``RuntimeContext::Clear`` clears
invocation values; it must not become the owner of persistent request state.

The new work connects typed representations and kernel preparation to these
facilities, then adds an explicit request lifetime and mutation contract.

Three independent decisions
+++++++++++++++++++++++++++

Keep logical meaning, physical representation and lifetime independent:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Axis
     - Examples
     - Contract
   * - Logical meaning
     - Dense tensor, affine quantization, codebook quantization, custom value
     - Describes what a consumer computes, including decoded type and shape
       when the value denotes a tensor.
   * - Physical representation
     - Dense bytes, blocked INT4 with scales, tiled FP32, custom records
     - Describes exact fields, buffers, bit layout, padding and format
       identity; it is not inferred from logical dtype alone.
   * - Lifetime and access
     - Immutable session prepack, mutable request cache, invocation workspace
     - Determines ownership, sharing, synchronization and release, not the
       numerical type.

A quantized value can use either a conventional block layout or a custom
structure. A compiled representation can be quantized or floating point.
A mutable state slot can contain a dense tensor or a structured value.
No inheritance chain can express these three independent choices cleanly.

Representation model: structured storage plus semantic profiles
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Use composition, with the following proposed responsibilities. Names and
wire field numbers are finalized in PR01; no ONNX-standard status is implied.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Descriptor
     - Responsibility
   * - ``StructTypeProto``
     - Describes a portable physical layout: fixed-width fields, nested
       records, arrays and bit packing. Counts are concrete after shape
       binding; references are acyclic and size arithmetic is checked.
   * - ``StructProto``
     - Carries a concrete type reference and its owned or external payload.
       The type determines the exact serialized byte ranges.
   * - ``TensorRepresentationProto``
     - Associates structured storage with logical tensor type/shape when
       applicable, a versioned format identity and optional semantic
       quantization metadata. A custom non-tensor value has no fabricated
       logical tensor type.
   * - ``QuantizationDescriptorProto``
     - Describes affine or codebook semantics using references to fields:
       codes, scales, zero points, axes, block sizes, rounding and decoding.
       It does not own another copy of the payload.
   * - ``CompiledTensorProto``
     - Wraps one prepared representation with all source dependencies,
       preparation recipe and compatibility requirements. It remains an
       optional derived cache, not a new graph-level numerical meaning.

``QuantizedTensorProto`` may be an authoring convenience for
``TensorRepresentationProto`` plus a quantization descriptor. It must not
be a separately maintained wire layout, allocation path or shape registry.

The representation supports three progressively specialized cases:

1. **Declarative quantization:** packed INT4 codes, block scales and zero
   points have a known affine profile. Generic tooling can inspect them and
   an explicit decoder can produce the logical tensor.
2. **Hybrid quantized/custom layout:** a named structure interleaves codes,
   scales, correction sums and tile padding. The same semantic profile
   references its fields; a kernel consumes that layout directly without
   materializing the decoded tensor.
3. **Fully custom representation:** a versioned domain/type identity describes
   records or opaque bytes with a registered consumer. A decoder is optional
   for a derived prepack with a portable source fallback. An authoritative
   custom graph input without a consumer or decoder fails explicitly.

For example, an INT4 matrix representation can contain an array of records
``{codes, scale, zero_point, compensation, padding}``. Quantization identifies
the first three fields and their logical block mapping; compensation and
padding remain physical implementation details. An FP32 packed matrix uses
the same storage system without any quantization descriptor.

A registered native C++ type can bind a descriptor to a typed view or create
an owned runtime object with auxiliary indexes. Serialized bytes are not a
dump of that C++ object: no pointers, vtables, native padding or process-local
handles go on the wire. Endianness, alignment, field offsets and destructors
remain explicit. Zero-copy typed access is allowed only when alignment,
lifetime and layout compatibility are proved.

Keep per-weight scales and zero points in value storage, not in the reusable
type catalogue. Only true format constants belong to the type. Registered
validation and decoding are explicit operations; merely loading a descriptor
must not execute arbitrary decoder code.

Prepacking: prepare once, share only when compatible
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A preparation request identifies the selected consumer contract and all
constant inputs it depends on, not just one ``source_name``. For example,
``MatMulNBits`` preparation may depend on packed weights, scales, zero points
and a bias. A multi-node fusion may depend on several initializers.

The canonical prepared key contains:

* ordered source roles and content identities, including type and shape;
* quantization descriptors and relevant operator attributes such as transpose,
  grouping, block size and any fused epilogue;
* representation format/version and kernel/prepacker ABI;
* required device capabilities, including exact ISA subsets and OS-enabled
  features, with alignment/layout requirements;
* any tuning choice that actually changes packed bytes or their interpretation.

Thread count or a node name is not a mandatory key component if it does not
change representation compatibility. Different consumers may share one
object only when their consumer contracts agree. One source may legitimately
have several prepared variants. Byte equality alone does not prove semantic
equivalence.

Reuse the existing lifecycle:

.. code-block:: text

    resolve consumer and source identities
        -> find compatible resident or persisted representation
        -> load packed bytes OR load source dependencies and prepare
        -> validate and atomically publish one immutable generation
        -> bind typed, pinned views to consumer kernels
        -> optionally persist, evict when unpinned, and reload

Concurrent requests for the same key share the existing in-flight generation.
Preparation is a session task; dynamic operands remain invocation work unless
the caller supplies an explicit immutable identity/version contract. Never
cache mutable inputs by pointer address.

Normal kernel execution uses its prepared binding without repeating layout
discovery, registry lookup or prepacking. Kernel construction/configuration
and asynchronous prepared dependencies must agree on the same binding. A
single scoped execution plan remains authoritative.

Disk-cache compatibility is resolved before selecting the payload manifest.
Skipping portable payload reads requires a trustworthy source content identity
already available from an immutable artifact manifest or prior validation.
Do not trust an unverified digest merely copied from a cache entry. If the
source identity cannot be established without reading it, perform that
validation and report its I/O cost instead of claiming a no-source-read hit.

The portable source remains recoverable; a compiled payload never replaces it
as the sole authoritative model value. Capability/ABI mismatch or stale
content is a diagnosed cache miss. Corrupt optional cache files may be
discarded with diagnostics and rebuilt, as the existing cache does. Invalid
authoritative model descriptors are load/checker errors. A rebuild failure
must propagate, not become a success-shaped fallback.

Persistent state: explicit request ownership
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Share representation and allocation descriptors with prepared values, but
never store mutable KV buffers in the immutable ``PreparedObjectStore``.

.. list-table::
   :header-rows: 1
   :widths: 23 39 38

   * - Lifetime
     - Owner and access
     - Examples
   * - Session
     - Immutable model and prepared store; shareable with consumer pins
     - Packed weights, read-only tables, kernel configuration
   * - Request
     - Explicit state handle; mutable under exclusive invocation access
     - KV data, valid lengths, positions, page tables
   * - Invocation
     - Submission-owned values and scratch; released after completion
     - Temporary attention buffers and ordinary outputs

The public API needs an explicit state object created from the prepared
session. Proposed lifecycle, not an existing API:

.. code-block:: text

    session = prepare(model)
    state_a = session.create_state(capacity)
    state_b = session.create_state(capacity)
    run(session, inputs_a, state_a)
    run(session, next_inputs_a, state_a)
    run(session, inputs_b, state_b)
    reset(state_a)
    close(state_a)

The session owns the immutable state specification; each request handle owns
its mutable allocations and metadata. A borrowed ``RuntimeContext`` binding
retains the handle for the complete asynchronous invocation, but clearing that
context does not reset the state. Existing stateless ``Run`` behavior stays
unchanged.

The state contract requires:

* independent requests share weights but never mutable cache storage;
* concurrent use of the same mutable state handle fails before execution;
  separate handles may execute concurrently;
* reset, resize and destruction cannot race with an active invocation;
* capacity, logical length, storage identity and allocated bytes are distinct;
  fixed-capacity overflow fails before writes;
* reset clears validity and positions without requiring allocation or a full
  payload clear; invalid entries must never be read;
* successful completion publishes new lengths only after all relevant writes
  and device events complete;
* cancellation or failure after mutation marks the state unusable until
  explicit reset or restore; do not promise rollback without a real journal;
* request buffers cannot be evicted as if they were reconstructible weights.

The initial implementation uses fixed-capacity contiguous KV storage. Paging,
quantized pages and explicit checkpoint/restore extend the same state model
later. Process-lifetime persistence does not imply automatic disk persistence.
State snapshots, if added, are opt-in, versioned and bound to model identity;
they are never written into the reusable weight cache.

Mutation and graph compatibility
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Mandatory mutation/aliasing is different from opportunistic last-use buffer
reuse. A kernel declares state reads/writes, affected regions and required
aliases; the execution plan orders conflicting accesses even when SSA names
alone provide no data dependency.

The first backend-neutral state slots are runtime bindings, not mutable
``TensorProto`` initializers and not hidden mutable members of shared kernel
instances. The graph and schema contracts must identify state effects at
function and subgraph boundaries. Existing ordinary ONNX graphs keep their
functional input/output semantics.

A state-aware rewrite of tensor ``past``/``present`` is opt-in and legal only
when external observations and ownership allow mutation. An ordinary fetched
``present`` tensor retains ordinary output lifetime semantics; returning a
live alias requires an explicit state-view API and cannot silently change a
previously returned tensor during the next decode step.

Required aliases preserve storage identity, offsets and strides. Immutable
initializers, unsafe shared writable bindings or unsupported views fail rather
than triggering a hidden full-cache copy. Dense KV append writes only new
tokens and committed metadata; Attention reads the valid prefix directly.

GraphBuilder, shape inference and serialization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ShapesContext`` stays the single source of truth for symbolic value types.
Extend it with the structured physical descriptor and optional logical view;
do not add an independent quantization registry in ``GraphBuilder``.
Known logical dimensions do not permit a tensor-only operator to consume
structured bytes implicitly: use an explicit decoder or a matching schema.

``GraphBuilder`` preserves structured source initializers, type references,
external payload ownership and quantization metadata through import/export,
functions and subgraphs. Deduplication considers semantic profiles as well
as bytes. Rewrites that change any preparation dependency invalidate the
corresponding compiled binding.

Prefer an optional companion compiled store for the first implementation.
Do not make the core plan depend on modifying upstream ONNX wire messages or
on proto inheritance. If model extensions are later serialized, document their
version and round-trip behavior explicitly; standard ONNX export must lower
to supported tensors/operators or report an unsupported export, never silently
drop authoritative structured values or state effects.

Implementation sequence
+++++++++++++++++++++++

All new steps are pending; completed foundations above are reused.

.. list-table::
   :header-rows: 1
   :widths: 8 27 48 17

   * - PR
     - Scope
     - Acceptance
     - Depends on
   * - PR01
     - Representation and lifetime contracts
     - Freeze identities, physical/semantic separation, native custom-type
       binding, request state effects and version rules. Fixtures cover
       dense prepack, affine INT4, custom records and mutable dense KV.
     - Existing runtime APIs
   * - PR02
     - Structured storage and quantization profiles
     - Checked layout/type resolution, buffer ownership and explicit decoding
       support declarative, hybrid and fully custom fixtures. Round-trips
       preserve metadata and reject malformed sizes and cyclic references.
     - PR01
   * - PR03
     - Typed preparation and kernel binding
     - Extend the existing object store and task bindings without a second
       scheduler. One FP32 pack and one hybrid INT4 pack share compatible
       objects, remain pinned during use, and never repack at each invocation.
     - PR02
   * - PR04
     - Persisted compiled representations
     - Reuse PreparedTensorCache with multi-source keys, compatibility,
       invalidation, atomic publication and explicit miss diagnostics.
       Warm hits skip preparation; no-source-read claims have verified
       lineage. Round-trip typed/native objects via versioned payloads.
     - PR03
   * - PR05
     - GraphBuilder and model-resolution integration
     - Structured initializers, logical/physical inference, scope-aware
       references, deduplication and prepared payload selection agree.
       Rewrites invalidate stale bindings; standard export never loses data.
     - PR02, PR03
   * - PR06
     - Request state and mutation planning
     - Introduce explicit state handles, persistent allocations, exclusive
       binding, effect dependencies, reset and failure semantics. Two
       requests share weights and remain isolated across repeated runs.
     - PR01; existing allocation/task infrastructure
   * - PR07
     - Contiguous KV and CPU consumer integration
     - CPU kernels consume the backend-neutral state API. Append touches
       only new tokens; decode performs no full-cache output allocation or
       copy. Verify dynamic lengths, capacity, cancellation and stateless
       compatibility against tensor past/present execution.
     - PR03, PR06; CPU backend integration
   * - PR08
     - End-to-end prepared/stateful acceptance
     - Measure cold/warm preparation, repeated decode and simultaneous
       independent requests. Report source/packed/state/scratch bytes,
       preparation counts and per-token copies; verify stale-cache handling,
       eviction pins and request reset/isolation.
     - PR04, PR05, PR07
   * - Later
     - Paging, quantized state and snapshots
     - Extend the accepted request contract without a second type system or
       implicit disk persistence; each feature has separate correctness,
       lifetime and memory gates.
     - PR08

PR06 can proceed in parallel with PR02-PR05 after PR01. Initial contiguous
state does not depend on every quantization format or GraphBuilder extension.
The declarative format catalogue is not a prerequisite for FP32 prepacking.

Ownership and acceptance
++++++++++++++++++++++++

``onnx-light`` owns the type/serialization contracts, prepared identities,
allocation and lifecycle, graph/schema integration, effect scheduling and
request state. ``onnx-light-cpu`` supplies its format validators, prepackers,
typed consumers, KV append and Attention implementation. It does not create
another persistent-state manager or private executor.

Acceptance uses C++ fixtures and existing runtime/backend test infrastructure.
Compare packed versus unpacked computation and stateful versus functional
tensor-cache execution with the same numerical contract. Test concurrent
preparation, active pins during eviction, changed scales with unchanged code
bytes, incompatible ISA/ABI, missing consumers, reset, invalid capacities,
failed mutations and independent requests.

Structural gates are explicit: a reused prepared object has no repeat
prepacking, a compatible verified disk hit does not read portable payloads,
and a fixed-capacity decode step neither allocates nor copies a full KV cache.
Publish latency, dispersion, peak/resident bytes and copy/read counters;
performance claims must distinguish source validation, preparation, inference
and state-management cost.
