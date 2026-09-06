.. _l-next-steps-prepared-values-and-persistent-state:

Prepared values, custom representations, and persistent state
================================================================================

:Date: 2026-09
:Updated: 2026-09-06

**planned**

Objective and consolidation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Define one runtime contract for reusable prepared weights and request-owned
persistent state, with a deliberately small set of built-in quantized forms
and generic structures for every other format. The trade-off is proto-library
size versus convenience for common formats, not a complete quantization
taxonomy. The first end-to-end consumer is
Qwen: shared packed projection weights plus independent mutable KV caches
for successive decode requests.

This roadmap replaces the independent implementation sequences in:

* :ref:`l-next-steps-custom-types`;
* :ref:`l-next-steps-quantization`;
* :ref:`l-next-steps-graph-builder-quantized-tensor`;
* :ref:`l-next-steps-compiled-tensor`;
* :ref:`l-next-steps-mutable-cache`.

Those pages remain design history and format examples. Where their proposals
conflict, this page is authoritative. In particular, only a small closed set
of common quantized forms gets specialized proto support; the format catalogue
does not become a proto hierarchy. A compiled cache entry is not the same
thing as a quantized source value.

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
value system. The proposed ``StructTypeProto`` and ``EncodedValueProto``
are not existing serialized contracts. Prepared
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

Representation model: a small quantized core plus generic structs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Use one ``EncodedValueProto`` container with a small set of built-in layouts and a
generic structured layout for the long tail. Quantized, prepacked and mutable
are not separate storage categories. Names and wire field numbers are
finalized in PR01; no ONNX-standard status is implied.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Descriptor
     - Responsibility
   * - ``StructTypeProto``
     - Describes one fixed-size physical element: fields, nested records,
       fixed arrays and bit packing. Internal counts are concrete,
       references are acyclic and size arithmetic is checked. The number of
       repeated elements belongs to the value, not this type.
   * - ``EncodedValueProto``
     - One value container with optional logical tensor type/shape, a layout
       choice, physical ``storage_shape`` and owned or external payload.
       Layout is either a small built-in dense/affine form or a concrete
       ``StructTypeProto`` reference. INT8 and blockwise INT4 are
       configurations, not distinct messages.
   * - Optional preparation metadata
     - Records source dependencies, preparation recipe and compatibility
       requirements for a derived value. These are metadata on the same
       container, not a second container owning another payload.

``EncodedValueProto`` replaces the separate ``StructProto``,
``QuantizedTensorProto`` and ``CompiledTensorProto`` value-container proposals;
it is not an additional wrapper around all three. Existing ordinary
``TensorProto`` values remain supported without migration. A common runtime
view adapts both existing tensors and encoded values, reusing allocation,
shape and external-data machinery.

``Encoded`` identifies a representation that needs a layout-aware
interpretation, rather than merely indicating that bytes are stored.
``Value`` permits custom records containing multiple buffers as well as
logical tensors. The name applies equally to packed weights and KV blocks;
it does not imply immutability or disk persistence.

The affine layout parameters are a small nested descriptor, not a growing
``QuantizationDescriptorProto`` hierarchy. Source INT4 weights, their custom
prepacked form, and an INT4 KV block use the same container with different
layouts and lifetime bindings. Only derived prepacked values need preparation
provenance; authoritative request state is not a reconstructible weight cache.

The initial specialized subset is a proposal to freeze in PR01, not permission
to add all formats expressible by the catalogue. Additional built-in forms
require demonstrated common use and an explicit proto-size review.

The representation supports three cases:

1. **Common quantization:** INT8 per-tensor/per-axis and INT4 blockwise affine
   values select the common container's built-in affine layout.
2. **Other quantized formats:** codebooks, non-linear quantization, sparse
   outliers, mixed-bit blocks, rotations, and vendor-specific layouts use
   its structured layout plus a versioned format identity and an explicit decoder
   or registered consumer. Their schemas and numerical implementations live
   outside the proto library; adding one must not grow its message set.
3. **Fully custom structures:** arbitrary records use the same generic
   structure mechanism, with or without tensor semantics. A decoder is
   optional for a derived prepack with a portable source fallback. An
   authoritative custom graph input without a consumer or decoder fails
   explicitly.

For example, an INT4 matrix representation can contain an array of records
``{codes, scale, zero_point, compensation, padding}``. This custom packed
layout selects a ``StructTypeProto`` rather than adding a tile-specific quantized
message. Its registered format defines the logical block mapping and field
interpretation. An FP32 packed matrix also uses structures, without inventing
a quantization descriptor for non-quantized data.

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

One element type, many storage shapes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Follow the existing tensor distinction between element type and shape.
``StructTypeProto`` defines one encoded element, which can itself be a
fixed-size block. ``EncodedValueProto.storage_shape`` specifies how many
such elements are stored and their physical array dimensions. Changing that
shape does not instantiate or create a new type.

For example, the shared catalogue contains one block declaration:

.. code-block:: text

    ModelProto.struct_types[3] = StructTypeProto {
        name: "Int4Block"
        structure: {
            codes: INT4[32]
            scale: FLOAT
        }
    }

    EncodedValueProto {
        struct_type: { type_index: 3 }
        storage_shape: [128]
        logical_type: FLOAT[4096]
        raw_data: ...                 // 128 * 20 = 2560 bytes
    }

    EncodedValueProto {
        struct_type: { type_index: 3 }
        storage_shape: [256]
        logical_type: FLOAT[8192]
        raw_data: ...                 // 256 * 20 = 5120 bytes
    }

This is descriptive syntax, not the final wire schema. Each physical element
contains 16 bytes of INT4 codes followed by one 4-byte FLOAT scale, with no
implicit padding. The registered decoder defines the signed-code scaling and
mapping from blocks to logical elements. Scale values differ between blocks
and remain in the payload; the declaration only specifies their placement.

There are three distinct quantities:

* **element type:** the fixed physical layout of one ``Int4Block``;
* **storage shape:** the physical array of those blocks;
* **logical shape:** the dimensions exposed by the decoder or consuming
  kernel, not the number of physical records.

For the structured branch, require:

.. code-block:: text

    element_bytes = checked_size(resolved_struct_type)
    payload_bytes = checked_product(storage_shape) * element_bytes

The concrete element must be byte-aligned; explicit padding is part of its
type. Storage dimensions are non-negative concrete integers. An empty
``storage_shape`` means one scalar record; any zero dimension means zero
records. There is no ``-1`` dimension, inferred count from payload length or
automatic padding. Validate all dimensions and arithmetic before allocation
or access, then require the exact inline/external payload length.

The first version stores records densely in a documented row-major order.
Layouts requiring internal strides, tile padding or multiple fields express
them in the fixed element structure or a supported built-in layout, not in
an implicit reshape. Built-in dense/affine layouts have their own explicit
size rules, including parameter storage; do not apply the struct formula
blindly to them.

Logical dimensions are checked by the format/decoder contract and may differ
from ``storage_shape``. They never make a tensor-only operator accept encoded
bytes implicitly.

Shared catalogue, not template instantiations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Model values reference a declaration in ``ModelProto.struct_types`` via
``type_index``. The value's reference may select that declaration, while the
resolved declaration must be concrete. An inline declaration remains useful
for standalone values; import/export should share equivalent declarations
without merging different decoding semantics.

Resolve and validate each type once in its catalogue scope. Values with
different storage shapes share that resolved type; do not create a cache of
``(type, template_arguments)`` instantiations. They retain their own
shape/extent checks and payload owners.

Dynamic KV values use a session-owned catalogue with stable resolved type
handles. The model catalogue is read-only; additional session types are
interned without mutating it or duplicating a declaration for every page.
Indices are catalogue-local, not process-global identities. Export of a
session-created value includes or remaps its referenced declarations;
an index from another catalogue cannot be consumed without resolution.

Do not add generic template parameters, argument lists or an expression
language to the initial proto contract. Fixed arrays inside a record remain
part of its element type. If a later use case requires variable dimensions
inside the record, evaluate reuse of ONNX symbolic dimensions separately,
with explicit binding and size rules; it is not implicit support in PR02.

Proto-library size gate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PR01 records the existing minimal proto-library binary size, dependencies and
exported symbols with one reproducible Release configuration, and fixes the
allowed size increase before implementation. PR02 reports the delta under
identical compiler, linker, stripping and build settings.

The proto target contains only the selected compact messages, serialization
and structural machinery. Format-specific validators, decoders, prepackers,
catalogue data and registration tables belong to optional compute/runtime
components, not transitive dependencies of the proto library.

A codebook or mixed-bit fixture must round-trip through structures without
adding a specialized proto message, parser branch or enum entry for its
format. Existing ONNX scalar types are reused rather than duplicated.
Exceeding the agreed binary-size budget requires reducing the built-in
subset or an explicit design decision, not silently increasing the budget.

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

The first runtime increment uses fixed-capacity contiguous KV storage, but
the representation contract includes heterogeneous quantized blocks from PR01.
Paged mixed-format execution is a required next increment, not a redesign
deferred until after acceptance.
Process-lifetime persistence does not imply automatic disk persistence.
State snapshots, if added, are opt-in, versioned and bound to model identity;
they are never written into the reusable weight cache.

Paged KV with independently quantized blocks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The request owns a logical block table. Each K or V block is an instance of
the same ``EncodedValueProto`` representation used for source weights and prepacking.
The table does not impose one global physical dtype or format. Different
layers, heads, token ranges, and K/V blocks can choose different layouts.

.. code-block:: text

    request state
      layer 0, head group 0:
        tokens [0, 128):
          K -> EncodedValueProto {affine INT4, own scales and zero points}
          V -> EncodedValueProto {affine INT8, own scales and zero points}
        tokens [128, 256):
          K -> EncodedValueProto {structured custom format A}
          V -> EncodedValueProto {affine INT4, own scales and zero points}
        tokens [256, 384), valid length 17:
          K -> EncodedValueProto {dense FP16}
          V -> EncodedValueProto {dense FP16}

This example allocates capacity for the last block but exposes only 17 valid
tokens. Logical page size, quantization group size and physical allocation
size are independent: a page may contain several quantization groups.
Finer-grained mixed layouts can be represented by multiple logical blocks or
an explicit structured layout, not by adding a proto per combination.

Each block descriptor carries its logical range, tensor geometry, valid
length, concrete layout/type, payload owner and byte extent. Scales, zero
points, group axes and sizes belong to that block's representation, not a
global cache descriptor. K and V formats may differ, but their logical token
ranges must agree. For a given layer/head mapping, the decoder output types
and geometry must satisfy the shared Attention contract even when physical
layouts differ.

For a structured block, ``storage_shape`` determines the allocated physical
records, not the current valid token count. Pages with different capacities
can share the same element type while carrying different storage shapes.
Appending a token changes request validity and data, not the type catalogue.
Mapping quantization groups and padding to valid tokens remains an explicit
layout/Attention contract.

The block table is runtime state using existing container machinery or a
structured descriptor, not a new ``QuantizedKVCacheProto`` or a separate
paged quantization hierarchy. It retains owners/generations, not serialized
raw pointers. A block identity includes its request and logical position;
mutable blocks are never deduplicated by the prepared-weight cache.

Attention obtains a block iterator with logical ranges, valid extents and
resolved layout-specific readers. It dispatches once per block or tile to
dequantize/consume bounded data and carries the same online-softmax state
across blocks. It must not flatten, concatenate or fully dequantize the cache
before execution. Masks and positions use logical token indices, not physical
page offsets. Unsupported layouts are rejected before state mutation or
partial execution unless an explicitly selected bounded decoder is available.

Appending to the current block requires an explicit quantization policy:

* freeze the active group's scale/zero point and apply its documented rounding
  and saturation rules; or
* keep an active dense block and quantize it when sealed; or
* recompute parameters and requantize the affected group/block.

Changing a scale while leaving previous codes untouched is invalid. Published
sealed blocks are unchanged by later appends unless an explicit conversion is
requested. Converting INT8 to INT4, or replacing a custom layout, may allocate
a replacement block and update its table entry after successful conversion;
it must not rebuild the complete cache. Existing readers retain the old owner
until completion, and the temporary replacement counts against the request
memory budget. Failed conversion before publication leaves the old block
valid; failed in-place mutation follows the request invalidation contract.

Validation must cover mixed INT4/INT8/custom/dense blocks, distinct K and V
formats, partial final groups, page boundaries, per-block parameter changes,
reset/isolation, masks and conversion failures. Compare Attention with the
reference computation over the same decoded quantized values; assess loss
against unquantized KV separately. Report append, conversion and decode
latency, allocated bytes and dequantization workspace per format.

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
physical storage shapes, external payload ownership and quantization metadata
through import/export, functions and subgraphs. Deduplication considers
semantic profiles as well as bytes and both physical and logical shapes.
Rewrites that change any preparation dependency invalidate the corresponding
compiled binding.

Prefer an optional companion compiled store for the first implementation.
Do not make the core plan depend on modifying upstream ONNX wire messages or
on proto inheritance. If model extensions are later serialized, document their
version and round-trip behavior explicitly; standard ONNX export must lower
to supported tensors/operators or report an unsupported export, never silently
drop authoritative structured values or state effects.

Implementation sequence
+++++++++++++++++++++++

All new steps are pending; completed foundations above are reused.

The first concrete implementation is the **structured representation**:
``StructTypeProto`` and the structured-layout branch of ``EncodedValueProto``.
PR01 first freezes their minimal contract and the proto-size budget. PR02
implements checked fields, arrays, bit packing, type references, payload
ownership, per-value storage shapes and serialization before adding the
small built-in affine subset.
Custom packed weights and heterogeneous KV-block fixtures must work through
structures without requiring a catalogue of native quantized types.

Use :ref:`l-next-steps-custom-types` as physical-layout design material, not
as a separate roadmap to implement verbatim: its old ``StructProto`` value
container is replaced by ``EncodedValueProto``. Typed prepacking, graph
integration and persistent-state consumers build on this common foundation.

.. list-table::
   :header-rows: 1
   :widths: 8 27 48 17

   * - PR
     - Scope
     - Acceptance
     - Depends on
   * - PR01
     - Representation and lifetime contracts
     - Freeze the small built-in affine subset, struct-based extension path,
       element-type/storage-shape separation, catalogue identities, native
       bindings and request state effects, including heterogeneous K/V block
       descriptors. Record the
       minimal proto size baseline and agree a size budget before PR02.
     - Existing runtime APIs
   * - PR02
     - Structs first, then minimal built-in layouts
     - Implement StructTypeProto and EncodedValueProto's structured branch
       first; then add common INT8/INT4 layouts. Round-trip custom records,
       codebook/mixed-bit formats and a heterogeneous KV-block fixture.
       Prove one type is shared by different storage shapes; check scalar,
       empty, overflow, catalogue resolution and exact payload-size cases.
       Report proto binary-size growth within the PR01 budget; no
       format-specific decoder is linked into the proto target.
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
   * - PR07a
     - Contiguous KV and CPU consumer integration
     - CPU kernels consume the backend-neutral state API. Append touches
       only new tokens; decode performs no full-cache output allocation or
       copy. Verify dynamic lengths, capacity, cancellation and stateless
       compatibility against tensor past/present execution.
     - PR03, PR06; CPU backend integration
   * - PR07b
     - Paged KV with heterogeneous quantization
     - The shared EncodedValueProto representation supports different K/V and
       per-block formats. Blockwise append/conversion and Attention preserve
       validity, numerical contracts and bounded workspace without copying
       or dequantizing the entire cache.
     - PR02, PR07a; CPU backend integration
   * - PR08
     - End-to-end prepared/stateful acceptance
     - Measure cold/warm preparation, repeated decode and simultaneous
       independent requests. Report source/packed/state/scratch bytes,
       preparation counts and per-token copies; verify stale-cache handling,
       eviction pins, request reset/isolation and the final proto-size budget.
     - PR04, PR05, PR07b
   * - Later
     - Snapshots and advanced page policies
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

Type/value tests also round-trip two encoded values with the same catalogue
reference but different storage shapes and payloads. Verify a single shared
resolved descriptor, scalar and zero-size storage, malformed lengths,
arithmetic overflow, external payload extents and session-to-model catalogue
remapping without copying type declarations per KV block.

Structural gates are explicit: a reused prepared object has no repeat
prepacking, a compatible verified disk hit does not read portable payloads,
and a fixed-capacity decode step neither allocates nor copies a full KV cache.
Publish latency, dispersion, peak/resident bytes and copy/read counters;
performance claims must distinguish source validation, preparation, inference
and state-management cost.
