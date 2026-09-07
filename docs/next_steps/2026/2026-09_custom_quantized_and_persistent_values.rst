.. _l-next-steps-prepared-values-and-persistent-state:
.. _l-next-steps-custom-quantized-persistent-values:

Custom, quantized, and persistent values
================================================================================

:Date: 2026-09
:Updated: 2026-09-07

**planned**

Objective and consolidation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Cover three uses of values: custom structs, quantized representations, and
values retained between model calls. Prepared weights reuse the same
representations. Persistent state is built from an explicit mapping of model
outputs back to inputs, not a separate state type system. Qwen is the first
consumer: shared packed weights and per-request KV values carried between
decode calls.

This page contains the structured-type contract, quantization examples,
prepared-cache format, and output-to-input state binding in one
implementation sequence. The separate structured-types and mutable-cache
pages have been removed; their retained contracts and examples are integrated
here, not maintained as competing proposals.

:ref:`l-next-steps-quantization` and
:ref:`l-next-steps-graph-builder-quantized-tensor` remain format and authoring
design references, not independent implementation sequences. Where their
proposals conflict, this page is authoritative. Only a small closed set
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
facilities, then adds a small wrapper for feeding retained outputs into the
next model call.

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
     - Immutable session prepack, retained request cache, invocation workspace
     - Determines ownership, sharing, synchronization and release, not the
       numerical type.

A quantized value can use either a conventional block layout or a custom
structure. A compiled representation can be quantized or floating point.
A retained input can contain a dense tensor or a structured value.
No inheritance chain can express these three independent choices cleanly.

Representation model: a small quantized core plus generic structs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Use one ``StructTypeProto`` to describe structs, including structs containing
tensors, nested structs and sequences. Use ``EncodedValueProto`` for their
byte-encoded representations when a fixed physical layout exists, alongside
the small set of built-in layouts. A runtime struct does not need a second
type category called a composite. Quantized, prepacked and mutable are not
separate storage categories. Names and wire field numbers are finalized in
PR01; no ONNX-standard status is implied.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Descriptor
     - Responsibility
   * - ``StructTypeProto``
     - Describes fields and their types, nested structs, arrays and bit
       packing. A struct need not have a fixed physical size. The subset with
       statically sized inline fields also describes one byte-encoded
       element; only this subset uses checked size arithmetic and record
       counts derived from payload length.
   * - ``EncodedValueProto``
     - One value container with optional logical tensor type/shape, a layout
       choice and owned or external payload with a known byte extent.
       Layout is either a small built-in dense/affine form or a concrete
       ``StructTypeProto`` reference whose physical element size is known.
       A general struct with dynamic fields cannot be dumped into this byte
       payload. INT8 and blockwise INT4 are configurations, not distinct messages.
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
``Value`` permits custom records as well as logical tensors. A runtime struct
instance groups independently owned buffers through its fields, using the
same ``StructTypeProto`` rather than a parallel descriptor system. Those
buffers are not inline pointers in a serialized physical record.
The name applies equally to packed weights and KV blocks;
it does not imply immutability or disk persistence.

Keep the name ``EncodedValueProto`` for every value example and proposed
API. ``StructProto`` would suggest that every value must instantiate a
``StructTypeProto``, whereas built-in layouts need not do so. Reserve
``StructTypeProto`` for the reusable struct description, including its field
types and constants. Do not introduce a ``StructProto`` alias, base
class, nested value wrapper or parallel value category.

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

The examples below distinguish parameters stored
as constants in the type from parameters stored as scalar fields in each
payload. Outside the byte buffer does not mean outside the type: constant
scale and zero point are serialized once in the shared ``StructTypeProto``
declaration, while each value stores only codes. The decoder combines those
codes with the type's constants to expose a logical FLOAT tensor. All values
of that type share the same parameters; varying them without changing the
type requires the per-value payload form.

.. _l-next-steps-custom-types:

Struct types and byte-encodable layouts
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``StructTypeProto`` describes a struct's fields and types. A concrete
declaration selects ``array``, ``bit_packing`` or ``structure``. Each field
selects either a value type or a tensor constant in the declaration.
It may describe a cache with tensor and sequence fields, not just a packed
numeric record.

Fixed physical size is an eligibility condition for byte encoding, not a
requirement on every struct. The following wire sketch shows the struct type
and the byte-encoded value container; PR01 freezes field numbers:

.. code-block:: text

    message EncodedValueProto {
        oneof layout {
            StructTypeProto struct_type = <N>;  // exact reference or inline type
            // The small built-in layout alternatives are omitted here.
        }
        optional TypeProto logical_type = <N>;
        bytes raw_data = <N>;
        repeated StringStringEntryProto external_data = <N>;
        string name = <N>;
        string doc_string = <N>;
        // Optional preparation metadata is described below.
    }

    message StructTypeProto {
        message Structure {
            message Field {
                string name = 1;
                oneof content {
                    TypeProto type = 2;
                    TensorProto constant = 4;
                }
                string doc_string = 3;
            }
            repeated Field field = 1;
        }
        message BitPacking {
            message Component {
                string name = 1;
                uint32 bit_width = 2;
            }
            repeated Component component = 1;
            uint64 dimension = 2;
        }
        message Array {
            TypeProto element_type = 1;
            uint64 dimension = 2;
        }
        oneof kind {
            Array array = 1;
            Structure structure = 2;
            BitPacking bit_packing = 3;
            uint64 type_ref = 5;
        }
        optional FunctionProto decoder = 6;
        optional FunctionProto encoder = 7;
        string name = 8;
        string doc_string = 9;
        repeated StringStringEntryProto metadata_props = 10;
        optional uint64 type_id = 11;
    }

    message TypeProto {
        oneof value {
            // Existing alternatives remain unchanged.
            StructTypeProto struct_type = <N>;
        }
    }

    message ModelProto {
        repeated StructTypeProto struct_types = <N>;
    }

``EncodedValueProto.struct_type`` selects an exact ``type_ref`` or a concrete
inline declaration eligible for byte encoding. An exact type reference may
also appear inside ``TypeProto`` and nested fields, including references to
structs with dynamic fields. A ``StructTypeProto`` with its kind
unset is an unconstrained struct category, permitted only inside ``TypeProto``
for heterogeneous sequence/map element constraints, never as a payload layout.
Reference and category forms carry no declaration ID, decoder, encoder, name,
or declaration metadata of their own.

``Field.type`` and ``Array.element_type`` use ``TypeProto``: tensors, nested
structs, sequences, maps and optional values use their existing type
alternatives and validation rules. Tensor dimensions may be dynamic when
the consuming contract permits it. Array lengths and bit-packing counts
remain explicit concrete integers; a dynamic-length collection uses a sequence.
Opaque fields require an explicit supported native binding and lifecycle.

For byte encoding, every non-constant field must instead resolve recursively
to fixed-size inline data. Dynamic dimensions, sequences, maps, optional
values and opaque objects have no implicit inline size. Such a struct remains
a valid runtime type but cannot select the raw/external byte-payload layout.
Its runtime field values retain their ordinary owners and checked views.
There is no second composite type or descriptor to define; see
:ref:`l-next-steps-persistent-struct-state`.

``Field.constant`` is the actual ``TensorProto`` value, not a graph input or
a second value buffer. It must have concrete dimensions and matching data.
Only true shared format constants belong here; changing a constant changes
the type identity. Mutable lengths, positions and per-block quantization
parameters are instance data, not declaration constants.

Byte-encoding rules and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a byte-encodable struct, compute sizes recursively in bits with checked
``uint64`` arithmetic:

.. code-block:: text

    size(scalar(T))             = bit_width(T)
    size(tensor(T, dims))       = checked_product(dims) * bit_width(T)
    size(Array(T, n))           = n * size(T)
    size(BitPacking(c..., n))   = n * sum(c.bit_width)
    size(Field(constant))       = 0
    size(Field(T))              = size(T)
    size(Structure(f...))       = sum(size(f))
    size(type_ref=id)           = size(resolve_type_id(id))

Arrays and bit packings are tight, fields follow declaration order, and
padding is explicit. Bits run from least to most significant within a byte;
multi-byte values are little-endian. Only fixed-width ONNX scalar leaves are
physical data. No native alignment or process-local pointer representation is
implicit in this contract. Every field, array and bit-packing read is
bounds-checked; a typed view does not bypass alignment or byte-extent checks.

Type checking rejects duplicate field/component names, fields with both or
neither content alternatives, invalid field types/constants, zero component
widths, unresolved or cyclic references, and invalid array/bit-packing counts.
Byte-encoding validation additionally rejects fields without a fixed physical
size, unsupported physical leaves and overflowing size arithmetic. The encoded
root must have strictly positive size divisible by eight. Constant-only
structs are valid types but cannot be roots of this byte encoding; nested
constant-only groups may contribute zero bytes. Payload divisibility and
external bounds follow the next section.

For encoded values, only the decoder or encoder of the concrete root is invoked.
A nested ``type_ref`` contributes layout and constants; its decoder or encoder
is not composed implicitly. Loading a type must not execute decoder code.

One element type, many payload lengths
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In the byte-encodable subset, ``StructTypeProto`` defines one encoded element,
which can itself be a fixed-size block. ``EncodedValueProto`` stores a flat
sequence of such elements. The payload byte length and the resolved element byte size
determine the number of records; a different count does not instantiate or
create a new type. No physical shape or redundant record count is serialized.

For example, the shared catalogue contains one block declaration:

.. code-block:: text

    StructTypeProto {                 // declaration in ModelProto.struct_types
        type_id: 2001
        name: "Int4Block"
        structure: {
            codes: INT4[32]
            scale: FLOAT
        }
    }

    EncodedValueProto {
        struct_type: { type_ref: 2001 }
        logical_type: FLOAT[4096]
        raw_data: ...                 // 128 * 20 = 2560 bytes
    }

    EncodedValueProto {
        struct_type: { type_ref: 2001 }
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
* **payload byte length:** the extent of the flat sequence of blocks, from
  which their count is derived;
* **logical shape:** the dimensions exposed by the decoder or consuming
  kernel, not the number of physical records.

For the structured branch, require:

.. code-block:: text

    element_bytes = checked_size(resolved_struct_type)
    payload_bytes = raw_data.size()        // inline payload
    // Or external_data.length for a validated external payload extent.
    require(element_bytes > 0)
    require(payload_bytes % element_bytes == 0)
    element_count = payload_bytes / element_bytes

The concrete root element must have a strictly positive, byte-aligned size;
explicit padding is part of its type. Validate its fixed dimensions and size
arithmetic before allocation or access. The two buffers above therefore
contain ``2560 / 20 = 128`` and ``5120 / 20 = 256`` records of the same type.
An empty payload means zero records; exactly one element's byte size means
one record. Reject partial records rather than rounding their count.

Reject a zero-sized encoded root because its payload length cannot determine
its number of instances. Zero-sized nested structures, such as constant-only
field groups, remain allowed within a positive-sized root.

Inline length is already carried by ``raw_data``. External data must provide
an explicit length and a valid backing-file extent; do not infer the length
from the remainder of a file or accept conflicting payload sources.
Do not add a second serialized byte-count field.

The first version stores records densely in buffer order.
Layouts requiring internal strides, tile padding or multiple fields express
them in the fixed element structure or a supported built-in layout, not in
an implicit reshape. Built-in dense/affine layouts have their own explicit
size rules, including parameter storage; do not apply the struct formula
blindly to them.

Logical dimensions remain optional value information checked by the
format/decoder contract against the derived record count. A byte length does
not determine tensor rank or shape. Logical dimensions never make a
tensor-only operator accept encoded bytes implicitly.

Shared catalogue, not template instantiations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Concrete declarations in ``ModelProto.struct_types`` carry a nonzero
``uint64 type_id``. Model values reference that stable number through
``struct_type: { type_ref: id }``; they never reference a declaration's
position in the repeated field. The resolved declaration must be concrete.
An inline declaration remains useful for standalone values.

Producers assign IDs through a shared type registry so the same type can
keep the same number in different models. List order and payload length do
not affect the ID. Reusing a number requires identical physical layout,
format constants, and decoding/encoding semantics; names alone are not
identities. A different definition requires a different ID. The illustrative
numbers in these examples do not reserve global IDs.

Each model includes its referenced declarations. Reject missing references,
zero IDs, duplicate IDs within a model and conflicting definitions under
one ID when combining catalogues. Import/export must not silently renumber
a conflict or merge different decoding semantics.

Resolve and validate each type once in its catalogue scope. Values with
different payload lengths share that resolved type; do not create a cache of
``(type, template_arguments)`` instantiations. They retain their own
logical-shape/byte-extent checks and payload owners.

Dynamic KV values use a session-owned catalogue with stable resolved type
handles. The model catalogue is read-only; additional session types are
interned without mutating it or duplicating a declaration for every page.
Serialized type IDs retain their meaning across compatible catalogues;
runtime handles or dense lookup indices remain catalogue-local. Export of a
session-created value includes its declarations and preserves their IDs.
Import resolves those IDs against the destination catalogue, sharing matching
declarations and rejecting conflicts instead of remapping their identity.

Do not add generic template parameters, argument lists or an expression
language to the initial proto contract. Fixed arrays inside a byte-encoded
record remain part of its element type. General structs may already use
ONNX symbolic dimensions in their tensor fields, but that does not make them
fixed-size packed records. A runtime dimension binding alone does not select
a serialized layout or introduce an implicit reshape.

Quantization examples: constants and per-value parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following descriptive notation abbreviates ordinary ONNX tensor types as
``tensor(FLOAT, [])`` and tensor constants as ``tensor(FLOAT, [], 0.25)``.
These are explanatory helpers, not additional messages or protobuf syntax.

.. code-block:: text

    ModelProto {
        struct_types: {
            type_id: 1003
            name: "LINEAR_INT4_128_FIXED_PARAMETERS"
            structure: {
                field: {name: "values", type: array(INT4, dimension=128)}
                field: {name: "scale", constant: tensor(FLOAT, [], 0.25)}
                field: {name: "zero_point", constant: tensor(INT64, [], -2)}
            }
            decoder: DecodeLinearInt4
        }
        struct_types: {
            type_id: 1002
            name: "LINEAR_INT4_128_WITH_PARAMETERS"
            structure: {
                field: {name: "values", type: array(INT4, dimension=128)}
                field: {name: "scale", type: tensor(FLOAT, [])}
                field: {name: "zero_point", type: tensor(INT64, [])}
            }
            decoder: DecodeLinearInt4
        }
    }

    EncodedValueProto {
        struct_type: {type_ref: 1003}
        logical_type: FLOAT[128]
        raw_data: <64 code bytes for weight_a>
    }
    EncodedValueProto {
        struct_type: {type_ref: 1003}
        logical_type: FLOAT[128]
        raw_data: <64 code bytes for weight_b>
    }
    EncodedValueProto {
        struct_type: {type_ref: 1002}
        logical_type: FLOAT[128]
        raw_data: <64 code bytes, FLOAT scale=0.125, INT64 zero_point=0>
    }
    EncodedValueProto {
        struct_type: {type_ref: 1002}
        logical_type: FLOAT[128]
        raw_data: <64 code bytes, FLOAT scale=0.25, INT64 zero_point=-2>
    }

For type 1003, both constants are serialized once at
``ModelProto.struct_types[*].structure.field[*].constant``. Neither value
buffer contains them; there are no separate graph inputs for these parameters.
Each payload is exactly ``128 * 4 / 8 = 64`` bytes, and decoding applies
``(code - (-2)) * 0.25``. Changing the constants requires a different type ID.

For type 1002, both payloads occupy ``64 + 4 + 8 = 76`` bytes. The decoder
reads each value's parameters and applies ``(code - zero_point) * scale``.
Changing those values does not change the layout or type ID. The INT64 at
byte offset 68 is not necessarily aligned; typed readers must handle it.
Outside the byte buffer means inside the shared type declaration only for
true format constants, not arbitrary per-value state.

.. _l-next-steps-custom-types-codebook:

Codebook quantization through a shared subtype
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A subtype describes 32 two-bit indices and a constant four-entry codebook.
A parent embeds that subtype by stable ID and adds a per-block FLOAT scale.
This is composition by reference in the type catalogue, not inheritance or
a pointer to another value: each parent payload contains its own code bytes.

.. code-block:: text

    ModelProto {
        struct_types: {
            type_id: 1101
            name: "CODEBOOK2_BLOCK_32"
            structure: {
                field: {
                    name: "codes"
                    type: {
                        struct_type: {
                            bit_packing: {
                                component: {name: "index", bit_width: 2}
                                dimension: 32
                            }
                        }
                    }
                }
                field: {
                    name: "codebook"
                    constant: tensor(FLOAT, [4], [-1.0, -0.25, 0.25, 1.0])
                }
            }
        }
        struct_types: {
            type_id: 1102
            name: "SCALED_CODEBOOK2_BLOCK_32"
            structure: {
                field: {
                    name: "quantized"
                    type: {struct_type: {type_ref: 1101}}
                }
                field: {name: "scale", type: tensor(FLOAT, [])}
            }
            decoder: DecodeScaledCodebookBlocks
        }
    }

    EncodedValueProto {
        struct_type: {type_ref: 1102}
        logical_type: FLOAT[32]
        raw_data: <E4 E4 E4 E4 E4 E4 E4 E4 00 00 00 40>
    }
    EncodedValueProto {
        struct_type: {type_ref: 1102}
        logical_type: FLOAT[4096]
        raw_data: <128 records, each containing 8 code bytes and one FLOAT scale>
    }

Hexadecimal payload notation is illustrative, not a literal string. ``E4``
packs indices ``0, 1, 2, 3`` in least-significant-bit order; ``00 00 00 40``
encodes FLOAT 2.0. The codebook lives once in type 1101's constant field and
contributes no payload bytes.

Resolution follows ``1102 -> quantized.type -> 1101``. The root decoder reads
the subtype's codebook and each record's scale and indices:

.. code-block:: text

    table = resolved_type(1101).structure.field["codebook"].constant
    output[block * 32 + i] = record.scale * table[record.quantized.codes[i].index]

This is descriptive field-view notation. The decoder uses resolved types,
not model catalogue positions; it does not invoke a subtype decoder.
The first value decodes to ``[-2.0, -0.5, 0.5, 2.0]`` repeated eight times.
The second flattens 128 decoded blocks in storage order.

Type 1101 contributes eight bytes; type 1102 contributes ``8 + 4 = 12`` bytes
per record. Payload lengths 12 and 1536 imply one and 128 records without
serialized counts or physical shapes. Exactly four FLOAT codebook entries
make all two-bit indices valid.

Other parent types may reuse subtype 1101. Changing its constant codebook
requires a new subtype ID and a new parent ID when the reference changes.
Codes, per-block scales and payload lengths may change without new types.

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

.. _l-next-steps-custom-types-prepared-values:

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

Typed compiled-cache attachment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A compiled cache stores the result of a preparation operation. For example,
a quantized matrix prepack may be computed from weights ``W``, scales ``S``
and zero points ``Z``. Its bytes belong to ``EncodedValueProto``; preparation
metadata explains how those bytes were produced and when they can be reused.

The metadata answers three separate questions:

* ``sources``: which input values are needed to prepare or rebuild this value?
  Each entry records the operand's role and a reference resolved in its graph
  or function scope. Include bias only if preparation actually depends on it.
* The source's ``content_id``: are these still the same contents? The graph
  name ``W`` can stay unchanged while its weights change. A durable identity
  covers the canonical content, type, dimensions and interpretation, including
  relevant layout definitions and type constants, not just the display name.
* ``recipe`` and ``device``: which transformation produced the bytes, and
  which consumer can use them? They identify the preparation operation,
  version, relevant attributes and runtime/device compatibility.

Use one content identity per source, not a mandatory second
``sources_digest`` alongside those identities. The runtime derives the
complete prepared lookup key from the ordered source roles/identities,
recipe, output layout and compatibility requirements. It may hash that key
internally; that is not another serialized proof that the sources are current.
If a source reference is already content-addressed, reuse its identity instead
of duplicating it in a separate fingerprint field.

The following example uses graph-name references plus per-source hashes.
It is descriptive syntax, not a finalized wire schema. The digest values
are placeholders for hashes computed from the actual source values:

.. code-block:: text

    message DeviceProto {
        string type = 1;              // "cpu", "cuda", "rocm", ...
        optional int32 index = 2;     // exact ordinal only when required
        string architecture = 3;      // "x86_64-avx2", "sm_80", ...
        string runtime = 4;
        string runtime_version = 5;
        repeated StringStringEntryProto metadata_props = 6;
    }

    message ModelProto {
        repeated StructTypeProto struct_types = <N>;
        repeated DeviceProto devices = <N+1>;
        repeated EncodedValueProto prepared_values = <N+2>;
    }

    EncodedValueProto {
        struct_type: {type_ref: 3001}
        logical_type: FLOAT[K, N]
        raw_data: <complete packed records>
        preparation: {
            sources: [
                {
                    role: "weights"
                    value: {scope: "main", name: "W"}
                    content_id: {algorithm: "blake3", digest: <W identity>}
                },
                {
                    role: "scales"
                    value: {scope: "main", name: "S"}
                    content_id: {algorithm: "blake3", digest: <S identity>}
                },
                {
                    role: "zero_points"
                    value: {scope: "main", name: "Z"}
                    content_id: {algorithm: "blake3", digest: <Z identity>}
                }
            ]
            recipe: {
                name: "cpu.matmul.pack"
                version: 1
                attributes: {transpose_b: false}
            }
            device: 1                 // index into ModelProto.devices
        }
    }

Changing ``S`` changes its content identity and invalidates this prepared
value even when ``W`` is unchanged. Changing ``transpose_b`` or the packing
version changes the recipe and likewise prevents reuse. The source references
still identify the inputs from which to rebuild.

For a session-only in-memory prepared value, a tracked source identity and
generation may replace content hashing under an explicit immutability/version
contract. A pointer address alone is insufficient. Process-local generations
cannot validate a disk cache in another process; persisted entries require
durable content identities. A hash copied from an untrusted cache is not proof
of the current source: validation follows the source-identity rules above.

``source_lineage`` in the existing prepared-execution APIs means this ordered
source list. It does not require another source concept or registry in the
new value format, and the new example does not prescribe renaming those APIs.

Device type, architecture, implementation ABI and metadata express
compatibility, not merely a device ordinal. The device index is an index
into the model's device list; it is unrelated to stable structured type IDs.
One set of sources may have several entries for different devices or recipes.
FP32, tiled and quantized packs all use this attachment without another
compiled or quantized value-container hierarchy.

The checker validates unique device descriptors, in-range device indices,
unique complete preparation keys, source existence and unambiguous scope,
valid source content identities (including algorithm and digest for the hash
form), unique metadata keys, exact layouts
and valid byte extents. A payload cannot select an unconstrained structured
category. Device compatibility and digest comparison may be deferred until
load time, but structural errors are independent of hardware availability.
A missing portable decoder is acceptable for a runtime-specific derived pack
only because its original source remains available.

.. _l-next-steps-mutable-cache:

Persistent state from model inputs and outputs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

State is simply the model outputs retained to supply some inputs of the next
call. The model already describes their types; the caller supplies only the
output-to-input mapping and the initial values. Matching names or shapes is
not enough to infer which output is meant to feed which input.

.. code-block:: text

    model inputs:  tokens, past_key, past_value
    model outputs: logits, present_key, present_value

    state = make_state(
        model,
        feedback={
            "past_key": "present_key",       // input <- output
            "past_value": "present_value"
        },
        initial={"past_key": empty_key, "past_value": empty_value}
    )

    out = state.run({"tokens": first_tokens})
    out = state.run({"tokens": next_tokens})

This proposed helper is equivalent to the ordinary stateless loop:

.. code-block:: text

    feeds = {"tokens": tokens, **state.values}
    outputs = run(model, feeds)
    state.values = {
        input_name: outputs[output_name]
        for input_name, output_name in feedback.items()
    }

The retained values constitute the state. There is no separately authored
``state_spec``, hidden kernel state or new persistent proto. ``make_state``
derives field types from the selected model inputs and checks that the
corresponding outputs can feed them. Initial contents and unresolved dimensions
must be supplied; types alone cannot determine an initial cache.

.. _l-next-steps-persistent-composite-state:
.. _l-next-steps-persistent-struct-state:

A struct with only one persistent part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same mapping can select a field instead of an entire input. These are
ordinary structs described by ``StructTypeProto``:

.. code-block:: text

    Cache = struct {
        keys: Tensor
        values: Tensor
        length: INT64
    }

    model input request: struct {
        tokens: INT64[batch, sequence]
        cache: Cache
    }
    model output response: struct {
        logits: FLOAT[batch, sequence, vocabulary]
        cache: Cache
    }

    state = make_state(
        model,
        feedback={"request.cache": "response.cache"},
        initial={"request.cache": initial_cache}
    )
    out = state.run({"request.tokens": first_tokens})
    out = state.run({"request.tokens": next_tokens})

Only ``request.cache`` persists. Tokens are supplied anew and logits are not
retained by the state. To retain a smaller part, select its field path instead.
The struct declaration contains no ``persistent`` flag: the feedback mapping
selects the persistent part of this instance.

Dotted paths are shorthand for a graph input/output name followed by struct
field names, not additional model inputs. The helper assembles the input struct
from current feeds and retained fields. The actual tensor types and shape
constraints come from the model; ``Tensor`` above only abbreviates them.

Minimal rules
~~~~~~~~~~~~~

* Every selected input/output path must exist and have compatible types,
  representation contracts and shape constraints. Dynamic dimensions are
  checked on the actual values before they become next-call inputs.
* Every required input field comes from either the current feeds or retained
  state. Missing initial values, duplicate assignments and overlapping
  destination paths are errors.
* Update all retained fields only after successful execution and validation.
  Two states are independent; simultaneous calls using the same state are rejected.
* ``reset(initial)`` restores caller-supplied initial values; ``close`` releases
  retained values. Neither action may race with an active call.

Persistence does not imply mutation, a packed byte layout or disk persistence.
The first implementation preserves ordinary model input/output semantics.
In-place KV reuse is a later optimization when the kernel and ownership permit
it; it must not modify previously returned ordinary outputs. If such an
optimization fails after modifying state, reset is required before reuse.
Snapshots, a new alias-annotation wire format and region-level mutation
scheduling are outside this initial design.

Quantized and paged caches use the same feedback
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A cache can contain a sequence of blocks instead of contiguous K/V tensors.
This changes its fields and representations, not its persistence mechanism:

.. code-block:: text

    Cache = struct {
        blocks: sequence<KVBlock>
        length: INT64
    }
    feedback = {"request.cache": "response.cache"}

Each K/V block may use a different ``EncodedValueProto`` layout, such as
INT4, INT8 or a codebook struct, provided its decoded type and geometry satisfy
the consumer contract. The block's logical token range and valid length are
value data; payload byte length measures physical records, not valid tokens.
Changing scales requires a defined conversion of the affected codes.

Paging, block conversion and zero-copy Attention are optional consumer
optimizations after basic input/output feedback works. They must not require
another state description. Their later acceptance measures bounded workspace,
per-block allocation/conversion and no full-cache copy or dequantization;
they are not prerequisites for constructing persistent state.

GraphBuilder, shape inference and serialization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ShapesContext`` stays the single source of truth for symbolic value types.
Extend it with ``StructTypeProto`` field types and, where applicable, the
byte-encoded layout and optional logical view;
do not add an independent quantization registry in ``GraphBuilder``.
Known logical dimensions do not permit a tensor-only operator to consume
structured bytes implicitly: use an explicit decoder or a matching schema.

``GraphBuilder`` preserves structured source initializers, type references,
byte extents, external payload ownership and quantization metadata
through import/export, functions and subgraphs. Deduplication considers
semantic profiles as well as payload bytes, layout and optional logical shapes.
Rewrites that change any preparation dependency invalidate the corresponding
compiled binding. Feedback bindings are validated against the final model
inputs/outputs; a rewrite that removes or changes a selected path requires an
updated mapping, not a silently retained hidden state.

General struct declarations and their tensor/sequence field constraints
round-trip through the type catalogue as well. This does not make a runtime
cache instance a serializable initializer: the byte-payload encoding remains
limited to eligible fixed-layout structs. Unsupported field-value export
fails explicitly rather than dropping dynamic fields or serializing pointers.

Prefer an optional companion compiled store for the first implementation.
Do not make the core plan depend on modifying upstream ONNX wire messages or
on proto inheritance. If model extensions are later serialized, document their
version and round-trip behavior explicitly; standard ONNX export must lower
to supported tensors/operators or report an unsupported export, never silently
drop authoritative structured values or feedback bindings.

Implementation sequence
+++++++++++++++++++++++

All new steps are pending; completed foundations above are reused.

The first concrete implementation is the **structured representation**:
``StructTypeProto`` and the structured-layout branch of ``EncodedValueProto``.
PR01 first freezes their minimal contract and the proto-size budget. PR02
implements checked typed/constant fields, arrays, bit packing, type references,
and runtime field views using the same declarations. It implements payload
ownership, byte extents, derived record counts and value serialization for the
byte-encodable subset before adding the small built-in affine subset.
Custom packed weights and heterogeneous KV-block fixtures must work through
structures without requiring a catalogue of native quantized types.

The struct types, physical layouts, examples, prepared attachments and state
bindings above belong to this single sequence. Typed prepacking, graph
integration and persistent-state consumers build on the same representation
foundation, without a second cache or struct roadmap.

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
       fixed element types and payload-derived counts, catalogue identities,
       native bindings and typed input/output feedback. Use one struct type
       system and define which structs admit fixed-size byte encoding.
       Freeze whole-value/field-path matching and initial-value validation;
       do not require a separate state specification or alias proto.
       Record the minimal proto size baseline and agree a size budget before PR02.
     - Existing runtime APIs
   * - PR02
     - Structs first, then minimal built-in layouts
     - Implement StructTypeProto and EncodedValueProto's structured branch
       first; then add common INT8/INT4 layouts. Round-trip type declarations
       with tensor/sequence fields and expose checked runtime field views.
       Round-trip byte-encoded custom records,
       codebook/mixed-bit formats and a heterogeneous KV-block fixture.
       Prove one type is shared by different payload lengths; check single-record,
       empty, zero-sized-root rejection, overflow, catalogue resolution and
       exact payload-divisibility cases. A struct with dynamic fields is a
       valid type but is rejected as a flat byte-payload layout.
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
       Sources carry reusable content identities; an additional aggregate
       sources digest is not a required serialized field.
       Warm hits skip preparation; no-source-read claims have verified
       source identities. Round-trip typed/native objects via versioned payloads.
     - PR03
   * - PR05
     - GraphBuilder and model-resolution integration
     - Structured initializers, logical/physical inference, scope-aware
       references, deduplication and prepared payload selection agree.
       Rewrites invalidate stale bindings; standard export never loses data.
     - PR02, PR03
   * - PR06
     - State from model input/output feedback
     - Build state from selected input/output pairs and initial values.
       Infer its types from the model; retain only selected values or struct
       fields. Repeated calls match a manual stateless feedback loop.
       Verify initialization, validation, reset, failures and independent states.
     - PR02; existing allocation/task infrastructure
   * - PR07a
     - Contiguous KV and CPU consumer integration
     - Optimize the same past/present inputs and outputs when ownership
       permits buffer reuse. Append touches only new tokens, preserves
       ordinary fetched outputs and matches functional execution. Verify
       capacity/cancellation and measure allocation/copy costs; do not make
       zero-copy a precondition for the PR06 state helper.
     - PR03, PR06; CPU backend integration
   * - PR07b
     - Optional paged KV with heterogeneous quantization
     - The shared EncodedValueProto representation supports different K/V and
       per-block formats. Blockwise append/conversion and Attention preserve
       validity, numerical contracts and bounded workspace without copying
       or dequantizing the entire cache.
     - PR02, PR07a; CPU backend integration
   * - PR08
     - End-to-end prepared/stateful acceptance
     - Measure cold/warm preparation, repeated decode and simultaneous
       independent feedback states. Report source/packed/state/scratch bytes,
       preparation counts and per-token copies; verify stale-cache handling,
       eviction pins, request reset/isolation and the final proto-size budget.
     - PR04, PR05, PR06, PR07a
   * - Later
     - Snapshots and advanced page policies
     - Extend the input/output feedback contract without a second state type
       system. Snapshots, explicit alias annotations and advanced mutation
       scheduling remain outside the first implementation.
     - PR08

PR06 can proceed in parallel with PR03-PR05 after PR02 provides the shared
struct types and field views. Basic feedback state does not depend on every
quantization format, paging or a new mutation protocol. PR07b is optional and
does not block PR08.
The declarative format catalogue is not a prerequisite for FP32 prepacking.

Ownership and acceptance
++++++++++++++++++++++++

``onnx-light`` owns the type/serialization contracts, prepared identities,
allocation and lifecycle, graph/schema integration, effect scheduling and
input/output feedback state. ``onnx-light-cpu`` supplies its format validators, prepackers,
typed consumers, KV append and Attention implementation. It does not create
another persistent-state manager or private executor.

Acceptance uses C++ fixtures and existing runtime/backend test infrastructure.
Compare packed versus unpacked computation and the state helper versus a
manual output-to-input loop with the same numerical contract. Test concurrent
preparation, active pins during eviction, changed scales with unchanged code
bytes, incompatible ISA/ABI, missing consumers, reset, invalid capacities,
failed mutations and independent requests.

State fixtures retain a whole tensor and only the cache field of a larger
struct. Verify that tokens/logits are not retained, types come from model I/O,
the next call receives exactly the previous selected outputs, and missing,
overlapping or incompatible bindings fail explicitly. Test failed calls
without partial state updates and two independent feedback loops.
Dynamic fields remain valid struct types but cannot be encoded as fixed-size
inline payload fields. Optional reuse must not corrupt retained ordinary outputs.

Type/value tests also round-trip two encoded values with the same stable type
ID but different payload lengths and derived record counts. Include two models
with reordered catalogues and unchanged references, per-value scale/zero-point parameters,
missing and duplicate IDs, and conflicting definitions under the same ID.
Round-trip constants inside the shared type declaration without including
them in value-buffer sizes. Reject fields with both or neither of ``type``
and ``constant`` and invalid tensor constants. Reject unknown-size fields only
when validating eligibility for the byte-payload layout, not when declaring a
general struct type.
Verify a single shared resolved descriptor, one-record and empty payloads,
zero-sized-root rejection, allowed constant-only nested structures, partial
records, arithmetic overflow, explicit external lengths and validated extents,
logical shapes inconsistent with derived record counts, and
session-to-model catalogue resolution preserving IDs without copying type
declarations per KV block.

Prepared-cache fixtures change a source's contents without changing its graph
name, change only a scale or a recipe attribute, and reuse unchanged
content-addressed sources. No extra aggregate digest is needed to reject a
stale value. Session-only generation identities must not validate persisted
entries in another process.

Structural gates are explicit: a reused prepared object has no repeat
prepacking and a compatible verified disk hit does not read portable payloads.
The basic state helper promises correct feedback, not zero-copy execution.
Only a demonstrated fixed-capacity reuse path may claim no full-cache
allocation or copy.
Publish latency, dispersion, peak/resident bytes and copy/read counters;
performance claims must distinguish source validation, preparation, inference
and state-management cost.
