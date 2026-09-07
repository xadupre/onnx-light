.. _l-next-steps-custom-types:

Structured types and prepared values
====================================

:Date: 2026-08

**consolidated design reference**

.. note::

    The implementation sequence and current decisions are consolidated in
    :ref:`l-next-steps-prepared-values-and-persistent-state`. This page retains
    the physical-layout and typed prepared-cache proposals; their new proto
    messages are not implemented. The unified plan takes precedence where
    details differ.
    Its first concrete implementation is ``StructTypeProto`` together with
    the structured branch of ``EncodedValueProto``. All value examples below
    use this single container; there is no separate ``StructProto``.
    The current contract describes one fixed-size element and derives the
    number of records from the payload byte length. Different repetition
    counts share one catalogue declaration, without template parameters,
    type instantiations or a serialized physical shape.
    Type references use explicit stable numeric identifiers, not positions
    in ``ModelProto.struct_types``.
    The former compiled-tensor proposal is incorporated in
    :ref:`l-next-steps-custom-types-prepared-values` below, using the same
    value container with optional preparation metadata.

Motivation
++++++++++

``TypeProto.Opaque`` identifies a runtime-owned value by domain and name,
but it gives no information about its serialized representation. A generic
reader cannot determine how many values are present, where fields begin,
or how many bytes may safely be read.

Conversely, adding one protobuf message for every quantization or custom
format creates a closed hierarchy that must grow whenever a new layout is
introduced.

``EncodedValueProto`` owns or references a byte buffer and selects its
physical layout. Its structured branch references a ``StructTypeProto``;
the other branches cover the small built-in dense/affine subset. The
following sketch shows only the structured branch, with wire field numbers
still to be frozen in the unified plan:

.. code-block:: text

    message EncodedValueProto {
        oneof layout {
            StructTypeProto struct_type = <N>;  // type_ref or concrete inline type
            // Other built-in layout branches are omitted here.
        }
        optional TypeProto logical_type = <N>;
        bytes raw_data = <N>;
        repeated StringStringEntryProto external_data = <N>;
        string name = <N>;
        string doc_string = <N>;
    }

The complete container contract belongs to the unified plan. This page
defines ``StructTypeProto``: a reusable description of one physical element,
including its constant fields, not a second payload container. ``Encoded``
does not imply compression or quantization; it also covers custom records
and prepacked values.

The type system adds three serialized structural kinds:

* an array of a statically sized ``TypeProto``;
* a bit packing of repeated named components;
* a structure containing named fields.

Each structure field selects either a statically sized ``TypeProto`` for
data in the value buffer, or a ``TensorProto constant`` stored in the type.
A constant is a field value, not a fourth structural kind, and consumes no
bytes in the encoded value buffer.

Scalars and ordinary tensors continue to use ``TypeProto.Tensor``. Quantized
values, packed records, custom numeric types, image pixels, and other static
binary formats are recursive compositions of existing ONNX types and these
additions.

Requirements
++++++++++++

* The inline or external payload length is an exact multiple of the
  strictly positive physical element byte size.
* Every read performed by the structured view is bounds-checked.
* Bits and multi-byte values use one canonical ordering convention.
* A structure may be nested and repeated without introducing a new proto
  for each format.
* Every array and bit-packing length is a concrete non-negative integer.
* The number of records is computable from the type and payload byte length
  without interpreting the payload contents.
* The physical structure is inspectable without loading a vendor plugin.
* An optional standard ONNX decoder defines logical semantics such as
  dequantization.

Stable contract
+++++++++++++++

The proposal has three valid uses of ``StructTypeProto``:

``concrete declaration``
    Selects ``array``, ``bit_packing``, or ``structure``. It appears in
    ``ModelProto.struct_types`` or in
    ``EncodedValueProto.struct_type``. It completely determines the size of
    one physical element.

``exact static reference``
    Selects ``type_ref`` and appears inside ``TypeProto`` or the structured
    layout of ``EncodedValueProto``. Its numeric value identifies the
    declaration with that ``type_id``, independent of its position in the
    model's catalogue.

``unconstrained static category``
    Leaves ``kind`` unset and appears only inside ``TypeProto``. It accepts
    any concrete structured declaration. This form is used by heterogeneous
    sequences and maps.

``type_ref`` may also occur below a concrete root through
``Array.element_type`` or ``Structure.Field.type``. A constant tensor value is
attached directly to a ``Structure.Field``. A concrete root may not
be a ``type_ref`` or an unset ``kind``. Static reference/category forms may
not carry their own declaration ``type_id``, ``decoder``, ``encoder``,
``name``, or metadata.

Only the ``decoder`` and ``encoder`` attached to the selected concrete root
are invoked. A declaration reached through a nested ``type_ref`` contributes
only its physical structure and constants; its decoder and encoder are not
composed implicitly.

Counts inside a type remain explicit and concrete. Only the number of
complete records in a value is derived from its payload byte length.
There are no inferred field dimensions, implicit alignment, hidden padding,
semantic traits, or alternate byte orders.

Physical size function
++++++++++++++++++++++

The serialized size is computed recursively in bits from the concrete root
declaration:

.. code-block:: text

    size(scalar(T))             = bit_width(T)
    size(tensor(T, dims))       = checked_product(dims) * bit_width(T)
    size(Array(T, n))           = n * size(T)
    size(BitPacking(c..., n))   = n * sum(c.bit_width)
    size(Field(constant))       = 0
    size(Field(T))              = size(T)
    size(Structure(f...))       = sum(size(f))
    size(type_ref=id)           = size(resolve_type_id(id))

All arithmetic is checked in ``uint64``. References must be acyclic. The
concrete root of an encoded value must have a strictly positive size
divisible by eight. For the structured layout:

.. code-block:: text

    element_bytes = size(resolved_struct_type) / 8
    payload_bytes = raw_data.size()        // inline payload
    // Or external_data.length for a validated external payload extent.
    require(element_bytes > 0)
    require(payload_bytes % element_bytes == 0)
    element_count = payload_bytes / element_bytes

The selected payload source supplies the byte length: ``raw_data.size()``
for inline bytes, or an explicit ``external_data.length`` for external bytes.
Validate external offsets, lengths and actual backing-file bounds; an absent
external length is not an instruction to consume the rest of a file.
Conflicting inline and external payload sources are rejected.
Do not serialize a redundant byte count, record count or physical shape.

An empty payload means zero records of a positive-sized type. A payload of
exactly ``element_bytes`` means one record. A zero-sized root is rejected:
its byte length cannot distinguish zero, one or many instances. Zero-sized
substructures, including constant-only field groups, remain valid inside a
positive-sized root and consume no payload bytes.

The optional logical type/shape describes the decoded value. Its consistency
with the derived record count is checked by the decoder or format contract;
byte length alone cannot determine tensor rank or dimensions.

StructTypeProto
+++++++++++++++

The complete proposal adds one top-level structured type message.
``Structure``, ``Array`` and ``BitPacking`` below are its nested message
declarations, not undefined external types. In particular,
``StructTypeProto.Structure.Field`` explicitly declares the ``constant``
alternative:

.. code-block:: text

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
        optional uint64 type_id = 11;  // identity of a concrete declaration
    }

Where constants are stored
~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact path is ``StructTypeProto.structure.field[i].constant``.
``Structure`` groups the fields; it does not own another value buffer.
For each field, ``oneof content`` selects exactly one of:

* ``type``: describes bytes to read from ``EncodedValueProto.raw_data`` or
  its external payload;
* ``constant``: contains the actual ``TensorProto`` value in the type
  declaration, without advancing the value-buffer offset.

For example, the constant scale is represented by this fragment of a
``StructTypeProto`` in protobuf text notation:

.. code-block:: text

    structure {
        field {
            name: "scale"
            constant {
                data_type: 1          # TensorProto.FLOAT
                float_data: 0.25
                # No dims entries: this is a scalar.
            }
        }
    }

There is no ``structure: Structure { ... }`` constructor in that notation:
``structure { ... }`` selects the field whose declared message type is
``StructTypeProto.Structure``. The compact examples below abbreviate ONNX
tensor types as ``tensor(FLOAT, [])``, constant tensor values as
``tensor(FLOAT, [], 0.25)``, and fixed array types as
``array(INT4, dimension=128)``. Those helpers are explanatory shorthand, not
additional proto messages or wire syntax.

``Field.type`` accepts only a statically sized tensor type or a concrete
structured type, directly or by reference. Unknown rank, symbolic dimensions,
sequences, maps, optional values and opaque types cannot describe fixed
payload fields. ``Field.constant`` must be a valid tensor value with concrete
dimensions and matching data; it is not a reference to a graph input.

Integration
+++++++++++

.. code-block:: text

    message ModelProto {
        repeated StructTypeProto struct_types = <N>;
    }

    message TypeProto {
        oneof value {
            ...
            StructTypeProto struct_type = <N>;
        }
    }

A reusable value selects a declaration through
``EncodedValueProto.struct_type: { type_ref: id }``. Every
declaration in ``ModelProto.struct_types`` has its own nonzero ``type_id``;
the model builds an ID-to-declaration lookup rather than interpreting the ID
as an array index. A standalone value can place a concrete declaration in
``EncodedValueProto.struct_type`` instead, with no ``-1`` sentinel. Nested
references use ``type_ref`` with the same identifier.

The identifier belongs to the type contract, not to one model. An exporter
can reuse, for example, ``type_id=1001`` in several models even if the
declaration occupies different positions in their lists. List reordering
must not change any reference. The example numbers on this page are
illustrative, not reservations in a global registry.

Stable identifiers must be assigned and shared explicitly by the producer's
type registry; they are not generated from insertion order or from a type's
display name. Reusing an identifier requires the same physical definition,
format constants, and logical interpretation, including encoder/decoder
semantics. Changing any of those requires a different identifier. Friendly
names and documentation alone do not define identity.

Zero is reserved as an invalid identifier. Reject duplicate IDs in one model,
unresolved references, and conflicting definitions for the same ID when
combining models or session catalogues; never silently reinterpret or
renumber them. A model includes the declarations needed to resolve its
references without an implicit external registry. Runtime-local dense
indices may accelerate lookup, but are not serialized type identities.

Physical rules
++++++++++++++

* Arrays and bit packings are tightly packed.
* Structure fields are serialized in declaration order.
* Constants consume no payload bytes.
* Padding must be represented explicitly.
* Bits are ordered from least to most significant within each byte.
* Multi-byte values are little-endian.
* Only fixed-width ONNX scalar types are valid physical leaves.
* The decoder maps physical fields to one logical ONNX value.

Example: quantization parameters fixed by the type
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following type stores 128 ``INT4`` values plus format constants:

.. code-block:: text

    StructTypeProto {
        type_id: 1001
        name: "LINEAR_INT4_128"
        structure: {
            field: {
                name: "values"
                type: array(INT4, dimension=128)
            }
            field: {
                name: "scale"
                constant: tensor(FLOAT, [], 0.125)
            }
            field: {
                name: "zero_point"
                constant: tensor(INT64, [], 0)
            }
        }
    }

    EncodedValueProto {
        struct_type: { type_ref: 1001 }
        logical_type: FLOAT[128]
        raw_data: <64 bytes>
        name: "weight"
    }

The payload size is ``128 * 4 / 8 = 64`` bytes. Constants are stored in the
type and do not contribute to that size. This example is intentionally kept:
every value of type 1001 uses scale 0.125 and zero point 0. Different fixed
parameters would define a different type.

Example: quantization parameters supplied by each value
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here the scale and zero-point values are not part of the type definition.
The reusable layout describes only their scalar types and positions in the
payload, just as it describes the array of codes:

.. code-block:: text

    StructTypeProto {
        type_id: 1002
        name: "LINEAR_INT4_128_WITH_PARAMETERS"
        structure: {
            field: {
                name: "values"
                type: array(INT4, dimension=128)
            }
            field: {
                name: "scale"
                type: tensor(FLOAT, [])
            }
            field: {
                name: "zero_point"
                type: tensor(INT64, [])
            }
        }
    }

    EncodedValueProto {
        struct_type: { type_ref: 1002 }
        logical_type: FLOAT[128]
        raw_data: <64 code bytes, FLOAT scale=0.125, INT64 zero_point=0>
        name: "weight_a"
    }

    EncodedValueProto {
        struct_type: { type_ref: 1002 }
        logical_type: FLOAT[128]
        raw_data: <64 code bytes, FLOAT scale=0.25, INT64 zero_point=-2>
        name: "weight_b"
    }

Both payloads contain exactly ``64 + 4 + 8 = 76`` bytes, in field order and
little-endian representation, with no implicit alignment padding. The
``raw_data`` descriptions above are illustrative, not text stored on the
wire. A typed reader cannot assume that the INT64 at byte offset 68 is
aligned.

The decoder reads the scale and zero point from each value and applies
``decoded[i] = (INT4(values[i]) - zero_point) * scale``. Their numeric values
neither change ``type_id=1002`` nor require a new type declaration.
The two values may belong to different models, each declaring the same
type 1002 at any position in its catalogue.

Variant: parameters outside the byte buffer, stored in the type
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Outside the byte buffer does not mean outside ``StructTypeProto``. To keep
only the codes in each payload, store scale and zero point as constant fields
in the shared type declaration. This uses the same mechanism as the first
example, with the model-level storage and value references shown explicitly:

.. code-block:: text

    ModelProto {
        struct_types: {
            type_id: 1003
            name: "LINEAR_INT4_128_FIXED_PARAMETERS"
            structure: {
                field: {
                    name: "values"
                    type: array(INT4, dimension=128)
                }
                field: {
                    name: "scale"
                    constant: tensor(FLOAT, [], 0.25)
                }
                field: {
                    name: "zero_point"
                    constant: tensor(INT64, [], -2)
                }
            }
            decoder: DecodeLinearInt4  // returns FLOAT[128]
        }
    }

    EncodedValueProto {
        struct_type: { type_ref: 1003 }
        logical_type: FLOAT[128]
        raw_data: <64 code bytes for weight_a>
        name: "weight_a"
    }

    EncodedValueProto {
        struct_type: { type_ref: 1003 }
        logical_type: FLOAT[128]
        raw_data: <64 code bytes for weight_b>
        name: "weight_b"
    }

The model serializes the two constants once inside
``ModelProto.struct_types`` in the declaration whose ``type_id`` is 1003.
Neither ``raw_data`` buffer contains them, and there are no separate graph
inputs or initializers for these parameters. The decoder resolves the type,
reads its constants and computes ``(code - (-2)) * 0.25`` for either value.

Both values share one declaration and each payload remains exactly 64 bytes.
Another model can include that same declaration with the same ID. Changing
scale or zero point changes the type definition and therefore requires a
different ID; use the preceding per-value payload example when parameters
must vary without changing the type.

.. _l-next-steps-custom-types-codebook:

Example: codebook quantization through a shared subtype
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The subtype below describes 32 two-bit indices and a constant four-entry
codebook. A second type embeds that subtype by its stable ID and adds a
per-block FLOAT scale. This is composition by reference, not type
inheritance; neither the subtype declaration nor its codebook is copied
into each parent record. The reference lives in the type declaration:
each parent payload still contains its own eight code bytes, not a pointer
to another value.

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
                                component: { name: "index", bit_width: 2 }
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
                    type: { struct_type: { type_ref: 1101 } }
                }
                field: {
                    name: "scale"
                    type: tensor(FLOAT, [])
                }
            }
            decoder: DecodeScaledCodebookBlocks
        }
    }

    EncodedValueProto {
        struct_type: { type_ref: 1102 }
        logical_type: FLOAT[32]
        raw_data: <E4 E4 E4 E4 E4 E4 E4 E4 00 00 00 40>
        name: "one_block"
    }

    EncodedValueProto {
        struct_type: { type_ref: 1102 }
        logical_type: FLOAT[4096]
        raw_data: <128 records, each containing 8 code bytes and one FLOAT scale>
        name: "weight"
    }

The first ``raw_data`` is shown as hexadecimal bytes, not a string to store
literally. Each byte ``E4`` packs indices ``0, 1, 2, 3`` in least-significant
bit order; ``00 00 00 40`` is the little-endian FLOAT value 2.0.
The codebook is serialized once in the ``constant`` member of type 1101's
field named ``codebook``, not in either value buffer.

Resolution follows ``1102 -> quantized.type -> 1101``. The root decoder
reads each block's scale and packed indices from its payload, and the
codebook from the resolved subtype's constant field:

.. code-block:: text

    table = resolved_type(1101).structure.field["codebook"].constant
    output[block * 32 + i] = record.scale * table[record.quantized.codes[i].index]

Here the field-name lookup and ``record`` view are explanatory notation.
The registered decoder is bound to the resolved descriptors; it does not
hard-code a model catalogue position. Only the parent decoder is invoked.
A decoder attached to the subtype would not be composed automatically.
The first value decodes to ``[-2.0, -0.5, 0.5, 2.0]`` repeated eight times.
For the second value, the decoder flattens 128 decoded blocks in storage order.

Type 1101 contributes ``32 * 2 / 8 = 8`` physical bytes. Its codebook occupies
space in the type declaration but contributes zero bytes to the value
payload. Type 1102 therefore occupies ``8 + 4 = 12`` bytes per record:
the two value buffers contain exactly 12 and ``128 * 12 = 1536`` bytes.
The format validator requires exactly four FLOAT codebook entries, so every
two-bit index addresses a valid entry.

The byte lengths therefore imply one and 128 records respectively; no
record count or physical shape is serialized alongside them.

Different codes, per-block scales and payload lengths reuse both declarations.
Other structured types may also reference subtype 1101. Changing the constant
codebook requires a new subtype ID and a new parent ID when its reference
changes; it must not silently change the meaning of the existing IDs.

Validation
++++++++++

A checker rejects:

* an invalid or cyclic type reference;
* a zero, missing, duplicate or conflicting catalogue type identifier;
* a field without exactly one of ``type`` and ``constant``;
* an invalid tensor constant or a field type without a concrete physical size;
* duplicate field or component names;
* zero component widths or unsupported physical leaf types;
* a physical size that is not byte-aligned;
* a zero-sized root used by an encoded value;
* a negative or non-concrete physical field dimension, or overflowing size arithmetic;
* a payload length that is not an exact multiple of the element byte size;
* a missing external length, invalid byte extent or conflicting payload sources;
* implicit padding or untyped trailing bytes.

.. _l-next-steps-custom-types-prepared-values:

Prepared values and compiled caches
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A runtime may transform an initializer into a device-specific packed
representation. Persisting that representation avoids repeating an expensive
prepacking step when the same model is loaded again.

A compiled tensor is a cache, not a new tensor semantics. The graph continues
to reference the original initializer, which remains the portable fallback.
The cached bytes use the same encoded-value representation described above
and are ignored when the current runtime or device is incompatible.

Reuse the implemented prepared object store and disk cache. The proposed
typed serialization uses ``EncodedValueProto`` with optional preparation
metadata, not a separate compiled-value container or wrapper. Preparation
keys cover all source operands and their semantics.

Value and preparation metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prepared payload is an ``EncodedValueProto``. Preparation metadata
records its provenance and compatibility without owning another payload.
The following descriptive sketch does not freeze wire field numbers or
metadata field names:

.. code-block:: text

    EncodedValueProto {
        struct_type: { type_ref: 3001 }
        logical_type: FLOAT[K, N]
        raw_data: <complete packed records>
        preparation: {
            source_lineage: ...       // ordered operands and their semantics
            source_lineage_digest: ...
            digest_algorithm: "blake3"
            recipe: ...               // operator, packing and tuning choices
            device: 1                 // index into ModelProto.devices
        }
    }

The structured branch resolves a stable ``type_ref`` ID or an inline
concrete ``struct_type``. An unconstrained static type is not valid for a
payload. A built-in layout may instead use the same value container without
a structured declaration.

Source lineage resolves every operand in its graph scope, including scales,
zero points, bias and other inputs when they contribute to preparation.
Its digest prevents stale prepared data from being used after any dependency
changes. The digest algorithm is explicit. The key covers canonical source
content, types, dimensions, type constants and interpretation, together with
the recipe and compatibility metadata; it is not merely a digest of one
weight buffer. Tensor names and external-data locations are not sufficient
content identities.

DeviceProto
~~~~~~~~~~~

The device descriptor identifies compatibility, not only a device ordinal:

.. code-block:: text

    message DeviceProto {
        string type = 1;              // "cpu", "cuda", "rocm", ...
        optional int32 index = 2;     // exact ordinal only when required
        string architecture = 3;      // "x86_64-avx2", "sm_80", ...
        string runtime = 4;           // producer/runtime domain
        string runtime_version = 5;   // compatible runtime version
        repeated StringStringEntryProto metadata_props = 6;
    }

``architecture`` records the instruction-set or accelerator compatibility
needed by the packed representation. ``runtime`` and ``runtime_version``
identify the implementation ABI that interprets the type. Additional
compatibility keys may be stored in ``metadata_props``.

ModelProto extension
~~~~~~~~~~~~~~~~~~~~

An illustrative model extension stores the same value messages directly:

.. code-block:: text

    message ModelProto {
        ...
        repeated StructTypeProto struct_types = <N>;
        repeated DeviceProto devices = <N+1>;
        repeated EncodedValueProto prepared_values = <N+2>;
    }

Several prepared entries may use the same source lineage for different
architectures, runtimes, or packing strategies. Stable structured type IDs,
resolved definitions and compatibility metadata distinguish physical formats;
display names alone do not. The collection is a cache attachment, not a new
value category or a replacement for portable graph initializers.

Loading rules
~~~~~~~~~~~~~

A runtime uses a compiled entry only when all of the following hold:

* every source operand resolves unambiguously in its scope;
* the source-lineage digest and preparation recipe match current dependencies;
* ``device`` is an in-range model-level index;
* device type, architecture, runtime, version, and required metadata are
  compatible;
* the encoded value, selected layout and byte extent pass structural and
  payload-size validation, including exact record divisibility for structures;
* the runtime recognizes that physical type and compiled-format version.

If a compatibility condition or digest comparison fails, the runtime treats
the entry as a cache miss and rebuilds it from the portable source operands.
Invalid compiled data must never change graph results or make an otherwise
valid portable model unloadable. Malformed indices, payloads, or digest
declarations are checker errors; ordinary incompatibility or a stale digest
is only a cache miss.

Quantized and tiled tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

No dependency on ``QuantizedTensorProto`` is needed. A quantized, tiled, or
otherwise packed cache entry is represented by the same
``EncodedValueProto`` mechanism:

.. code-block:: text

    EncodedValueProto {
        struct_type: { type_ref: 3002 } // stable packed CUDA type ID
        logical_type: FLOAT[K, N]
        raw_data: <complete packed records>
        preparation: {
            source_lineage: ...        // includes weight and all other operands
            source_lineage_digest: ...
            digest_algorithm: "blake3"
            recipe: ...
            device: 1                  // e.g. CUDA sm_80 + runtime ABI
        }
    }

The referenced structured type describes the complete byte layout. Its
optional decoder describes portable interpretation for inspectable formats.
A runtime-specific prepack may omit the decoder when only the named runtime
can consume it; the original initializer still guarantees portability.

Prepared-value validation
~~~~~~~~~~~~~~~~~~~~~~~~~

A checker validates:

* unique device descriptors and valid device indices;
* unique complete preparation keys, including lineage, recipe, layout and device;
* source operand existence and unambiguous scope;
* non-empty digest and algorithm fields;
* exact layout resolution, stable structured IDs and payload size;
* absence of unconstrained static structured types;
* metadata keys are unique.

Digest comparison and runtime compatibility may be deferred until load time,
but structural errors are rejected independently of hardware availability.

Relationship to other proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparation reuses the physical representation in ``EncodedValueProto``.
The specialized hierarchy in
:ref:`l-next-steps-quantization` may remain a format catalogue, but it is not a
storage dependency. Proto inheritance and wrapper containers are unnecessary:
the preparation recipe is optional metadata on the same encoded value.
