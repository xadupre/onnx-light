.. _l-next-steps-custom-types:

Structured views over byte buffers
==================================

:Date: 2026-08

**consolidated design reference**

.. note::

    The implementation sequence and current decisions are consolidated in
    :ref:`l-next-steps-prepared-values-and-persistent-state`. This page retains
    the original physical-layout proposal; its structures are not implemented
    proto contracts. The unified plan takes precedence where details differ.
    Its first concrete implementation is ``StructTypeProto`` together with
    the structured branch of ``EncodedValueProto``, which replaces the
    historical ``StructProto`` container below.
    The current contract separates the fixed-size element type from
    ``EncodedValueProto.storage_shape``: different repetition counts share
    one catalogue declaration, without template parameters or type
    instantiations. The historical size rules below describe one element;
    its total value size additionally includes the storage-shape product.
    Type references use explicit stable numeric identifiers, not positions
    in ``ModelProto.struct_types``. The same rule applies to both the
    illustrative ``StructProto`` below and the unified ``EncodedValueProto``.

Motivation
++++++++++

``TypeProto.Opaque`` identifies a runtime-owned value by domain and name,
but it gives no information about its serialized representation. A generic
reader cannot determine how many values are present, where fields begin,
or how many bytes may safely be read.

Conversely, adding one protobuf message for every quantization or custom
format creates a closed hierarchy that must grow whenever a new layout is
introduced.

This proposal assumes that a ``StructProto`` already exists. It owns
or references a byte buffer and references the physical type from which its
exact byte size is computed:

.. code-block:: text

    message StructProto {
        oneof type {
            uint64 type_id = 1;           // stable identifier, never a list index
            StructTypeProto struct_type = 2;  // standalone inline declaration
        }
        bytes raw_data = 3;
        repeated StringStringEntryProto external_data = 4;
        string name = 5;
        string doc_string = 6;
    }

The container itself is outside the scope of this page. The purpose of the
specification is to define ``StructTypeProto``: a portable structure
that can be overlaid on the bytes of a ``StructProto``.

The type system adds three serialized structural kinds and constant fields:

* an array of a statically sized ``TypeProto``;
* a bit packing of repeated named components;
* a structure containing named, statically sized ``TypeProto`` fields;
* a tensor constant consuming no payload bytes.

Scalars and ordinary tensors continue to use ``TypeProto.Tensor``. Quantized
values, packed records, custom numeric types, image pixels, and other static
binary formats are recursive compositions of existing ONNX types and these
additions.

Requirements
++++++++++++

* The size implied by the physical type equals the inline or external payload
  length.
* Every read performed by the structured view is bounds-checked.
* Bits and multi-byte values use one canonical ordering convention.
* A structure may be nested and repeated without introducing a new proto
  for each format.
* Every array and bit-packing length is a concrete non-negative integer.
* The number of payload bytes of a value is computable from its type without
  reading the payload.
* The physical structure is inspectable without loading a vendor plugin.
* An optional standard ONNX decoder defines logical semantics such as
  dequantization.

Stable contract
+++++++++++++++

The proposal has three valid uses of ``StructTypeProto``:

``concrete declaration``
    Selects ``array``, ``bit_packing``, or ``structure``. It appears in
    ``ModelProto.struct_types`` or in
    ``StructProto.struct_type``. It completely determines the payload size.

``exact static reference``
    Selects ``type_ref`` and appears inside ``TypeProto``. Its numeric value
    identifies the declaration with that ``type_id``, independent of its
    position in the model's catalogue.

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

No other interpretation of an absent field is permitted. In particular,
counts are concrete, and there are no inferred lengths, implicit alignment,
hidden padding, semantic traits, or alternate byte orders.

Physical size function
++++++++++++++++++++++

The serialized size is computed recursively in bits from the concrete root
declaration:

.. code-block:: text

    size(scalar(T))             = bit_width(T)
    size(Array(T, n))           = n * size(T)
    size(BitPacking(c..., n))   = n * sum(c.bit_width)
    size(Field(constant))       = 0
    size(Field(T))              = size(T)
    size(Structure(f...))       = sum(size(f))
    size(type_ref=id)           = size(resolve_type_id(id))

All arithmetic is checked in ``uint64``. References must be acyclic. The
concrete root size must be divisible by eight and equal the inline
``raw_data`` length or the external-data ``length``, so the physical schema
and payload remain independently checkable.

StructTypeProto
+++++++++++++++

The complete proposal adds one top-level structured type message.

.. code-block:: text

    message StructTypeProto {
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

        message Field {
            string name = 1;
            oneof content {
                TypeProto type = 2;
                TensorProto constant = 4;
            }
            string doc_string = 3;
        }

        message Structure {
            repeated Field field = 1;
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

A reusable value selects a declaration by ``StructProto.type_id``. Every
declaration in ``ModelProto.struct_types`` has its own nonzero ``type_id``;
the model builds an ID-to-declaration lookup rather than interpreting the ID
as an array index. An inline value selects ``StructProto.struct_type``
instead, with no ``-1`` sentinel. Nested references use ``type_ref`` with
the same identifier.

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
+++++++++++++++++++++++++++++++++++++++++++++++++

The following type stores 128 ``INT4`` values plus format constants:

.. code-block:: text

    StructTypeProto {
        type_id: 1001
        name: "LINEAR_INT4_128"
        structure: Structure {
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

    StructProto {
        type_id: 1001
        raw_data: <64 bytes>
        name: "weight"
    }

The payload size is ``128 * 4 / 8 = 64`` bytes. Constants are stored in the
type and do not contribute to that size. This example is intentionally kept:
every value of type 1001 uses scale 0.125 and zero point 0. Different fixed
parameters would define a different type.

Example: quantization parameters supplied by each value
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here the scale and zero-point values are not part of the type definition.
The reusable layout describes only their scalar types and positions in the
payload, just as it describes the array of codes:

.. code-block:: text

    StructTypeProto {
        type_id: 1002
        name: "LINEAR_INT4_128_WITH_PARAMETERS"
        structure: Structure {
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

    StructProto {
        type_id: 1002
        raw_data: <64 code bytes, FLOAT scale=0.125, INT64 zero_point=0>
        name: "weight_a"
    }

    StructProto {
        type_id: 1002
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

Variant: parameters completely outside the structured layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If neither scale nor zero point should even be a field of the physical type,
keep them as separate graph tensors. The structured type then describes only
the codes, and its decoder returns an ordinary INT4 tensor, not dequantized
FLOAT values:

.. code-block:: text

    StructTypeProto {
        type_id: 1003
        name: "INT4_CODES_128"
        structure: Structure {
            field: {
                name: "values"
                type: array(INT4, dimension=128)
            }
        }
        decoder: DecodeInt4Codes      // returns INT4[128]
    }

    StructProto {
        type_id: 1003
        raw_data: <64 code bytes>
        name: "weight_codes"
    }

    // Separate ordinary tensor inputs or initializers, not type constants.
    weight_scale = tensor(FLOAT, [], 0.25)
    weight_zero_point = tensor(INT4, [], -2)
    codes = DecodeInt4Codes(weight_codes)
    decoded = DequantizeLinear(codes, weight_scale, weight_zero_point)

The graph binds the three values explicitly. Other weights reuse type 1003
with their own code buffers and parameter tensors. There is no implicit
name-based lookup or hidden capture of a model initializer by the type.
The structured payload remains 64 bytes; the separate tensor owners retain
their scale and zero-point storage. A prepared dequantization or fused
consumer must include both parameter tensors in its preparation key.

Validation
++++++++++

A checker rejects:

* an invalid or cyclic type reference;
* a zero, missing, duplicate or conflicting catalogue type identifier;
* a field without exactly one of ``type`` and ``constant``;
* duplicate field or component names;
* zero component widths or unsupported physical leaf types;
* a physical size that is not byte-aligned;
* a payload whose length differs from the computed size;
* implicit padding or untyped trailing bytes.
