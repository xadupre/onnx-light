.. _l-next-steps-custom-types:

Structured views over byte buffers
==================================

:Date: 2026-08

**discussion**

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
        int32 type = 1;  // model-level index, or -1 when struct_type is present
        optional StructTypeProto struct_type = 2;
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
    Selects ``type_index`` and appears inside ``TypeProto``. It accepts only
    the referenced model-level declaration.

``unconstrained static category``
    Leaves ``kind`` unset and appears only inside ``TypeProto``. It accepts
    any concrete structured declaration. This form is used by heterogeneous
    sequences and maps.

``type_index`` may also occur below a concrete root through
``Array.element_type`` or ``Structure.Field.type``. A constant tensor value is
attached directly to a ``Structure.Field``. A concrete root may not
be a ``type_index`` or an unset ``kind``. Static forms may not carry
``decoder``, ``encoder``, ``name``, or metadata.

Only the ``decoder`` and ``encoder`` attached to the selected concrete root
are invoked. A declaration reached through a nested ``type_index`` contributes
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
    size(type_index=i)          = size(ModelProto.struct_types[i])

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
            int32 type_index = 5;
        }

        optional FunctionProto decoder = 6;
        optional FunctionProto encoder = 7;
        string name = 8;
        string doc_string = 9;
        repeated StringStringEntryProto metadata_props = 10;
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

A reusable value selects an entry in ``ModelProto.struct_types`` with
``StructProto.type``. An inline value sets ``type`` to -1 and provides
``StructProto.struct_type``. Nested structures use ``type_index`` to reference
another model-level declaration.

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

Example
+++++++

The following type stores 128 ``INT4`` values plus format constants:

.. code-block:: text

    StructTypeProto {
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
        type: <LINEAR_INT4_128 type index>
        raw_data: <64 bytes>
        name: "weight"
    }

The payload size is ``128 * 4 / 8 = 64`` bytes. Constants are stored in the
type and do not contribute to that size.

Validation
++++++++++

A checker rejects:

* an invalid or cyclic type reference;
* a field without exactly one of ``type`` and ``constant``;
* duplicate field or component names;
* zero component widths or unsupported physical leaf types;
* a physical size that is not byte-aligned;
* a payload whose length differs from the computed size;
* implicit padding or untyped trailing bytes.
