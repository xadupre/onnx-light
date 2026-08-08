.. _l-next-steps-custom-types:

Structured views over byte buffers
==================================

Motivation
++++++++++

``TypeProto.Opaque`` identifies a runtime-owned value by domain and name,
but it gives no information about its serialized representation. A generic
reader cannot determine how many values are present, where fields begin,
or how many bytes may safely be read.

Conversely, adding one protobuf message for every quantization or custom
format creates a closed hierarchy that must grow whenever a new layout is
introduced.

This proposal assumes that an ``UnstructuredProto`` already exists. It owns
or references a byte buffer, gives its exact byte size, and references its
physical type:

.. code-block:: text

    message UnstructuredProto {
        int32 type = 1;  // model-level index, or -1 when unstructured_type is present
        optional UnstructuredTypeProto unstructured_type = 2;
        bytes raw_data = 3;
        repeated StringStringEntryProto external_data = 4;
        string name = 5;
        string doc_string = 6;
    }

The container itself is outside the scope of this page. The purpose of the
specification is to define ``UnstructuredTypeProto``: a portable structure
that can be overlaid on the bytes of an ``UnstructuredProto``.

``dims`` is intentionally absent because not every unstructured value is
tensor-shaped. Physical dimensions are already part of its arrays and tensor
fields.

The type system adds two serialized structural kinds and one type-level
constant leaf:

* an array of a statically sized ``TypeProto``;
* a structure containing named, statically sized ``TypeProto`` fields;
* a constant ``TensorProto`` that consumes no payload bytes.

Scalars and ordinary tensors continue to use ``TypeProto.Tensor``. Quantized
values, packed records, custom numeric types, image pixels, and other static
binary formats are recursive compositions of existing ONNX types and these
additions.

Requirements
++++++++++++

* The buffer size is implied by the physical type and must equal the inline
  or external payload length.
* Every read performed by the structured view is bounds-checked.
* Bits and multi-byte values use one canonical ordering convention.
* A structure may be nested and repeated without introducing a new proto
  for each format.
* Every array length is a concrete non-negative integer.
* The physical structure is inspectable without loading a vendor plugin.
* An optional standard ONNX decoder defines logical semantics such as
  dequantization.

Stable contract
+++++++++++++++

The proposal has three valid uses of ``UnstructuredTypeProto``:

``concrete declaration``
    Selects ``array`` or ``structure``. It appears in
    ``ModelProto.unstructured_types`` or in
    ``UnstructuredProto.unstructured_type``. It completely determines the
    payload size.

``exact static reference``
    Selects ``type_index`` and appears inside ``TypeProto``. It accepts only
    the referenced model-level declaration.

``unconstrained static category``
    Leaves ``kind`` unset and appears only inside ``TypeProto``. It accepts
    any concrete unstructured declaration. This form is used by heterogeneous
    sequences and maps.

``type_index`` may also occur below a concrete root through
``Array.element_type`` or ``Structure.Field.type``. A constant value is
attached directly to a ``Structure.Field``. A concrete root may not be a
``type_index`` or an unset ``kind``. Static forms may not carry ``decoder``,
``encoder``, ``name``, or metadata.

Only the ``decoder`` and ``encoder`` attached to the selected concrete root
are invoked. A declaration reached through a nested ``type_index`` contributes
only its physical structure and constants; its decoder and encoder are not
composed implicitly.

No other interpretation of an absent field is permitted. In particular,
there are no symbolic physical dimensions, inferred lengths, implicit
alignment, hidden padding, semantic traits, or alternate byte orders.

Physical size function
++++++++++++++++++++++

The serialized size is computed recursively in bits:

.. code-block:: text

    size(scalar(T))       = bit_width(T)
    size(Array(T, n))     = n * size(T)
    size(Field(T, constant)) = 0
    size(Field(T))        = size(T)
    size(Structure(f...)) = sum(size(f))
    size(type_index=i)    = size(ModelProto.unstructured_types[i])

All arithmetic is checked in ``uint64``. References must be acyclic. The
concrete root size must be divisible by eight and equal the inline
``raw_data`` length or the external-data ``length``. These rules make the
physical schema and payload independently checkable.

UnstructuredTypeProto
+++++++++++++++++++++

The complete proposal is one top-level message with nested structural
messages.

.. code-block:: text

    message UnstructuredTypeProto {
        // Repeats one physical element a fixed number of times.
        message Array {
            TypeProto element_type = 1;  // repeated physical element
            uint64 dimension = 2;        // exact element count
        }

        // Names one component of a structure.
        message Field {
            string name = 1;                 // unique within the structure
            TypeProto type = 2;              // field type
            string doc_string = 3;           // field documentation
            optional TensorProto constant = 4;  // value absent from payload
        }

        // Concatenates fields in declaration order.
        message Structure {
            repeated Field field = 1;  // serialized in declaration order
        }

        // Defines, references, or constrains the structured type.
        oneof kind {
            Array array = 1;          // repeated elements
            Structure structure = 2;  // ordered named fields
            int32 type_index = 4;     // ModelProto.unstructured_types index
        }

        optional FunctionProto decoder = 5;  // structured leaves to ONNX value
        optional FunctionProto encoder = 6;  // ONNX value to canonical bytes
        string name = 7;                     // reusable type name
        string doc_string = 8;               // type documentation
        repeated StringStringEntryProto metadata_props = 9;  // type metadata
    }

``UnstructuredTypeProto`` is itself the new ``TypeProto`` branch; there is no
intermediate ``Layout`` message. A concrete model-level or inline physical
type selects exactly one of ``array`` and ``structure``. A nested or static
exact type may select one model-level declaration with ``type_index``. A
static type may leave ``kind`` unset only for the unconstrained category
defined above.

``UnstructuredProto.type`` and ``UnstructuredTypeProto.type_index`` are both
model-level indices. The former selects the type of one value; the latter
references a type from another physical declaration.

The corresponding ``UnstructuredProto`` values may appear in graph values,
sequences, maps, optionals, and attributes. ``AttributeProto`` adds
``UNSTRUCTURED`` and ``UNSTRUCTUREDS`` categories for the singular and
repeated attribute forms described below.

.. code-block:: text

    message TypeProto {
        oneof value {
            ...
            UnstructuredTypeProto unstructured_type = <N>;
        }
    }

Type storage and references
+++++++++++++++++++++++++++

Reusable types are stored once at model level:

.. code-block:: text

    message ModelProto {
        ...
        repeated UnstructuredTypeProto unstructured_types = <N>;
    }

Shared types use a non-negative ``type`` index. The integer is local to the
model and is remapped by model composition tools. A value whose type is unique
sets ``type`` to -1 and stores it in ``unstructured_type`` instead, avoiding a
single-use entry in the model list.

The two forms are mutually exclusive:

* ``type >= 0`` requires an in-range model index and forbids
  ``unstructured_type``;
* ``type == -1`` requires ``unstructured_type``;
* every other negative value is invalid.

As with ``TensorProto``, ``UnstructuredProto.name`` identifies the concrete
value and its ``doc_string`` documents that value.
``UnstructuredTypeProto.name`` identifies the type declaration instead. It
must be non-empty and unique among ``ModelProto.unstructured_types`` entries;
it is optional for an inline type. ``UnstructuredTypeProto.doc_string``
documents that type.

Standard ONNX leaves
++++++++++++++++++++

There is no custom scalar kind. Primitive leaves use a scalar
``TypeProto.Tensor`` with an empty shape. Every repetition, including a
multidimensional one, uses ``Array`` rather than a shaped tensor leaf.
Physical leaves require a fixed-width element type. This includes
``FLOAT16``, ``BFLOAT16``, the float8 formats,
``FLOAT4E2M1``, ``INT4``, ``UINT4``, ``INT2``, ``UINT2``, and ``BOOL``.
``STRING`` is not a valid physical leaf. ``BOOL`` occupies one byte; it is
not a one-bit storage type.

A non-standard bit field is decomposed into standard ONNX fields and
interpreted by the decoder rather than introducing another scalar taxonomy.
A name such as ``FLOAT6_E3M2`` identifies a concrete unstructured physical
declaration; it is not a ``TypeProto.Tensor.elem_type``. In particular, a
generic ``IEEE_FLOAT`` tensor element type is prohibited: it fixes neither
the physical width nor the logical standard ONNX element type. The decoder of
every custom numeric encoding must declare its output using a standard ONNX
``elem_type``.

Storage uses one canonical convention:

* bytes are addressed in increasing buffer order;
* bit zero is the least significant bit of ``raw_data[0]``;
* consecutive bits increase from least to most significant within a byte;
* multi-byte integers and IEEE values are little-endian.

This matches ONNX ``TensorProto.raw_data``. A format with a different ordering
can declare raw byte fields and normalize them in its decoder.

The structured reader exposes every leaf using its declared ``TypeProto``.
An uninterpreted region is represented as an array of scalar ``UINT8`` leaves.

Array type
++++++++++

An array repeats an existing ONNX ``TypeProto``. A scalar element is a
scalar-shaped ``tensor_type``. A structured element is an
``unstructured_type`` referencing an entry in
``ModelProto.unstructured_types``. For a physical array element, that
``unstructured_type`` contains either one ``type_index`` or one inline
physical kind so its byte representation is unambiguous. Other ``TypeProto``
branches are invalid here unless ONNX defines a canonical fixed-width byte
representation for them.

``Array.dimension`` is a concrete non-negative integer. Its Protobuf default
is zero; it is never symbolic or inferred from the enclosing buffer.
Consequently, the physical size of every array is known from its type alone.

For readability, a fixed-width ONNX data type such as ``INT4`` denotes its
scalar ``TypeProto.Tensor`` leaf directly. ``array(T, dimension=d)`` denotes
one ``Array`` and ``array(T, dimensions=[d0, d1, ...])`` denotes nested
arrays. The outermost array has dimension ``d0`` and the innermost element is
``T``.

Array elements are tightly packed. An array requiring padding between values
uses a structure element with an explicit final padding field.

No physical dimension is derived from a decoder output.
``Array.dimension`` describes physical storage only.

Structure type
++++++++++++++

Fields are serialized consecutively in declaration order. The first field
starts at bit zero and every following field starts immediately after the
previous one. The structure size is therefore the sum of its field sizes.
Alignment and padding are represented by ordinary named padding fields, so
every serialized bit remains explicit.

Constant fields
+++++++++++++++

``Structure.Field.constant`` stores a constant without replacing its declared
``type``. For example, a scalar parameter is written as
``type: INT32, constant: sign_bits``, not as a distinct constant type. A field
with ``constant`` contributes a typed leaf but consumes zero bits from the
``UnstructuredProto`` payload. This keeps codebooks, fixed scales, defaults,
and other shared decoder inputs explicit without complicating the type tree.

The value is encoded as an inline ``TensorProto`` whose type and concrete
shape must exactly match ``Field.type``. It must not use ``external_data``.
Only structure fields may carry values; roots and array elements remain
physical types. Examples use the concise ``constant: ...`` notation and omit
the underlying ``TensorProto`` encoding.

Applying a type to a buffer
+++++++++++++++++++++++++++

Parsing starts with the selected ``array`` or ``structure`` kind at bit offset
zero. A successful parse produces a tree of typed views into the original
buffer plus the constants embedded in the type. Implementations should avoid
copying byte-aligned fields and may lazily unpack bit fields.

When ``external_data`` is empty, ``raw_data`` is the payload and may itself be
empty for a zero-sized physical type. When ``external_data`` is non-empty,
``raw_data`` must be empty and the external metadata must provide an exact
``length`` entry. The parse succeeds only if:

* the payload length equals the size computed from the physical type;
* the computed physical size is a whole number of bytes;
* arithmetic on dimensions and element sizes does not overflow ``uint64``;
* the root physical kind consumes the whole buffer, including explicit
  padding.

The last rule prevents untyped trailing bytes. A format that intentionally
contains an uninterpreted suffix must declare it as a ``UINT8`` array.

Logical leaf view
+++++++++++++++++

Every scalar ONNX ``TypeProto`` field and every field with ``constant`` is a
leaf.
Unstructured arrays and structures only organize leaves. The canonical leaf
order is depth-first declaration order. Constant fields occur in that order
but do not advance the current buffer position.

When a repeated structure contains an array field, corresponding scalar
leaves are grouped into one decoder input. For example, an array of ten
blocks containing ``values[32]`` and one scalar ``scale`` produces:

.. code-block:: text

    values: tensor(...)[10, 32]
    scale:  tensor(...)[10]

This canonical view gives portable decoders stable inputs without exposing
the in-memory representation of a parser implementation.

Decoder contract
++++++++++++++++

The optional decoder maps the physical leaf view to the represented ONNX
value:

.. code-block:: text

    Decode(
        leaf_0,
        ...,
        leaf_N
    ) -> value

Leaf inputs follow canonical depth-first order and have mandatory
``ValueInfoProto`` type information. The decoder:

* has no captures;
* calls only deterministic standard-domain ONNX operators;
* does not call custom or model-local functions;
* receives shared tensor constants through declared fields with ``constant``
  rather than hidden initializers;
* does not use external data or graph-valued attributes;
* has exactly one output whose type is declared by a corresponding
  ``ValueInfoProto`` in the function.

The decoder is the semantic oracle. A runtime plugin may replace it with an
optimized decoder or fused kernel, but a plugin is not required for
correctness.

If no decoder is present, the type defines only a physical structured view.
This is useful for inspection and for custom operators that consume the
fields directly.

Encoder contract
++++++++++++++++

An optional encoder defines one canonical serialization:

.. code-block:: text

    Encode(
        value
    ) -> buffer: tensor(UINT8)[serialized_size]

The result must parse successfully with the same
``UnstructuredTypeProto``. An encoder does not define calibration,
training policy, or parameter selection. Formats with several valid
encodings may omit it.
The encoder input type is declared by its ``ValueInfoProto`` and must equal
the decoder output type when both functions are present.
Fields with ``constant`` are type information and are not written to the
result buffer.

Static type and static data
+++++++++++++++++++++++++++

The type and the data are both static:

* ``UnstructuredTypeProto`` fixes field order, widths, counts, and
  interpretation.
* ``raw_data`` or external-data length fixes the concrete payload size.

There is no symbolic relation to solve between logical and physical
dimensions. The checker validates the concrete overlay and the decoder
validates the logical interpretation.

TypeProto integration
++++++++++++++++++++++

``UnstructuredProto`` is a value category, not merely an initializer
encoding. ``TypeProto`` therefore contains ``UnstructuredTypeProto`` directly,
as shown above. This makes recursion uniform: ``Array.element_type`` and
``Field.type`` reuse ``TypeProto`` rather than an intermediate layout
language. A physical element or field must nevertheless have a canonical
fixed size; currently this permits standard tensor types and exact
unstructured types, but not sequences, maps, or optionals.

The concrete ``UnstructuredProto`` selects exactly one
``UnstructuredTypeProto`` declaration. The static type is a constraint:

* when ``UnstructuredTypeProto.type_index`` is present, the value must
  reference that exact model-level declaration;
* when ``kind`` is unset, the explicitly unconstrained static category accepts
  any concrete unstructured type.

Model-level and ``UnstructuredProto.unstructured_type`` definitions must
select ``array`` or ``structure``. A ``TypeProto.unstructured_type`` may
instead:

* select a physical kind to define one exact inline type;
* select one model-level declaration with ``type_index``;
* leave ``kind`` unset as the unconstrained category.

These forms are mutually exclusive. This distinction permits both exact and
heterogeneous types. A graph input may require one exact type, while a
container using the unconstrained category may contain different physical
types.

Because ``Sequence``, ``Map``, and ``Optional`` already refer recursively to
``TypeProto``, no special container type grammar is needed:

.. code-block:: text

    sequence(unstructured(...))
    map(int64, unstructured(...))
    optional(unstructured(...))
    sequence(map(int64, unstructured(...)))

The usual ONNX map-key restrictions remain unchanged.

Container values
++++++++++++++++

The value protos must also transport the new category. ``MapProto`` already
stores its values in a ``SequenceProto``, so extending sequences also extends
maps:

.. code-block:: text

    message SequenceProto {
        enum DataType {
            ...
            UNSTRUCTURED = <N>;
        }
        ...
        repeated UnstructuredProto unstructured_values = <N>;
        optional TypeProto value_type = <N+1>;
    }

    message OptionalProto {
        enum DataType {
            ...
            UNSTRUCTURED = <N>;
        }
        ...
        optional UnstructuredProto unstructured_value = <N>;
        optional TypeProto value_type = <N+1>;
    }

``elem_type`` continues to identify the broad value category for compatibility.
``value_type`` carries the complete recursive static constraint and must agree
with it. It is required for ``UNSTRUCTURED`` values. For graph values, it must
also agree with the corresponding ``ValueInfoProto``. This makes standalone
sequence/map attributes self-describing rather than relying on surrounding
graph type information.

All values in a sequence remain homogeneous with respect to their static
``TypeProto.Unstructured`` category. With the unconstrained static form, they
need not have the same concrete declaration: each value still carries its
exact physical type.

AttributeProto integration
++++++++++++++++++++++++++

Operators and functions may also need an unstructured value as an attribute.
``TYPE_PROTO`` transports only its type and cannot carry the serialized
payload. ``AttributeProto`` therefore gains singular and repeated value
categories, following the existing tensor and sparse-tensor pattern:

.. code-block:: text

    message AttributeProto {
        enum AttributeType {
            ...
            UNSTRUCTURED = 15;
            UNSTRUCTUREDS = 16;
        }

        ...
        optional UnstructuredProto unstructured = 24;
        repeated UnstructuredProto unstructureds = 25;
    }

``AttributeProto.type`` must match the populated field. Exactly one attribute
content field is allowed, as for every other attribute category. A
``ref_attr_name`` may refer to either new category from a function body; its
resolved parent attribute must have the same category.

Quantization profile
++++++++++++++++++++

A quantized buffer provides a physical type plus a decoder. The former
quantization families become layout compositions:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Family
     - Physical layout
     - Decoder
   * - Linear
     - values, scales, zero-points
     - ``(q - zero_point) * scale``
   * - Codebook
     - packed indices and codebook fields
     - scalar or vector lookup
   * - Floating point
     - sign, exponent, and mantissa bit fields
     - floating-point reconstruction
   * - Sparse
     - dense section plus counted outlier fields
     - dense decode followed by replacement
   * - Logarithmic
     - sign and exponent fields
     - exponential reconstruction
   * - Tiled or blocked
     - nested arrays and structures
     - reshape and tile placement
   * - Cast or identity
     - tensor or unstructured array
     - cast or reshape
   * - Structured block
     - explicit block structure
     - lookup, scaling, and scatter

These names are profiles and documentation conventions, not variants in a
protobuf ``oneof``.

Paged KV-cache
++++++++++++++

A paged KV-cache is naturally a sequence or map of unstructured pages. Each
page may use a different quantization type while satisfying one common static
constraint:

.. code-block:: text

    TypeProto {
        sequence_type: Sequence {
            elem_type: TypeProto {
                unstructured_type: UnstructuredTypeProto {}
            }
        }
    }

The first logical dimension contains K and V. Equivalent separate K and V
sequences are also valid.

A concrete sequence may mix page encodings:

.. code-block:: text

    ModelProto {
        unstructured_types: [
            UnstructuredTypeProto {
                name: "INT4_KV_PAGE"
                structure: Structure {
                    field: {
                        name: "values"
                        type: array(INT4, dimensions=[2, 32, 128, 128])
                    }
                    field: {
                        name: "scale"
                        type: array(FLOAT16, dimensions=[2, 32, 128, 1])
                    }
                }
                decoder: FunctionProto {
                    output: "Y"
                    value_info: {
                        name: "Y"
                        type: TypeProto {
                            tensor_type: Tensor {
                                elem_type: FLOAT16
                                shape: TensorShapeProto {
                                    dim: { dim_value: 2 }
                                    dim: { dim_value: 32 }
                                    dim: { dim_value: 128 }
                                    dim: { dim_value: 128 }
                                }
                            }
                        }
                    }
                    // Y = Cast(values, FLOAT16) * scale
                }
            },
            UnstructuredTypeProto {
                name: "FP8_E4M3_KV_PAGE"
                structure: Structure {
                    field: {
                        name: "values"
                        type: array(FLOAT8E4M3FN, dimensions=[2, 32, 128, 128])
                    }
                }
                decoder: FunctionProto {
                    output: "Y"
                    value_info: {
                        name: "Y"
                        type: TypeProto {
                            tensor_type: Tensor {
                                elem_type: FLOAT16
                                shape: TensorShapeProto {
                                    dim: { dim_value: 2 }
                                    dim: { dim_value: 32 }
                                    dim: { dim_value: 128 }
                                    dim: { dim_value: 128 }
                                }
                            }
                        }
                    }
                    // Y = Cast(values, FLOAT16)
                }
            }
        ]
    }

    SequenceProto {
        elem_type: UNSTRUCTURED
        value_type: TypeProto {
            unstructured_type: UnstructuredTypeProto {}
        }
        unstructured_values: [
            UnstructuredProto {
                type: 0
                raw_data: ...
            },
            UnstructuredProto {
                type: 1
                raw_data: ...
            }
        ]
    }

Both decoder signatures produce ``FLOAT16`` with the same logical page shape.
Their physical layouts and byte sizes may differ.

The INT4 page contains ``2 * 32 * 128 * 128`` four-bit values
(``524288`` bytes) and ``2 * 32 * 128`` FLOAT16 scales (``16384`` bytes), for
an exact payload of ``540672`` bytes. The FP8 page contains
``2 * 32 * 128 * 128`` one-byte values, for ``1048576`` bytes. The sequence
accepts both because its static element category is unconstrained, while each
``UnstructuredProto.type`` still selects one exact physical declaration.

For page lookup by identifier, ``MapProto`` uses integer keys and this
sequence as its values:

.. code-block:: text

    map(int64, unstructured)

An attention runtime may dispatch each page by its resolved model type and
fuse decoding with attention. A generic runtime can invoke each declaration's
decoder. Page eviction, allocation, and mutation policy remain runtime concerns
rather than properties of the serialized type.

STQ1_0 example
++++++++++++++

STQ1_0 stores 256 logical values in each 42-byte block:

* 64 four-bit codes;
* 16 four-bit words packing 64 sign bits;
* one FLOAT16 scale.

.. code-block:: text

    ModelProto {
        unstructured_types: [
            UnstructuredTypeProto {             // index 0: one physical block
                name: "STQ1_0_BLOCK"
                structure: Structure {
                    field: {
                        name: "code"
                        type: array(UINT4, dimension=64)
                    }
                    field: {
                        name: "packed_sign"
                        type: array(UINT4, dimension=16)
                    }
                    field: {
                        name: "scale"
                        type: FLOAT16
                    }
                }
            },
            UnstructuredTypeProto {             // index 1: complete value
                name: "STQ1_0"
                array: Array {
                    dimension: 10
                    element_type: TypeProto {
                        unstructured_type: UnstructuredTypeProto {
                            type_index: 0
                        }
                    }
                }
                decoder: FunctionProto {
                    output: "Y"
                    value_info: {
                        name: "Y"
                        type: TypeProto {
                            tensor_type: Tensor {
                                elem_type: FLOAT
                                shape: TensorShapeProto {
                                    dim: { dim_value: 2560 }
                                }
                            }
                        }
                    }
                    // sign = unpack four bits from each packed_sign value
                    // index = code + 16 * sign
                    // vector = ternary_codebook[index]
                    // scatter four values with stride 16
                    // Y = vector * scale
                }
            }
        ]
    }

For a 420-byte buffer, the root array contains exactly ten blocks and
describes 2560 logical values. Its complete structure and expected byte size
are derived exclusively from the static type:

.. code-block:: text

    block = 64 * 4 bits + 16 * 4 bits + 16 bits = 336 bits = 42 bytes
    value = 10 * 42 bytes = 420 bytes

Linear block example
++++++++++++++++++++

An INT4 format with 32 values followed by one FLOAT16 scale per block is:

.. code-block:: text

    ModelProto {
        unstructured_types: [
            UnstructuredTypeProto {             // index 0: one block
                name: "INT4_BLOCK_32"
                structure: Structure {
                    field: {
                        name: "values"
                        type: array(INT4, dimension=32)
                    }
                    field: {
                        name: "scale"
                        type: FLOAT16
                    }
                }
            },
            UnstructuredTypeProto {             // index 1: all blocks
                name: "INT4_BLOCKWISE"
                array: Array {
                    dimension: 10
                    element_type: TypeProto {
                        unstructured_type: UnstructuredTypeProto {
                            type_index: 0
                        }
                    }
                }
                decoder: FunctionProto {
                    output: "Y"
                    value_info: {
                        name: "Y"
                        type: TypeProto {
                            tensor_type: Tensor {
                                elem_type: FLOAT16
                                shape: TensorShapeProto {
                                    dim: { dim_value: 320 }
                                }
                            }
                        }
                    }
                    // Y = values * scale
                }
            }
        ]
    }

The canonical leaf view contains ``values[10, 32]`` and ``scale[10]``. A
linear decoder reconstructs ``values * scale``.

Decision-tree example
+++++++++++++++++++++

A finite decision tree does not require a recursive physical type. Child
relations are stored as node indices. The following three-class tree has
seven fixed-size nodes:

.. code-block:: text

    ModelProto {
        unstructured_types: [
            UnstructuredTypeProto {             // index 0: one node
                name: "DECISION_NODE_3_CLASSES"
                structure: Structure {
                    field: { name: "kind",       type: UINT8 }
                    field: { name: "feature_id", type: INT64 }
                    field: { name: "threshold",  type: FLOAT }
                    field: { name: "left",       type: INT32 }
                    field: { name: "right",      type: INT32 }
                    field: {
                        name: "value"
                        type: array(FLOAT, dimension=3)
                    }
                }
            },
            UnstructuredTypeProto {             // index 1: complete tree
                name: "DECISION_TREE_7_NODES_3_CLASSES"
                structure: Structure {
                    field: {
                        name: "nodes"
                        type: array(
                            unstructured(type_index=0),
                            dimension=7
                        )
                    }
                    field: {
                        name: "class_ids"
                        type: array(INT64, dimension=3)
                        constant: [0, 1, 2]
                    }
                }
            }
        ]
    }

    UnstructuredProto {
        type: 1
        raw_data: ...                 // exactly 231 bytes
    }

One node occupies ``1 + 8 + 4 + 4 + 4 + 3 * 4 = 33`` bytes, so the seven
serialized nodes occupy ``231`` bytes. ``class_ids`` is embedded in the type
and consumes no payload bytes. ``left`` and ``right`` use node indices or a
profile-defined sentinel for leaves. No decoder is required when a tree
operator consumes the canonical leaf view directly.

Reference-case validation
+++++++++++++++++++++++++

The three reference cases exercise distinct requirements:

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Case
     - Physical composition
     - Exact payload
     - Additional rule
   * - STQ1_0
     - Array of structured sub-byte blocks
     - 420 bytes
     - Decoder output is ``FLOAT[2560]``
   * - Paged KV-cache
     - Heterogeneous sequence of structures
     - 540672 or 1048576 bytes per page
     - Both decoders output the same page type
   * - Decision tree
     - Array of indexed nodes plus a constant
     - 231 bytes
     - Child indices are in ``[-1, 6]``

A conforming checker must accept these sizes without inspecting decoder
implementation details. It must reject any truncated or oversized payload,
cyclic ``type_index``, out-of-range tree child, or mismatched decoder
signature. The tree child-index rule and equal KV decoder signatures are
profile validation layered on top of the generic structural checker.

Validation and security
+++++++++++++++++++++++

A checker validates:

* non-negative type indices are in range;
* ``type`` and ``unstructured_type`` satisfy the -1 sentinel rules;
* model-level and inline concrete types select exactly one ``array`` or
  ``structure`` kind;
* ``type_index`` references are in range and do not form cycles;
* valid standard ONNX leaf types;
* inline, concretely shaped field values without external data and matching
  their declared field type;
* unique field names within each structure;
* valid concrete physical dimensions;
* every physical field and array element has a canonical fixed size;
* the total physical size is divisible by eight bits;
* exact consumption of the concrete buffer;
* decoder and encoder signatures;
* decoder restrictions and represented output type;
* every unstructured value satisfies its exact or unconstrained static
  category inside graph inputs, outputs, sequences, maps, and optionals.

Implementations must impose configurable limits on nesting depth, field
count, array count, constant bytes, and total extracted leaves. Validation
must use checked integer arithmetic before creating views or allocating
represented values.

Comparison with Opaque
++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 32 18 28

   * - Property
     - ``Opaque``
     - Structured buffer type
   * - Explicit type identity
     - Yes
     - Model index or inline type
   * - Exact buffer size
     - No
     - Physical type
   * - Explicit ordered fields
     - No
     - Yes
   * - Bit widths and canonical numbering
     - No
     - Yes
   * - Nested and repeated structures
     - No
     - Yes
   * - Portable logical semantics
     - No
     - Optional decoder
   * - Generic bounds validation
     - No
     - Yes
   * - Plugin required for correctness
     - Yes
     - No, when a decoder is present

``UnstructuredTypeProto`` is therefore a schema for bytes, not an opaque
escape hatch and not a closed list of application-specific formats.
