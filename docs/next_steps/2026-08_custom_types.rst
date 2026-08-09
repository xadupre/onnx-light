.. _l-next-steps-custom-types:

Structured views over byte buffers
==================================

:Date: 2026-08

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
        repeated uint64 dims = 7;  // values of the declared dimension parameters
    }

The container itself is outside the scope of this page. The purpose of the
specification is to define ``StructTypeProto``: a portable structure
that can be overlaid on the bytes of a ``StructProto``.

``dims`` is not a logical tensor shape. It holds the value of every dimension
parameter declared by the selected type, in declaration order. A type whose
repetition counts are all constants declares no parameter and leaves ``dims``
empty. A type describing a variable number of elements, such as a tensor of a
custom element type, declares one parameter per variable repetition, and the
exact payload size is then a function of the type and of ``dims`` only. It is
never inferred from the payload length.

The type system adds three serialized structural kinds and constant fields:

* an array of a statically sized ``TypeProto``;
* a packed array of repeated named bit components;
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
* Every array and packed-array length is either a concrete non-negative
  integer or one dimension parameter supplied by the value.
* The number of payload bytes of a value is computable from its type and its
  ``dims``, without reading the payload.
* The physical structure is inspectable without loading a vendor plugin.
* An optional standard ONNX decoder defines logical semantics such as
  dequantization.

Stable contract
+++++++++++++++

The proposal has three valid uses of ``StructTypeProto``:

``concrete declaration``
    Selects ``array``, ``packed_array``, or ``structure``. It appears in
    ``ModelProto.struct_types`` or in
    ``StructProto.struct_type``. Together with ``StructProto.dims`` it
    completely determines the payload size.

``exact static reference``
    Selects ``type_index`` and appears inside ``TypeProto``. It accepts only
    the referenced model-level declaration.

``unconstrained static category``
    Leaves ``kind`` unset and appears only inside ``TypeProto``. It accepts
    any concrete structured declaration. This form is used by heterogeneous
    sequences and maps.

``type_index`` may also occur below a concrete root through
``Array.element_type`` or ``Structure.Field.type``; such a reference must
select a declaration without dimension parameters. A constant tensor value is
attached directly to a ``Structure.Field``. A concrete root may not
be a ``type_index`` or an unset ``kind``. Static forms may not carry
``decoder``, ``encoder``, ``name``, ``dims_param``, or metadata.

Only the ``decoder`` and ``encoder`` attached to the selected concrete root
are invoked. A declaration reached through a nested ``type_index`` contributes
only its physical structure and constants; its decoder and encoder are not
composed implicitly.

No other interpretation of an absent field is permitted. In particular,
counts are either constants or parameters bound by ``StructProto.dims``,
and there are no inferred lengths, implicit alignment, hidden padding,
semantic traits, or alternate byte orders.

Dimension parameters
++++++++++++++++++++

A repetition count is either a constant declared by the type or a parameter
supplied by the value. The concrete root declaration lists its parameter names
in ``dims_param``, in declaration order. Inside that declaration, an ``Array``
or a ``PackedArray`` selects ``dimension`` for a constant count or
``dims_index`` for a parametric one; ``dims_index`` is a position in
``dims_param`` and in ``StructProto.dims``.

The rules are:

* a declaration with an empty ``dims_param`` is *static*: its byte size
  depends on the type alone and every ``count`` selects ``dimension``;
* only a concrete root declares parameters; an inline nested declaration
  leaves ``dims_param`` empty and its ``dims_index`` values are resolved
  against the enclosing root;
* a declaration with a non-empty ``dims_param`` is *parametric*: it may be
  selected by a value or by an exact static reference inside ``TypeProto``,
  but never nested through ``Array.element_type`` or ``Structure.Field.type``,
  so parameters are never rebound and size computation stays local;
* every ``dims_index`` is smaller than the number of declared parameters and
  every declared parameter is used at least once;
* ``StructProto.dims`` has exactly one entry per declared parameter;
* a static declaration requires an empty ``StructProto.dims``;
* the same parameter may be used by several arrays, which forces their counts
  to be equal;
* dimension names are documentation and binding keys only; no expression,
  arithmetic, or constraint language is introduced.

Consequently the payload size of a value is a pure function of its type and
its ``dims``, and it is still an error when it differs from the declared
payload length. The value never determines a count from the buffer size.

A decoder output shape may use ``dim_param`` names taken from ``dims_param``.
A runtime binds each such name to the corresponding ``StructProto.dims``
entry. Any other ``dim_param`` stays symbolic and never participates in the
size computation.

Physical size function
++++++++++++++++++++++

The serialized size is computed recursively in bits. It takes the concrete
root declaration and the dimension values ``d = StructProto.dims``:

.. code-block:: text

    count(dimension=n)          = n
    count(dims_index=k)         = d[k]
    size(scalar(T))             = bit_width(T)
    size(Array(T, cnt))         = count(cnt) * size(T)
    size(PackedArray(c..., cnt))
                                = count(cnt) * sum(c.bit_width)
    size(Field(constant))       = 0
    size(Field(T))              = size(T)
    size(Structure(f...))       = sum(size(f))
    size(type_index=i)          = size(ModelProto.struct_types[i])

All arithmetic is checked in ``uint64``. References must be acyclic. The
concrete root size must be divisible by eight and equal the inline
``raw_data`` length or the external-data ``length``. Only ``dims`` enters the
computation besides the type, so the physical schema and payload remain
independently checkable.

StructTypeProto
+++++++++++++++++++++

The complete proposal adds one top-level structured type message.

.. code-block:: text

    message StructTypeProto {
        // Repeats one physical element a fixed or parametric number of times.
        message Array {
            TypeProto element_type = 1;  // repeated physical element
            oneof count {
                uint64 dimension = 2;    // exact element count
                uint32 dims_index = 3;   // index into dims_param
            }
        }

        // Repeats one explicitly decomposed bit pattern.
        message PackedArray {
            message Component {
                string name = 1;       // unique component name
                uint32 bit_width = 2;  // bits per element
            }

            repeated Component component = 1;  // bit order within each element
            oneof count {
                uint64 dimension = 2;   // exact element count
                uint32 dims_index = 3;  // index into dims_param
            }
        }

        // Names one component of a structure.
        message Field {
            string name = 1;        // unique within the structure
            oneof content {
                TypeProto type = 2;               // schema read from payload
                TensorProto constant = 4;         // tensor; reads no payload
            }
            string doc_string = 3;  // field documentation
        }

        // Concatenates fields in declaration order.
        message Structure {
            repeated Field field = 1;  // serialized in declaration order
        }

        // Defines, references, or constrains the structured type.
        oneof kind {
            Array array = 1;                // repeated typed elements
            Structure structure = 2;        // ordered named fields
            PackedArray packed_array = 3;   // repeated packed bit patterns
            int32 type_index = 5;           // ModelProto.struct_types index
        }

        optional FunctionProto decoder = 6;  // structured leaves to ONNX value
        optional FunctionProto encoder = 7;  // ONNX value to canonical bytes
        string name = 8;                     // reusable type name
        string doc_string = 9;               // type documentation
        repeated StringStringEntryProto metadata_props = 10;  // type metadata
        repeated string dims_param = 11;     // names of the dimension parameters
    }

``StructTypeProto`` is itself the new ``TypeProto`` branch; there is no
intermediate ``Layout`` message. A concrete model-level or inline physical
type selects exactly one of ``array``, ``packed_array``, and ``structure``.
A nested or static exact type may select one model-level
declaration with ``type_index``. A static type may leave ``kind`` unset only
for the unconstrained category defined above.

``StructProto.type`` and ``StructTypeProto.type_index`` are both
model-level indices. The former selects the type of one value; the latter
references a type from another physical declaration.

The corresponding ``StructProto`` values may appear in graph values,
sequences, maps, optionals, and attributes. ``AttributeProto`` adds
``STRUCT`` and ``STRUCTS`` categories for the singular and
repeated attribute forms described below.

.. code-block:: text

    message TypeProto {
        oneof value {
            ...
            StructTypeProto struct_type = <N>;
        }
    }

Type storage and references
+++++++++++++++++++++++++++

Reusable types are stored once at model level:

.. code-block:: text

    message ModelProto {
        ...
        repeated StructTypeProto struct_types = <N>;
    }

Shared types use a non-negative ``type`` index. The integer is local to the
model and is remapped by model composition tools. A value whose type is unique
sets ``type`` to -1 and stores it in ``struct_type`` instead, avoiding a
single-use entry in the model list.

The two forms are mutually exclusive:

* ``type >= 0`` requires an in-range model index and forbids
  ``struct_type``;
* ``type == -1`` requires ``struct_type``;
* every other negative value is invalid.

As with ``TensorProto``, ``StructProto.name`` identifies the concrete
value and its ``doc_string`` documents that value.
``StructTypeProto.name`` identifies the type declaration instead. It
must be non-empty and unique among ``ModelProto.struct_types`` entries;
it is optional for an inline type. ``StructTypeProto.doc_string``
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
A name such as ``FLOAT6_E3M2`` identifies a concrete structured physical
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
``struct_type`` referencing an entry in
``ModelProto.struct_types``. For a physical array element, that
``struct_type`` contains either one ``type_index`` or one inline
physical kind so its byte representation is unambiguous. Other ``TypeProto``
branches are invalid here unless ONNX defines a canonical fixed-width byte
representation for them.

``Array.dimension`` is a concrete non-negative integer. Its Protobuf default
is zero; it is never symbolic or inferred from the enclosing buffer. When the
array instead selects ``dims_index``, the count comes from
``StructProto.dims`` at that position. Consequently, the physical size of
every array is known from its type and from the ``dims`` of the value, and
from nothing else.

For readability, a fixed-width ONNX data type such as ``INT4`` denotes its
scalar ``TypeProto.Tensor`` leaf directly. ``array(T, dimension=d)`` denotes
one ``Array`` and ``array(T, dimensions=[d0, d1, ...])`` denotes nested
arrays. The outermost array has dimension ``d0`` and the innermost element is
``T``.

Array elements are tightly packed. An array requiring padding between values
uses a structure element with an explicit final padding field.

No physical dimension is derived from a decoder output.
``Array.dimension`` describes physical storage only.

Packed array type
+++++++++++++++++

``PackedArray`` repeats a bit pattern whose components are declared in order.
For each element, the first component occupies the least-significant bits,
followed immediately by the remaining components. Elements are consecutive
with no padding. Its size is the element count multiplied by the sum of the
component widths.

Every component name is unique and every ``bit_width`` is positive. A parser
exposes one unsigned integer leaf per component, grouped over
``PackedArray.dimension``. For readability,
``packed_array(4, [sign:1, exponent:3, mantissa:2])`` denotes four consecutive
six-bit patterns.

This is a physical packing construct, not a new ``TensorProto.DataType``.
Logical interpretation remains the responsibility of the enclosing type's
decoder.

Structure type
++++++++++++++

Physical fields are serialized consecutively in declaration order. The first
physical field starts at bit zero and every following physical field starts
immediately after the previous one. A constant field occupies no range and
does not change the offset of following fields. The structure size is
therefore the sum of its non-constant field sizes. Alignment and padding are
represented by ordinary named padding fields, so every serialized bit remains
explicit.

Constant fields
+++++++++++++++

``Structure.Field`` selects exactly one of ``type`` and ``constant``. A
physical field uses ``type`` and consumes payload bits. A tensor constant uses
an inline ``TensorProto`` and consumes zero payload bits.
A scalar tensor constant has rank zero, represented by an empty ``dims`` list.
The tensor must not use ``external_data``.

Only structure fields may be constants; roots and array elements remain
physical types. For readability, examples write
``tensor(INT32, [], value)`` for a scalar and
``tensor(FLOAT, shape, values)`` for a non-scalar constant.

Applying a type to a buffer
+++++++++++++++++++++++++++

Parsing starts with the selected ``array``, ``packed_array``, or ``structure``
kind at bit offset zero. A successful parse produces a tree
of typed views into the original buffer plus the constants embedded in the
type. Implementations should avoid copying byte-aligned fields and may lazily
unpack bit fields.

When ``external_data`` is empty, ``raw_data`` is the payload and may itself be
empty for a zero-sized physical type. When ``external_data`` is non-empty,
``raw_data`` must be empty and the external metadata must provide an exact
``length`` entry. The parse succeeds only if:

* ``raw_data.size()`` or external-data ``length`` equals the size computed
  from the physical type and ``dims``;
* ``dims`` has one entry per declared dimension parameter;
* the computed physical size is a whole number of bytes;
* arithmetic on dimensions and element sizes does not overflow ``uint64``;
* the root physical kind consumes the whole buffer, including explicit
  padding.

The last rule prevents untyped trailing bytes. A format that intentionally
contains an uninterpreted suffix must declare it as a ``UINT8`` array.

Logical leaf view
+++++++++++++++++

Every physical scalar ONNX ``TypeProto`` field and constant field is a leaf.
Unstructured arrays and structures only organize leaves. The canonical
leaf order is depth-first declaration order. A constant field contributes one
scalar or tensor leaf at its declaration position, does not advance the
current buffer position, and is not replicated by an enclosing array. It is
shared by every instance of the declaring type.

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
``StructTypeProto``. An encoder does not define calibration,
training policy, or parameter selection. Formats with several valid
encodings may omit it.
The encoder input type is declared by its ``ValueInfoProto`` and must equal
the decoder output type when both functions are present.
Fields with ``constant`` are type information and are not written to the
result buffer.

Static type and static data
+++++++++++++++++++++++++++

The type and the data are both static:

* ``StructTypeProto`` fixes field order, widths, interpretation, and every
  count that is not a declared parameter.
* ``StructProto.dims`` fixes the remaining counts.
* ``raw_data`` or external-data length fixes the concrete payload size, which
  must equal the size computed from the two previous items.

There is no symbolic relation to solve between logical and physical
dimensions. A dimension parameter is bound by a value, not inferred. The
checker validates the concrete overlay and the decoder validates the logical
interpretation.

TypeProto integration
++++++++++++++++++++++

``StructProto`` is a value category, not merely an initializer
encoding. ``TypeProto`` therefore contains ``StructTypeProto`` directly,
as shown above. This makes recursion uniform: ``Array.element_type`` and
``Field.type`` reuse ``TypeProto`` rather than an intermediate layout
language. A physical element or field must nevertheless have a canonical
fixed size; currently this permits standard tensor types and exact structured
array, packed-array, and structure types, but not sequences, maps, or
optionals.

The concrete ``StructProto`` selects exactly one
``StructTypeProto`` declaration. The static type is a constraint:

* when ``StructTypeProto.type_index`` is present, the value must
  reference that exact model-level declaration;
* when ``kind`` is unset, the explicitly unconstrained static category accepts
  any concrete structured type.

Model-level and ``StructProto.struct_type`` definitions must
select ``array``, ``packed_array``, or ``structure``. A
``TypeProto.struct_type`` may instead:

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

    sequence(struct(...))
    map(int64, struct(...))
    optional(struct(...))
    sequence(map(int64, struct(...)))

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
            STRUCT = <N>;
        }
        ...
        repeated StructProto struct_values = <N>;
        optional TypeProto value_type = <N+1>;
    }

    message OptionalProto {
        enum DataType {
            ...
            STRUCT = <N>;
        }
        ...
        optional StructProto struct_value = <N>;
        optional TypeProto value_type = <N+1>;
    }

``elem_type`` continues to identify the broad value category for compatibility.
``value_type`` carries the complete recursive static constraint and must agree
with it. It is required for ``STRUCT`` values. For graph values, it must
also agree with the corresponding ``ValueInfoProto``. This makes standalone
sequence/map attributes self-describing rather than relying on surrounding
graph type information.

All values in a sequence remain homogeneous with respect to their static
``TypeProto.Struct`` category. With the unconstrained static form, they
need not have the same concrete declaration: each value still carries its
exact physical type.

AttributeProto integration
++++++++++++++++++++++++++

Operators and functions may also need an structured value as an attribute.
``TYPE_PROTO`` transports only its type and cannot carry the serialized
payload. ``AttributeProto`` therefore gains singular and repeated value
categories, following the existing tensor and sparse-tensor pattern:

.. code-block:: text

    message AttributeProto {
        enum AttributeType {
            ...
            STRUCT = 15;
            STRUCTS = 16;
        }

        ...
        optional StructProto struct = 24;
        repeated StructProto structs = 25;
    }

``AttributeProto.type`` must match the populated field. Exactly one attribute
content field is allowed, as for every other attribute category. A
``ref_attr_name`` may refer to either new category from a function body; its
resolved parent attribute must have the same category.

FLOAT6_E3M2 example
+++++++++++++++++++

The following concrete type stores four six-bit floating-point values in
three bytes. Each value has one sign bit, three exponent bits, and two
mantissa bits. Format parameters are rank-zero ``TensorProto`` constants and
therefore consume no payload bytes.

.. code-block:: text

    StructTypeProto {
        name: "FLOAT6_E3M2"
        structure: Structure {
            field: {
                name: "packed"
                type: packed_array(
                    dimension=4,
                    components=[sign:1, exponent:3, mantissa:2]
                )
            }
            field: { name: "exponent_bias", constant: tensor(INT32, [], 3) }
            field: { name: "has_inf", constant: tensor(BOOL, [], true) }
            field: { name: "has_nan", constant: tensor(BOOL, [], true) }
        }
        decoder: FunctionProto {
            output: "Y"
            value_info: {
                name: "Y"
                type: TypeProto {
                    tensor_type: Tensor {
                        elem_type: FLOAT
                        shape: TensorShapeProto {
                            dim: { dim_value: 4 }
                        }
                    }
                }
            }
            // Unpacks four 6-bit values and decodes E3M2 with bias 3.
        }
    }

The physical size is determined without executing the decoder:
``4 * (1 + 3 + 2) = 24 bits``. The three constant fields contribute zero
bits. A ``StructProto`` selecting this declaration must therefore contain
exactly three payload bytes:

.. code-block:: text

    StructProto {
        type: <FLOAT6_E3M2 type index>
        raw_data: <3 bytes>
    }

Tensor of FLOAT6_E3M2
+++++++++++++++++++++

The previous declaration has a constant count, so it describes exactly four
values. A tensor of ``FLOAT6_E3M2`` needs a count that belongs to the value,
which is what a dimension parameter provides:

.. code-block:: text

    StructTypeProto {
        name: "FLOAT6_E3M2_TENSOR"
        dims_param: ["n_values"]
        packed_array: PackedArray {
            component: { name: "sign",     bit_width: 1 }
            component: { name: "exponent", bit_width: 3 }
            component: { name: "mantissa", bit_width: 2 }
            dims_index: 0
        }
        decoder: FunctionProto {
            output: "Y"
            value_info: {
                name: "Y"
                type: TypeProto {
                    tensor_type: Tensor {
                        elem_type: FLOAT
                        shape: TensorShapeProto {
                            dim: { dim_param: "n_values" }
                        }
                    }
                }
            }
            // Decodes E3M2 with bias 3.
        }
    }

    StructProto {
        type: <FLOAT6_E3M2_TENSOR type index>
        dims: [1000]
        raw_data: <750 bytes>
    }

The size is ``1000 * 6 = 6000`` bits, that is ``750`` bytes, and four values
occupy three bytes as before. It is computed from the type and ``dims``
without reading the payload and without executing the decoder.

Six bits do not tile a byte, so a value is rejected when
``n_values * 6`` is not a multiple of eight. Two portable options exist:

* restrict the parameter to a byte-aligned unit by counting groups instead of
  values, for instance an array of the four-value ``FLOAT6_E3M2`` declaration
  with ``dims_index: 0``, where ``dims: [250]`` also gives ``750`` bytes;
* declare the padding explicitly as an additional structure field, which is
  possible only when its width is a constant.

The first option is the recommended profile for sub-byte element types,
because every count then addresses a whole number of bytes.

Tensor of 2D points
+++++++++++++++++++

A record element follows the same pattern. The point layout is static and
therefore shareable, while the number of points belongs to the value:

.. code-block:: text

    ModelProto {
        struct_types: [
            StructTypeProto {             // index 0: one point, 8 bytes
                name: "POINT2D"
                structure: Structure {
                    field: { name: "x", type: FLOAT }
                    field: { name: "y", type: FLOAT }
                }
            },
            StructTypeProto {             // index 1: n points
                name: "POINT2D_TENSOR"
                dims_param: ["n_points"]
                array: Array {
                    dims_index: 0
                    element_type: TypeProto {
                        struct_type: StructTypeProto {
                            type_index: 0
                        }
                    }
                }
            }
        ]
    }

    StructProto {
        type: 1
        dims: [1000]
        raw_data: <8000 bytes>
    }

The canonical leaf view contains ``x[1000]`` and ``y[1000]``. No decoder is
required when an operator consumes those leaves directly; a decoder producing
``FLOAT[n_points, 2]`` may be added when a single tensor is preferred.

A two-dimensional grid of points uses one parameter per axis:

.. code-block:: text

    StructTypeProto {
        name: "POINT2D_GRID"
        dims_param: ["height", "width"]
        array: Array {
            dims_index: 0
            element_type: TypeProto {
                struct_type: StructTypeProto {
                    array: Array {
                        dims_index: 1
                        element_type: TypeProto {
                            struct_type: StructTypeProto { type_index: 0 }
                        }
                    }
                }
            }
        }
    }

    StructProto {
        type: <POINT2D_GRID type index>
        dims: [4, 5]
        raw_data: <160 bytes>
    }

``POINT2D`` stays static and is shared by both declarations. The parametric
declarations are roots, so ``4 * 5 * 8 = 160`` bytes is obtained without any
parameter substitution across declarations.

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
     - tensor or structured array
     - cast or reshape
   * - Structured block
     - explicit block structure
     - lookup, scaling, and scatter

These names are profiles and documentation conventions, not variants in a
protobuf ``oneof``.

Paged KV-cache
++++++++++++++

A paged KV-cache is naturally a sequence or map of structured pages. Each
page may use a different quantization type while satisfying one common static
constraint:

.. code-block:: text

    TypeProto {
        sequence_type: Sequence {
            elem_type: TypeProto {
                struct_type: StructTypeProto {}
            }
        }
    }

The first logical dimension contains K and V. Equivalent separate K and V
sequences are also valid.

A concrete sequence may mix page encodings:

.. code-block:: text

    ModelProto {
        struct_types: [
            StructTypeProto {
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
            StructTypeProto {
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
        elem_type: STRUCT
        value_type: TypeProto {
            struct_type: StructTypeProto {}
        }
        struct_values: [
            StructProto {
                type: 0
                raw_data: ...
            },
            StructProto {
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
``StructProto.type`` still selects one exact physical declaration.

For page lookup by identifier, ``MapProto`` uses integer keys and this
sequence as its values:

.. code-block:: text

    map(int64, struct)

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
        struct_types: [
            StructTypeProto {             // index 0: one physical block
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
            StructTypeProto {             // index 1: complete value
                name: "STQ1_0"
                array: Array {
                    dimension: 10
                    element_type: TypeProto {
                        struct_type: StructTypeProto {
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
        struct_types: [
            StructTypeProto {             // index 0: one block
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
            StructTypeProto {             // index 1: all blocks
                name: "INT4_BLOCKWISE"
                array: Array {
                    dimension: 10
                    element_type: TypeProto {
                        struct_type: StructTypeProto {
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
        struct_types: [
            StructTypeProto {             // index 0: one node
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
            StructTypeProto {             // index 1: complete tree
                name: "DECISION_TREE_7_NODES_3_CLASSES"
                structure: Structure {
                    field: {
                        name: "nodes"
                        type: array(
                            struct(type_index=0),
                            dimension=7
                        )
                    }
                    field: {
                        name: "class_ids"
                        constant: tensor(INT64, [3], [0, 1, 2])
                    }
                }
            }
        ]
    }

    StructProto {
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
     - Array of indexed nodes plus a constant field
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
* ``type`` and ``struct_type`` satisfy the -1 sentinel rules;
* the computed physical size equals the inline or external payload length;
* every ``dims_index`` is in range and every declared parameter is used;
* ``StructProto.dims`` has one entry per declared parameter of its type;
* a parametric declaration is never nested inside another physical type;
* model-level and inline concrete types select exactly one ``array``,
  ``packed_array``, or ``structure`` kind;
* ``type_index`` references are in range and do not form cycles;
* every structure field selects exactly one of ``type`` and ``constant``;
* valid standard ONNX leaf types;
* constant tensors are inline, concretely shaped, and have no external data;
* unique field names within each structure;
* unique packed-component names and positive component widths;
* valid concrete physical dimensions;
* every physical field and array element has a canonical fixed size;
* the total physical size is divisible by eight bits;
* exact consumption of the concrete buffer;
* decoder and encoder signatures;
* decoder restrictions and represented output type;
* every structured value satisfies its exact or unconstrained static
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
     - Payload length checked against the physical type
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

``StructTypeProto`` is therefore a schema for bytes, not an opaque
escape hatch and not a closed list of application-specific formats.
