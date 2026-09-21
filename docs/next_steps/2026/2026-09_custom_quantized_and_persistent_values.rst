.. _l-next-steps-prepared-values-and-persistent-state:
.. _l-next-steps-custom-quantized-persistent-values:

Custom, quantized, and persistent values
================================================================================

:Date: 2026-09
:Updated: 2026-09-20

**in progress**

Objective and consolidation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Covers three uses of values: custom structs, quantized representations, and
values retained between calls. Persistent state is declared by
``GraphProto.persistent_bindings``, an explicit mapping of model outputs
back to inputs, not a separate state system. The model is immutable once
bound to a session. Retaining and forwarding state buffers requires no
payload copy or model serialization. Qwen is the first consumer:
quantized weights and per-request KV values carried
between decode calls.

This page merges the former structured-types and mutable-cache proposals
into one contract, authoritative over :ref:`l-next-steps-quantization` and
:ref:`l-next-steps-graph-builder-quantized-tensor` where they conflict.
Only a small closed set of common quantized forms gets specialized proto
support. Prepacking, prepared-object identity, prepared-object persistence
and scheduling stay owned by :ref:`l-next-steps-prepared-execution`, whose completed work
(native fast-loading, allocator, session executor) is the foundation this
plan builds on. :ref:`l-next-steps-proto-inheritance` is independent and
not a prerequisite.

Existing foundations and missing integration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``onnx_core/runtime/memory/simple_tensor.h`` already supplies ordinary
``Tensor`` storage owners, borrowed views and allocation handles. Prepared
execution already owns prepared-object identity, publication, residency,
eviction and persistence. ``StructTypeProto`` and ``EncodedValueProto``
and their initial GraphBuilder integration are implemented. The remaining
work connects the graph's persistence declarations to these ownership
facilities and feeds retained outputs into the next call without copying
their payloads. The graph attribute described below is planned, not yet
implemented by the completed representation PRs.

Three independent decisions
+++++++++++++++++++++++++++

Logical meaning, physical representation and lifetime stay independent:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Axis
     - Examples
     - Contract
   * - Logical meaning
     - Dense tensor, affine quantization, codebook quantization, custom value
     - What a consumer computes: decoded type and shape when the value
       denotes a tensor.
   * - Physical representation
     - Dense bytes, blocked INT4 with scales, tiled FP32, custom records
     - Exact fields, buffers, bit layout and format identity, not
       inferred from dtype alone.
   * - Lifetime and access
     - Immutable session value, retained request cache, invocation workspace
     - Ownership, sharing, synchronization and release, not the
       numerical type.

A quantized value can use a conventional block layout or a custom
structure; a compiled representation can be quantized or floating point; a
retained input can hold a dense tensor or a structured value.

Representation model: a small quantized core plus generic structs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

One ``StructTypeProto`` describes structs; ``EncodedValueProto`` holds
their byte-encoded representation when a fixed physical layout exists,
alongside the built-in affine layout frozen by PR01:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Descriptor
     - Responsibility
   * - ``StructTypeProto``
     - Fields and their types, nested structs, arrays and bit packing; a
       struct need not have a fixed physical size. The statically-sized
       subset describes one byte-encoded element, using checked size
       arithmetic and payload-derived counts.
   * - ``EncodedValueProto``
     - One value container with optional logical tensor type/shape, a
       layout choice and owned or external payload with a known byte
       extent. Layout is either a built-in dense/affine form or a
       concrete ``StructTypeProto`` reference.

``EncodedValueProto`` replaces the separate ``StructProto``,
``QuantizedTensorProto`` and ``CompiledTensorProto`` proposals; existing
``TensorProto`` values keep working unmigrated via a common runtime view.
``Value`` permits custom records as well as tensors, grouping
independently owned field buffers via ``StructTypeProto`` rather than
inline pointers, covering packed weights and KV blocks alike. Affine
layout parameters are a small nested descriptor: source INT4 weights, a
kernel-specific INT4 form and an INT4 KV block share one container with
different layouts and lifetime bindings.

The initial specialized subset is frozen in PR01; additional built-in
forms require demonstrated common use and a proto-size review. **Common
quantization** (INT8 per-tensor/per-axis, INT4 blockwise affine) selects
the built-in affine layout; **other quantized formats** (codebooks,
non-linear quantization, mixed-bit blocks, vendor-specific layouts) use
the structured layout plus a versioned format identity and an explicit
decoder or registered consumer; **fully custom structures** use the same
mechanism, failing explicitly when a graph input has no consumer or
decoder. A registered native C++ type can bind a descriptor to a typed
view or an owned runtime object; zero-copy access requires proved
compatibility.

Per-weight scales and zero points belong in value storage, not the
catalogue; only true format constants belong to the type, serialized once
in the shared ``StructTypeProto`` declaration while each value stores
only codes, as shown below.

.. _l-next-steps-custom-types:

Struct types and byte-encodable layouts
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``StructTypeProto`` describes a struct's fields and types: a concrete
declaration selects ``array``, ``bit_packing`` or ``structure``, and each
field selects either a value type or a tensor constant. It may describe a
cache with tensor and sequence fields; fixed physical size is an
eligibility condition for byte encoding, not a requirement on every
struct.

PR01 freezes the following wire contract. Field numbers 1000--1099 are
reserved for local onnx-light extensions so future upstream ONNX fields
can continue using the low-numbered range. ``TypeProto.struct_type``,
``GraphProto.encoded_initializer`` and ``ModelProto.struct_types`` each
use field 1000 in their respective messages.

.. code-block:: text

    message AffineLayoutProto {
        TensorProto.DataType storage_type = 1;
        TensorProto scale = 2;
        optional TensorProto zero_point = 3;
        optional int64 axis = 4;
        optional uint64 block_size = 5;
    }

    message EncodedValueProto {
        oneof layout {
            AffineLayoutProto affine = 1;
            StructTypeProto struct_type = 2;  // exact reference or inline type
        }
        optional TypeProto logical_type = 3;
        bytes raw_data = 4;
        repeated StringStringEntryProto external_data = 5;
        optional TensorProto.DataLocation data_location = 6;
        string name = 7;
        string doc_string = 8;
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
            uint64 type_ref = 4;
        }
        optional FunctionProto decoder = 5;
        optional FunctionProto encoder = 6;
        string name = 7;
        string doc_string = 8;
        repeated StringStringEntryProto metadata_props = 9;
        optional uint64 type_id = 10;
    }

    message TypeProto {
        oneof value {
            // Existing alternatives remain unchanged.
            StructTypeProto struct_type = 1000;
        }
    }

    message GraphProto {
        // Existing fields remain unchanged.
        repeated EncodedValueProto encoded_initializer = 1000;
    }

    message ModelProto {
        // Existing fields remain unchanged.
        repeated StructTypeProto struct_types = 1000;
    }

The specialized affine branch is deliberately closed:

* ``storage_type`` is one of ``INT8``, ``UINT8``, ``INT4`` or ``UINT4``.
  ``raw_data`` or the external payload contains only row-major codes using
  the corresponding ``TensorProto.raw_data`` packing. INT4/UINT4 stores
  the first element in the low nibble and the second in the high nibble;
  an unused final high nibble is zero.
* ``scale`` is a scalar or parameter tensor with floating element type;
  ``zero_point`` is optional, has ``storage_type``, and defaults to zero.
* Omitting ``axis`` selects per-tensor quantization. Setting ``axis``
  selects per-axis parameters. ``block_size`` is valid only with an axis
  and selects blocked quantization along it.
* Parameter shapes, axis normalization, code packing and decoded values
  follow the corresponding ``QuantizeLinear``/``DequantizeLinear``
  contract. Other affine forms use a structured layout instead of
  extending this message.

``logical_type`` must be a tensor type with concrete dimensions for the
affine branch. For ``n`` logical elements, the code payload is exactly
``ceil(n * bit_width(storage_type) / 8)`` bytes. It may be omitted for a
custom struct without tensor semantics; when present, the decoder or
native consumer must produce that exact type and shape.

Catalogue and identity contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``type_id`` is a nonzero, model-scoped stable format identity. Each ID
has exactly one declaration in ``ModelProto.struct_types``; duplicate
IDs, unresolved references and reference cycles are invalid. Reordering
the catalogue does not change identity. Changing fields, constants,
physical layout, decoder or encoder requires a new ID.

A ``type_ref`` contains only the referenced ID: declaration fields,
metadata, codecs and another kind must be absent. An inline declaration
has no ``type_id`` and cannot be referenced. A ``StructTypeProto`` with
no kind remains an unconstrained category only inside ``TypeProto``; it
is never a declaration or encoded layout.

``GraphProto.encoded_initializer`` names graph constants. Names are
unique across dense, sparse and encoded initializers and may also appear
in graph inputs. A structured graph input or output uses
``TypeProto.struct_type``; tensor-only operators do not accept it without
an explicit decoder or registered consumer.

Payload and lifetime contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``raw_data`` and ``external_data`` are mutually exclusive.
``data_location == EXTERNAL`` requires ``external_data`` with
``location`` and explicit ``length``; ``offset`` is optional. Default
location uses ``raw_data``. An empty inline payload represents zero
records. Payload bounds and layout are validated before any decoder or
native callback runs.

Inline bytes are owned or borrowed with an owner token. External mapped
bytes retain their mapping owner. Runtime structs retain each field's
owner independently; serialized records never contain pointers, native
padding, vtables or process-local handles. On the persistent execution
path, an invocation-only ownerless borrow cannot become a returned output
or retained state: the producer must supply a lifetime owner or transfer
its allocation. Unsupported ownership is rejected explicitly, never
repaired by a hidden payload copy. Arena storage must remain valid through
a retained allocation/owner and cannot be recycled while referenced.

Encoded inputs are read-only in the first implementation. A retained
state update becomes visible atomically only after execution and
validation succeed. Reset or close cannot race with a call. In-place KV
mutation and alias annotations remain later work.

Native bindings are keyed by ``type_id`` and verify the resolved
declaration before creating a typed view. Inline custom layouts require
generic field access or an explicit codec. Missing consumers fail
explicitly; loading a model never executes codec code. Format-specific
bindings and codecs stay outside ``lib_onnx_proto``.

``EncodedValueProto.struct_type`` selects an exact ``type_ref`` or a
concrete inline declaration eligible for byte encoding; a reference may
also appear inside ``TypeProto`` and nested fields. An unset-kind
``StructTypeProto`` is an unconstrained category, permitted only inside
``TypeProto`` for heterogeneous sequence/map elements, never as a payload
layout.

``Field.type`` and ``Array.element_type`` use ``TypeProto`` and its
existing validation rules for tensors, nested structs, sequences, maps and
optional values; tensor dimensions may be dynamic when permitted, and
array lengths and bit-packing counts stay explicit concrete integers. For
byte encoding, every non-constant field must resolve recursively to
fixed-size inline data; such a struct remains a valid runtime type but
cannot select the raw/external byte-payload layout. See
:ref:`l-next-steps-persistent-struct-state`.

``Field.constant`` is the actual ``TensorProto`` value, not a graph input,
and must have concrete dimensions and matching data. Only true shared
format constants belong here; mutable lengths, positions and per-block
quantization parameters remain instance data.

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
padding is explicit; bits run least-to-most-significant, multi-byte
values little-endian. Only fixed-width ONNX scalar leaves are physical
data, and every read is bounds-checked.

Type checking rejects malformed declarations (structural, naming, type
and count errors) plus, for byte encoding, non-fixed-size fields,
unsupported leaves and overflowing size arithmetic; the encoded root must
have strictly positive size divisible by eight. Constant-only structs are
valid types but cannot be encoding roots, though nested constant-only
groups may contribute zero bytes. Only the concrete root's decoder or
encoder is invoked; a nested ``type_ref`` contributes layout and
constants but not its own decoder/encoder.

One element type, many payload lengths
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In the byte-encodable subset, ``StructTypeProto`` defines one encoded
element and ``EncodedValueProto`` stores a flat sequence of them. Payload
byte length and resolved element size determine the record count; no
physical shape or redundant count is serialized, and a different count
does not create a new type.

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

This is descriptive syntax, not the final wire schema. Each physical
element holds 16 bytes of INT4 codes followed by one 4-byte FLOAT scale
with no implicit padding; the registered decoder defines signed-code
scaling and the block-to-logical-element mapping, with per-block scale
values in the payload rather than the declaration.

Three quantities are involved: the **element type** (the fixed physical
layout of one ``Int4Block``), the **payload byte length** (from which the
count is derived), and the **logical shape** (decoder- or kernel-exposed
dimensions).

For the structured branch, require:

.. code-block:: text

    element_bytes = checked_size(resolved_struct_type)
    payload_bytes = raw_data.size()        // inline payload
    // Or external_data.length for a validated external payload extent.
    require(element_bytes > 0)
    require(payload_bytes % element_bytes == 0)
    element_count = payload_bytes / element_bytes

The two buffers above thus contain ``2560 / 20 = 128`` and ``5120 / 20 =
256`` records of the same type; an empty payload means zero records, and
partial records are rejected. External data must supply an explicit
length and a valid backing-file extent.

Records are stored densely in buffer order; layouts needing internal
strides or padding express them in the fixed element structure. A byte
length never lets a tensor-only operator accept encoded bytes implicitly.

Quantization examples: constants and per-value parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The notation below abbreviates ordinary ONNX tensor types as
``tensor(FLOAT, [])`` and tensor constants as ``tensor(FLOAT, [], 0.25)``,
as in the block example above.

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
``ModelProto.struct_types[*].structure.field[*].constant``, and each
payload is exactly ``128 * 4 / 8 = 64`` bytes; decoding applies
``(code - (-2)) * 0.25``, and changing the constants requires a new type
ID. For type 1002, both payloads occupy ``64 + 4 + 8 = 76`` bytes; the
decoder reads each value's own parameters and applies
``(code - zero_point) * scale``, so changing those values leaves the
layout and type ID unchanged. The INT64 at byte offset 68 is not
necessarily aligned, so typed readers must handle it.

.. _l-next-steps-custom-types-codebook:

Codebook quantization through a shared subtype
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A subtype describes 32 two-bit indices and a constant four-entry codebook; a
parent embeds it by stable ID and adds a per-block FLOAT scale. This is
composition by reference, not inheritance: each parent payload still
contains its own code bytes.

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

``E4`` packs indices ``0, 1, 2, 3`` in least-significant-bit order and
``00 00 00 40`` encodes FLOAT 2.0. The codebook lives once in type 1101's
constant field and contributes no payload bytes. Resolution follows
``1102 -> quantized.type -> 1101``; the root decoder reads the subtype's
codebook and each record's scale and indices, without invoking a subtype
decoder:

.. code-block:: text

    table = resolved_type(1101).structure.field["codebook"].constant
    output[block * 32 + i] = record.scale * table[record.quantized.codes[i].index]

The first value decodes to ``[-2.0, -0.5, 0.5, 2.0]`` repeated eight
times; the second flattens 128 decoded blocks in storage order. Type 1101
contributes eight bytes and type 1102 contributes ``8 + 4 = 12`` bytes per
record, so payload lengths 12 and 1536 imply one and 128 records without
serialized counts. Other parent types may reuse subtype 1101; changing
its codebook requires a new subtype ID.

Proto-library size gate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The canonical PR01 baseline is the installed library from
`Linux core CI run 34234846380
<https://github.com/xadupre/onnx-light/actions/runs/34234846380>`_ at
commit ``f450bbbda903a94cce0bce1eff7fba94097190f5``. The job uses
Ubuntu 24.04 x86-64, GCC
13.3.0, CPython 3.13.15, the Python shared-library topology, Release mode,
CMake installation stripping, OpenSSL, and the existing CI command:

.. code-block:: bash

    pip install -C build-dir=build -C cmake.build-type=Release \
        -C cmake.define.ONNX_LIGHT_BUILD_TESTS=ON -e .[dev] -v
    proto_dir="$(python -c \
        'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec(
        "onnx_light.onnx_py._onnxpyprotoop").origin).parent)')"
    python .github/scripts/report_proto_binary_size.py \
        "${proto_dir}"

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Metric
     - PR01 baseline
     - PR02 maximum
   * - Stripped installed bytes
     - 1,062,872
     - 1,193,944
   * - Allocated section bytes
     - 1,054,507
     - Report only
   * - ``.text`` bytes
     - 724,330
     - 822,634
   * - Defined dynamic symbols
     - 696
     - 760
   * - ``DT_NEEDED``
     - ``libcrypto``, ``libstdc++``, ``libgcc_s``, ``libc``, ``ld-linux``
     - No additions

PR02 may add at most 128 KiB of stripped size, 96 KiB of ``.text`` and
64 dynamic symbols, while adding no shared-library dependency. Its
installed-size ceiling of 1,193,944 bytes is stricter than the existing
1.2 MiB project ceiling. The proto target contains only compact messages
and serialization machinery; format-specific validators, codecs,
catalogues and registration tables stay optional runtime dependencies.
The table's maxima were the absolute CI gates for PR02. The subsequent
native ORT reader/writer adds a separate, bounded allowance documented in
:ref:`l-design-ort-flatbuffer-format`; it does not expand the PR02
representation allowance. PR02 also reports a baseline and candidate built
side by side with the same workflow to verify the deltas. A runner
toolchain update refreshes the reference baseline in a separate PR, not as
part of a representation change.

.. _l-next-steps-custom-types-prepared-values:

Interoperability with prepared execution
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``EncodedValueProto`` may describe the bytes and logical view of an object
managed by :ref:`l-next-steps-prepared-execution`, but this plan does not
define prepack requests, prepared keys, prepared-object persistence,
compatibility checks, publication, eviction or scheduling; those stay entirely in
:ref:`l-next-steps-prepared-execution`, :ref:`l-next-steps-model-resolution`,
and :ref:`l-next-steps-native-fast-loading-completion`. The representation
layer exposes only enough validated type, layout and payload information
for a prepared consumer to bind a typed view, adding no preparation
provenance to ``EncodedValueProto`` and no ``prepared_values`` field to
``ModelProto``.

.. _l-next-steps-mutable-cache:

Persistent state from model inputs and outputs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

State is simply model outputs retained to supply some inputs of the next
call. The model describes both their types and their output-to-input
bindings. The caller supplies initial values and per-call feeds, not a
second independently configured mapping.

Graph-level persistence declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add ``persistent_bindings`` to ``GraphProto``, not a ``persistent`` flag
to ``StructTypeProto`` or ``EncodedValueProto``. A type can describe both
persistent and temporary values; an input can be an ordinary tensor or a
whole structure without an encoded payload. A boolean also cannot identify
the output that supplies the next value.

The proposed local wire extension is:

.. code-block:: text

    message PersistentBindingProto {
        string input_name = 1;
        string output_name = 2;
    }

    message GraphProto {
        // Existing fields, including encoded_initializer = 1000, remain.
        repeated PersistentBindingProto persistent_bindings = 1001;
    }

Names identify exact whole inputs and outputs of the declaring graph.
An input is entirely persistent or not persistent; partial field selection
is unsupported. Dots and backslashes in names are literal characters.
Field 1001 belongs to the reserved local
extension range; existing representation field numbers remain unchanged.

No bindings means ordinary stateless execution. The initial runtime scope
is the model's root graph; persistent bindings in control-flow subgraphs
are rejected explicitly until scoped state is supported. Passing state
values through stateless functions and subgraphs remains supported.
The declaration contains neither current cache contents nor native
ownership handles. It is serialized only during explicit model save/load,
not during state construction or execution.

.. code-block:: text

    model inputs:  tokens, past_key, past_value
    model outputs: logits, present_key, present_value
    model.graph.persistent_bindings:
        past_key <- present_key
        past_value <- present_value

    state = make_state(
        model,
        initial={"past_key": empty_key, "past_value": empty_value}
    )

    out = state.run({"tokens": first_tokens})
    out = state.run({"tokens": next_tokens})

This helper is equivalent to the ordinary stateless loop:

.. code-block:: text

    feeds = {"tokens": tokens, **state.values}
    outputs = run(model, feeds)
    state.values = {
        input_name: outputs[output_name]
        for input_name, output_name in model.graph.persistent_bindings
    }

These examples are conceptual API sketches. Assignment retains or moves
buffer owners rather than copying payloads. The retained values
constitute the request-local state; ``PersistentBindingProto`` declares
only the graph relationship, not a second value/type system or hidden
kernel state. ``make_state`` resolves the model's bindings and types once,
after graph rewrites. Initial contents must still be supplied.

.. important::

   The model must remain immutable for the lifetime of the bound session.
   Native callers keep it alive; Python bindings retain its owner.
   State construction and subsequent calls must not serialize, clone,
   hash or repeatedly compare the model to detect mutation. To change
   the graph, close the state and create a new session after rewriting.

.. _l-next-steps-persistent-composite-state:
.. _l-next-steps-persistent-struct-state:

Whole structured inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same mapping can retain an entire structured input. Fresh values belong
to separate graph inputs. These are ordinary structs described by ``StructTypeProto``:

.. code-block:: text

    Cache = struct {
        keys: Tensor
        values: Tensor
        length: INT64
    }

    model input tokens: INT64[batch, sequence]
    model input cache: Cache
    model output logits: FLOAT[batch, sequence, vocabulary]
    model output next_cache: Cache
    model.graph.persistent_bindings:
        cache <- next_cache

    state = make_state(
        model,
        initial={"cache": initial_cache}
    )
    out = state.run({"tokens": first_tokens})
    out = state.run({"tokens": next_tokens})

The whole ``cache`` input persists: tokens are supplied anew and logits are
separate outputs, not retained. The graph binding selects a whole value,
with no ``persistent`` flag on the shared struct type or encoded value.
``Tensor`` above only abbreviates the model's actual tensor types and
shape constraints.

Minimal rules
~~~~~~~~~~~~~

* Every selected graph input/output must exist with compatible types, representation
  contracts and shape constraints. Resolve the declarations once;
  validate actual value metadata before publication without copying or
  serializing payloads.
* Every required whole input comes from current feeds or retained state;
  missing initial values, duplicate input/output selections, and current
  feeds overriding retained inputs are errors.
* Retained values update only after successful execution and validation;
  each request owns its state bindings, and simultaneous calls on the
  same state are rejected. ``reset(initial)`` restores caller-supplied initial values;
  ``close`` releases retained values, and neither may race with an
  active call.

Zero-copy ownership from the first state implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PR04 must retain/share/move existing backing storage for initialization,
reset, per-call feeds, selected outputs and next-call inputs. State-value
access returns lifetime-safe shared views, not implicit deep-copy
snapshots. Copying small metadata, shape descriptors or owner handles is
allowed; copying tensor or encoded payload bytes for state management is
not. NumPy and PyTorch entry points must likewise retain supported source
buffers without materializing payload copies, or report unsupported
layouts/ownership explicitly.

Returned outputs and retained state may share storage. Releasing caller
references, resetting/closing a state or advancing another invocation must
not invalidate an outstanding view. Functions, ``If`` and structured
field transport must preserve these owners rather than call deep-copy
or serialization-based detachment helpers. Per-request state containers
are separate; shared buffers do not imply permission to mutate another
request's data.

.. warning::

   Inputs, retained state and returned views can alias the same payload.
   Callers and kernels must treat these buffers as read-only while shared
   or retained. PR04 does not promise independent mutable snapshots or
   rollback of external writes. It must not silently enable in-place
   reuse of an allocation still referenced by the previous state or a
   returned output.

Build and validate the next state's owner handles before publishing them
atomically. Failure or cancellation leaves the previous state and its
buffers valid; it must not require a backup copy. Persistence implies
neither mutation nor disk storage.

This ownership requirement is distinct from PR05's in-place KV append
and capacity management. Kernels may compute new outputs normally; PR04
must not duplicate those outputs merely to retain or return them.
Avoiding algorithmic full-cache reconstruction, paging, explicit
snapshots and region-level mutation scheduling remain separate work.

Quantized and paged caches use the same feedback
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A cache can contain a sequence of blocks instead of contiguous K/V tensors.
This changes its fields and representations, not its persistence mechanism:

.. code-block:: text

    Cache = struct {
        blocks: sequence<KVBlock>
        length: INT64
    }
    model.graph.persistent_bindings:
        cache <- next_cache

Each K/V block may use a different ``EncodedValueProto`` layout, such as
INT4, INT8 or a codebook struct, provided its decoded type and geometry
satisfy the consumer contract. The block's logical token range and valid
length are value data; payload byte length measures physical records,
not valid tokens.

Paging, block conversion and zero-copy Attention are optional consumer
optimizations built on already zero-copy state forwarding; acceptance
measures bounded workspace and no full-cache copy or dequantization.

GraphBuilder, shape inference and serialization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ShapesContext`` stays the single source of truth for symbolic value
types, extended with ``StructTypeProto`` field types and the byte-encoded
layout.

``GraphBuilder`` preserves structured source initializers, type
references, byte extents, external payload ownership and quantization
metadata through import/export. Deduplication considers semantic
profiles, not just payload bytes. It must also author and preserve
``GraphProto.persistent_bindings`` through import/export, graph copying,
renaming and optimization. Renaming an input/output updates its binding;
a rewrite removing or changing a selected input/output must update the declaration
consistently or fail, never silently drop it. Validate bindings against the
final graph before constructing a session. General struct
declarations round-trip through the type catalogue too; unsupported
field-value export fails explicitly rather than dropping dynamic fields.
The new binding field is a local ONNX extension, not a change required of
upstream ONNX or a dependency on proto inheritance. Standard ONNX export
must reject persistence declarations unless an explicit lowering to a
stateless graph and caller-managed feedback preserves the contract.
Silently stripping the attribute is not a valid stateful export.

Implementation sequence
+++++++++++++++++++++++

PR01 and PR02 are complete: the representation, identity and size contracts
are implemented, and ``StructTypeProto``,
``EncodedValueProto``, typed/constant fields, arrays, bit packing, type
references, payload ownership, value serialization and the built-in affine
subset are implemented in ``lib_onnx_proto``
(``onnx_light/onnx_proto/onnx.h`` and ``onnx_verify.h``/``.cc`` plus the
``TypeProto``/``GraphProto``/``ModelProto`` field-1000 branches), with
structural validation wired into ``VerifyModel``/``VerifyGraph``. PR03
integrates the representation with ``GraphBuilder`` authoring,
deduplication and inference; see
:ref:`l-howto-graph-builder-basics` for the supported native workflow and
export boundaries. The 2026-09-20 revision adds a graph-level persistence
declaration and makes zero-copy state forwarding mandatory. These are new
requirements, not claims that the completed PRs already implement them.
`PR #5016 <https://github.com/xadupre/onnx-light/pull/5016>`_
implements the PR04a declaration and PR04b runtime together: the
caller-supplied mapping, model-serialization guard and defensive
state-payload copies are replaced by graph bindings and retained owners.
The Linux Release measurement for this implementation is 1,388,072 installed
bytes, 959,130 ``.text`` bytes and 795 defined dynamic symbols. Relative to
the post-ORT limits, PR04 receives a bounded allowance of 32 KiB installed,
16 KiB of ``.text`` and eight symbols, setting the corresponding CI limits
to 1,423,320 bytes, 970,090 bytes and 800 symbols. The shared-library
dependency allowlist remains unchanged.

.. list-table::
   :header-rows: 1
   :widths: 8 27 48 17

   * - PR
     - Scope
     - Acceptance
     - Depends on
   * - PR01
     - Representation and lifetime contracts (**done**)
     - Freeze the built-in affine subset, struct extension path, fixed
       element types, payload-derived counts, catalogue identities,
       native bindings and feedback matching; record the proto-size
       baseline and budget before PR02.
     - Existing runtime APIs
   * - PR02
     - Structs first, then minimal built-in layouts (**done**)
     - Implement ``StructTypeProto`` and the structured branch of
       ``EncodedValueProto`` first, then common INT8/INT4 layouts.
       Round-trip tensor/sequence fields, byte-encoded custom records,
       codebook/mixed-bit formats and a heterogeneous KV-block fixture,
       proving one type shared across payload lengths within the
       binary-size budget.
     - PR01
   * - PR03
     - GraphBuilder and serialization integration (**done**)
     - Structured initializers, logical/physical inference, scope-aware
       references and deduplication agree. Standard export never loses
       data: unsupported structured constructs are rejected explicitly.
     - PR02
   * - PR04a
     - Graph-declared persistence
     - Add ``PersistentBindingProto`` and
       ``GraphProto.persistent_bindings``; native/Python bindings,
       parsing/serialization, validation and GraphBuilder preservation
       agree. Reject unsupported nested-graph declarations and standard
       export that would drop persistence semantics.
     - PR02, PR03
   * - PR04b
     - Zero-copy request-local feedback execution
     - Resolve graph bindings once against an immutable model. Retain
       buffers across initialization, reset, calls and state views without
       payload copies or model serialization. Verify aliases, lifetimes,
       atomic failure/cancellation and pointer identity, including
       structured/function/If paths.
     - PR04a; existing allocation/task infrastructure
   * - PR05
     - Contiguous KV and CPU consumer integration
     - Optimize past/present inputs when ownership permits buffer reuse:
       append touches only new tokens and matches functional execution.
       Verify capacity/cancellation and allocation/copy costs without
       invalidating prior returned views. Zero-copy state forwarding
       is already required by PR04b; this step optimizes kernel writes.
     - PR04b; CPU backend integration
   * - PR06
     - Optional paged KV with heterogeneous quantization
     - The shared ``EncodedValueProto`` representation supports
       different K/V and per-block formats. Blockwise append/conversion
       and Attention preserve validity and bounded workspace without
       copying or dequantizing the entire cache.
     - PR02, PR05; CPU backend integration
   * - PR07
     - End-to-end structured/stateful acceptance
     - Measure repeated decode and simultaneous independent feedback
       states; report state/scratch bytes and per-token copies, and
       verify request reset/isolation and the final proto-size budget.
     - PR03, PR04a, PR04b, PR05

PR04 is now split into the wire/GraphBuilder declaration (PR04a) and its
zero-copy runtime consumer (PR04b), in that order. Basic feedback does not
depend on quantization format or paging. PR06 is optional and does not
block PR07. Later work extends the same graph-declared feedback contract
without a second state system; explicit snapshots, alias annotations and
mutation scheduling stay outside these initial steps.

Ownership and acceptance
++++++++++++++++++++++++

``onnx-light`` owns type/serialization contracts, allocation and
lifecycle, graph/schema integration, effect scheduling and input/output
feedback state. ``onnx-light-cpu`` supplies format validators, typed
consumers, KV append and Attention, without creating another
persistent-state manager or private executor.

Acceptance uses C++ fixtures and existing runtime/backend test
infrastructure: compare encoded versus decoded computation and the state
helper versus a manual output-to-input loop, covering changed scales with
unchanged code bytes, missing consumers, reset, invalid capacities and
failed mutations.

State fixtures retain whole tensors and whole structured cache inputs,
verifying that separate unselected outputs are not retained and the next
call receives the same backing buffers as the previous selected outputs.
Measure pointer identity and payload-copy/allocation counters across
initialization, repeated calls, reset and state-view access, including
NumPy/PyTorch entry points, encoded values and function/If transport.
Exercise caller destruction, outstanding outputs after close, arena
reuse, invalid ownerless borrows and failed/cancelled runs.

Wire/GraphBuilder fixtures round-trip bindings, preserve them through
renames and rewrites, distinguish dotted names from nested fields, and
reject duplicate input/output selections, unknown exact names, incompatible
types/shapes and unsupported subgraph declarations. Two values sharing
one struct type can have different persistence bindings. Large-model
fixtures verify that constructing and running state performs no model
serialization, cloning or mutation-detection scan.

Type/value tests round-trip two encoded values sharing a type ID but
different payload lengths and record counts, covering catalogue
reordering and per-value scale/zero-point parameters, with
malformed-declaration and byte-encoding validation exercised through
representative cases.

PR04b acceptance requires zero payload-copy bytes attributable to state
management and no model serialization, both during setup and repeated
calls. Kernel computation and explicit model save/load are measured
separately, not hidden inside state-management overhead. PR05 must further
demonstrate fixed-capacity reuse before claiming no algorithmic full-cache
allocation or copy. Performance reports include latency, peak/resident
bytes and copy/read counters, distinguishing decoding, inference and
state-management cost.
