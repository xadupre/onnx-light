.. _l-next-steps-proto-inheritance:

Proto schema inheritance
========================

:Date: 2026-08

**in progress**

Motivation
++++++++++

Protocol Buffers does not support message inheritance. Related messages
therefore repeat common fields or introduce a nested base message:

.. code-block:: text

    message Derived {
        Base base = 1;
        ...
    }

Composition is portable but changes the wire path and the API:
``value.name()`` becomes ``value.base().name()``. It is inconvenient when one
message is a strict specialization of another, for example when
``QuantizedTensorProto`` adds quantization semantics to the bytes and metadata
already carried by ``StructProto``.

onnx-light owns its message classes and parser rather than relying on generated
Protobuf C++ classes. It can therefore support inheritance in its schema and
C++ API while keeping a conventional, flattened Protobuf wire representation.

Scope
+++++

The proposal defines **single inheritance for message fields**. It provides:

* inherited fields and accessors in the C++ API;
* one flattened wire message containing base and derived fields;
* source-level reuse in onnx-light declarations;
* an equivalent ordinary ``.proto`` schema for other implementations.

It does not define runtime-polymorphic protobuf fields. A field declared as
``StructProto`` still contains that wire message; it does not
automatically accept every derived message. Explicit ``oneof`` branches or a
common value category remain necessary for polymorphic containers.

Schema syntax
+++++++++++++

The conceptual syntax is:

.. code-block:: text

    message StructProto {
        int32 type = 1;
        optional StructTypeProto struct_type = 2;
        bytes raw_data = 3;
        repeated StringStringEntryProto external_data = 4;
        string name = 5;
        string doc_string = 6;
    }

    message QuantizedTensorProto extends StructProto {
        repeated int64 dims = 7;
        int32 quantized_type = 8;
        optional QuantizationProto quantization = 9;
    }

The equivalent standard Protobuf declaration is flattened:

.. code-block:: text

    message QuantizedTensorProto {
        int32 type = 1;
        optional StructTypeProto struct_type = 2;
        bytes raw_data = 3;
        repeated StringStringEntryProto external_data = 4;
        string name = 5;
        string doc_string = 6;
        repeated int64 dims = 7;
        int32 quantized_type = 8;
        optional QuantizationProto quantization = 9;
    }

No inheritance marker appears on the wire. Existing Protobuf tooling can use
the flattened declaration without understanding the onnx-light extension.

Rules
+++++

Single inheritance
^^^^^^^^^^^^^^^^^^

A message has at most one direct base. Chains are permitted, but the schema
compiler must reject cycles. Multiple inheritance is excluded because field
and oneof resolution would become ambiguous.

Field numbers
^^^^^^^^^^^^^

Base and derived fields share one field-number namespace:

* a derived field number may not collide with any ancestor field;
* an inherited reserved number or name remains reserved;
* changing the base after publishing a derived message may not claim a number
  already used by that derived message;
* removing an inherited field reserves its number and name in every
  descendant.

A practical convention is to allocate a field-number range to the base before
publishing descendants. The checker must nevertheless validate the complete
flattened hierarchy rather than rely on ranges.

Fields and oneofs
^^^^^^^^^^^^^^^^^

Inherited fields retain their original presence, default, packed encoding,
documentation, and oneof membership. A derived message:

* may add new fields and new oneofs;
* may not change an inherited field type, number, default, or cardinality;
* may not add alternatives to an inherited oneof;
* may not hide or redeclare an inherited field name.

These restrictions make flattening deterministic.

Wire compatibility
^^^^^^^^^^^^^^^^^^

Serialization emits inherited and local fields at their original field
numbers. Field order remains irrelevant, as in Protobuf.

The bytes of a derived payload can be parsed using the base schema: derived
fields are unknown fields. This does **not** create substitution between
message-typed fields because the enclosing field still declares one concrete
message type. Furthermore, round-tripping through a reader that discards
unknown fields loses the derived portion.

Inheritance must therefore not be used as a substitute for an explicit
``oneof`` or ``TypeProto`` category.

C++ API
+++++++

The desired API uses normal public inheritance:

.. code-block:: cpp

    class QuantizedTensorProto : public StructProto {
    public:
      // Inherited:
      // name(), doc_string(), raw_data(), external_data(), ...

      // Local:
      const std::vector<int64_t>& dims() const;
      int32_t quantized_type() const;
    };

The primary purpose is field and accessor reuse. It does not require virtual
serialization or ownership through base pointers. The current
``onnx_light::Message`` API is non-polymorphic, and this proposal should not
add a vtable to every message merely to support schema inheritance.

Copying, clearing, equality, hashing, printing, and Python bindings operate on
the complete flattened message. Passing a derived object by value as its base
still causes normal C++ slicing and must be avoided.

Implementation in onnx-light
++++++++++++++++++++++++++++

Current declarations use ``BEGIN_PROTO`` or ``BEGIN_PROTO_NOINIT``; both make
every message inherit directly from ``Message``. Serialization and parsing in
``onnx.cc`` explicitly enumerate each message's fields.

The declaration layer can add:

.. code-block:: cpp

    BEGIN_PROTO_DERIVED(QuantizedTensorProto, StructProto, "...")
    FIELD_REPEATED(int64_t, dims, 7, "Logical tensor shape.")
    FIELD_DEFAULT(int32_t, quantized_type, 8, -1, "Quantization type index.")
    FIELD_OPTIONAL(QuantizationProto, quantization, 9, "Inline quantization type.")
    END_PROTO()

``BEGIN_PROTO_DERIVED`` expands to a class inheriting from the supplied base
and declares the same serialization API as an ordinary message.

Serialization helpers
^^^^^^^^^^^^^^^^^^^^^

Size computation and writing can process base fields before local fields:

.. code-block:: cpp

    SerializeBaseFields(size, stream, options);
    SIZE_REPEATED_FIELD(size, options, stream, dims)
    ...

    WriteBaseFields(stream, options);
    WRITE_REPEATED_FIELD(options, stream, dims)
    ...

Parsing cannot call ``Base::ParseFromStream`` and then parse local fields:
the base parser consumes the complete message and skips fields it does not
know. Instead, parsing uses one loop and delegates individual fields:

.. code-block:: cpp

    while (stream.NotEnd()) {
      const auto field = stream.next_field();
      if (ParseLocalField(field, stream, options))
        continue;
      if (ParseBaseField(field, stream, options))
        continue;
      SkipUnknownField(field, stream, options);
    }

Each generated message therefore needs field-level helpers for:

* serialized-size accumulation;
* writing;
* parsing one field;
* printing;
* copying and clearing;
* equality or hashing when supported.

The same helpers can later reduce duplication in ordinary non-derived
messages.

Unknown fields
^^^^^^^^^^^^^^

onnx-light currently skips fields it does not model. Consequently, parsing a
derived payload as its base and serializing it again cannot be expected to
preserve derived fields. Schema inheritance does not change this behavior.
Unknown-field preservation would be an independent feature.

Bindings
^^^^^^^^

nanobind registrations for a derived message should declare its C++ base so
Python inherits the same accessors:

.. code-block:: cpp

    nb::class_<QuantizedTensorProto, StructProto>(m, "QuantizedTensorProto");

The flattened standard-Protobuf Python class will not expose Python
inheritance, but it will expose the same fields. Behavioral compatibility is
defined by fields and wire encoding, not by the host-language class hierarchy.

Schema generation and validation
++++++++++++++++++++++++++++++++

onnx-light should maintain one inheritance-aware schema description and
derive two outputs:

* onnx-light C++ declarations with real inheritance;
* ordinary flattened ``.proto`` declarations for interoperability.

Generation must fail when:

* the base is missing, incomplete, or creates a cycle;
* inherited and local numbers or names collide;
* a descendant redefines an inherited field;
* an inherited oneof is extended;
* a reserved name or number is reused;
* the flattened declaration differs from the checked wire schema.

Compatibility policy
++++++++++++++++++++

Adding inheritance to existing published messages is safe only if the
flattened fields and numbers remain unchanged. Moving an existing nested base
message to inherited fields changes wire numbers and is a migration, not a
refactor.

For new messages, inheritance is wire-compatible from the beginning when the
flattened schema is treated as normative. For existing
``QuantizedTensorProto`` and ``StructProto`` proposals, field numbers
should be finalized only after deciding whether inheritance is used.

Recommendation
++++++++++++++

onnx-light can support schema inheritance, but it should remain a declaration
and API convenience with explicit flattening. Recommended constraints are:

* single inheritance only;
* no field overriding;
* no extension of inherited oneofs;
* no implicit polymorphic message fields;
* flattened Protobuf schema as the normative wire contract;
* non-virtual C++ messages unless runtime polymorphism is independently
  justified.

Under these rules, ``QuantizedTensorProto extends StructProto`` removes
duplicated storage metadata without introducing a new or incompatible wire
format.
