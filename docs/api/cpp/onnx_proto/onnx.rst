.. _l-cpp-onnx-h:

onnx.h
======

This page documents all ONNX proto message classes defined in ``onnx.h``.
These classes mirror the Google Protocol Buffers schema from the
`ONNX specification <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>`_
but are generated entirely from lightweight C++ macros — no protobuf runtime is
required.

Field accessor pattern
----------------------

Every proto field named ``foo`` of type ``T`` exposes the following members:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Description
   * - ``T foo_``
     - The stored value (public data member).
   * - ``T & ref_foo()``
     - Returns a mutable reference to the field.
   * - ``const T & ref_foo() const``
     - Returns a const reference to the field.
   * - ``const T * ptr_foo() const``
     - Returns a const pointer to the field, or ``nullptr`` for absent optional
       fields.
   * - ``bool has_foo() const``
     - Returns ``true`` when the field holds a non-default value.
   * - ``void set_foo(const T &v)``
     - Assigns a new value to the field.
   * - ``int order_foo() const``
     - Returns the protobuf field number.
   * - ``static constexpr const char* DOC_foo``
     - The documentation string for this field, available at compile time.

Repeated fields (``FIELD_REPEATED``, ``FIELD_REPEATED_PACKED``) additionally
provide ``add_foo()`` and ``clr_foo()``, and store their values in a
:cpp:class:`onnx_light::utils::RepeatedField` or
:cpp:class:`onnx_light::utils::RepeatedProtoField` container.

Optional fields (``FIELD_OPTIONAL``, ``FIELD_OPTIONAL_ENUM``) wrap their value
in :cpp:class:`onnx_light::utils::OptionalField` or
:cpp:class:`onnx_light::utils::OptionalEnumField` and add ``reset_foo()`` and
``add_foo()`` members.

Every proto class inherits from :cpp:class:`onnx_light::Message` and includes the
following serialization / deserialization methods (added by
``SERIALIZATION_METHOD()``):

.. code-block:: cpp

    uint64_t SerializeSize() const;
    void ParseFromString(const std::string &raw);
    void ParseFromString(const std::string &raw, onnx_light::ParseOptions &options);
    void SerializeToString(std::string &out) const;
    void SerializeToString(std::string &out, onnx_light::SerializeOptions &options) const;
    uint64_t SerializeSize(onnx_light::utils::BinaryWriteStream &stream,
                           onnx_light::SerializeOptions &options) const;
    void ParseFromStream(onnx_light::utils::BinaryStream &stream, onnx_light::ParseOptions &options);
    void SerializeToStream(onnx_light::utils::BinaryWriteStream &stream,
                           onnx_light::SerializeOptions &options) const;
    void PrintToStringStream(std::stringstream &ss, onnx_light::utils::PrintOptions &options) const;

See :doc:`stream_class` for :cpp:class:`onnx_light::ParseOptions`,
:cpp:class:`onnx_light::SerializeOptions`, and :cpp:class:`onnx_light::Message`.

Structural equality
-------------------

Every proto class provides ``Equals`` for comparing two messages of the same
type without serialization:

.. code-block:: cpp

    std::string difference;
    bool equal = actual.Equals(expected, &difference);

The optional diagnostic receives the first differing field path and reason,
for example ``structure.field[1].name: values differ``. It is cleared when the
messages are equal. Omitting it performs the same comparison without collecting
the diagnostic.

Comparison includes field presence, nested messages, repeated-field order,
metadata and payload bytes. Floating-point fields are compared bit for bit:
identical NaN representations compare equal, while positive and negative zero
compare unequal. External data references are compared as stored, without loading
their files. This is structural equality, not model validation or numerical
tensor equality. Domain-specific comparisons such as ``SameDeclaredType`` remain
separate and explicitly exclude the fields they ignore.

The public ``Equals`` methods and their diagnostic implementation extend the
Linux proto-library size budget to 1,552,328 installed bytes, 1,085,786 ``.text``
bytes and 852 defined dynamic symbols, as measured in CI run ``36120766116``.
Shared quantization adds serialization, comparison and validation of
``EncodedValueProto.parameter_ref``. CI run ``36132778661`` measures the updated
budget at 1,556,424 installed bytes and 1,087,098 ``.text`` bytes, an increase of
4,096 and 1,312 bytes respectively. The limit of 852 defined dynamic symbols
and the shared-library dependency allowlist are unchanged.

The paged-cache messages add copy, serialization, comparison and validation code.
A Linux Release build measures 1,597,480 installed bytes, 1,121,802 ``.text``
bytes and 874 defined dynamic symbols. The CI limits match these measurements;
the shared-library dependency allowlist is unchanged.

API reference
-------------

.. doxygenfile:: onnx.h
   :project: onnx-light
