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
:cpp:class:`onnx::utils::RepeatedField` or
:cpp:class:`onnx::utils::RepeatedProtoField` container.

Optional fields (``FIELD_OPTIONAL``, ``FIELD_OPTIONAL_ENUM``) wrap their value
in :cpp:class:`onnx::utils::OptionalField` or
:cpp:class:`onnx::utils::OptionalEnumField` and add ``reset_foo()`` and
``add_foo()`` members.

Every proto class inherits from :cpp:class:`onnx::Message` and includes the
following serialization / deserialization methods (added by
``SERIALIZATION_METHOD()``):

.. code-block:: cpp

    uint64_t SerializeSize() const;
    void ParseFromString(const std::string &raw);
    void ParseFromString(const std::string &raw, onnx::ParseOptions &options);
    void SerializeToString(std::string &out) const;
    void SerializeToString(std::string &out, onnx::SerializeOptions &options) const;
    uint64_t SerializeSize(onnx::utils::BinaryWriteStream &stream,
                           onnx::SerializeOptions &options) const;
    void ParseFromStream(onnx::utils::BinaryStream &stream, onnx::ParseOptions &options);
    void SerializeToStream(onnx::utils::BinaryWriteStream &stream,
                           onnx::SerializeOptions &options) const;
    std::vector<std::string> PrintToVectorString(onnx::utils::PrintOptions &options) const;

See :doc:`stream_class` for :cpp:class:`onnx::ParseOptions`,
:cpp:class:`onnx::SerializeOptions`, and :cpp:class:`onnx::Message`.

API reference
-------------

.. doxygenfile:: onnx.h
   :project: onnx-light
