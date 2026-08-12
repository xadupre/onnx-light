.. _l-next-steps-onnx-proto:

Building ``onnx_proto``, the protobuf-free ONNX schema
======================================================

:Date: 2025-07

**complete**

Objective
+++++++++

``onnx-light`` started from the upstream ONNX pull request
`onnx/onnx#7208 <https://github.com/onnx/onnx/pull/7208>`_.  Its first goal was
to read and write ``.onnx`` files without linking ``libprotobuf``.  Every ONNX
model is the binary serialization of a
:class:`~onnx_light.onnx_lib.ModelProto` message described in `onnx.proto
<https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>`_, so the ``onnx_proto``
target had to reproduce that message hierarchy, its wire format, and enough of
the protobuf message API for existing consumers, while remaining a small,
self-contained C++ library.

The constraints were:

* preserve the ONNX wire format so that files produced by ``onnx`` and
  ``onnx-light`` are byte-compatible;
* expose message classes with typed field accessors that feel like the
  protobuf-generated ones;
* avoid the 2 GB message-size limit imposed by protobuf and support
  external-data / multi-file models natively;
* keep the code generator-free — the messages are hand-written but macro-driven
  rather than emitted by ``protoc``.

Post-mortem
+++++++++++

The library was built bottom-up, from the wire primitives to the generated
message classes and finally the compatibility surface consumers rely on.

Binary stream primitives
^^^^^^^^^^^^^^^^^^^^^^^^^

The protobuf wire format is a flat concatenation of *(tag, value)* pairs where
the tag packs a field number and a 3-bit wire type
(:ref:`see the format notes <l-design-protobuf-format>`).  ``stream.h`` /
``stream.cc`` implement the reading and writing of that encoding:

* varint, fixed32, fixed64, and length-delimited primitives;
* a ``BinaryStream`` reader and a ``BinaryWriteStream`` writer that the rest of
  the library targets;
* zero-copy spans (``simple_span.h``) so a tensor's ``raw_data`` can point
  directly into the source buffer instead of allocating a copy.

Field storage
^^^^^^^^^^^^^

``fields.h`` / ``fields.hpp`` provide the storage building blocks that back
every message field: optional scalars, optional strings
(``simple_string.h``), optional embedded messages, and repeated variants
(``RepeatedField`` for scalars, ``RepeatedProtoField`` for messages).  These
own their memory and offer move-friendly setters
(``set_<name>(T&&)``, ``add_<name>(T&&)``) so assembling a graph does not deep
copy every ``TensorProto``.

Macro-generated messages
^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of running ``protoc``, message classes are declared with the macros in
``stream_class.h``.  Each ``.proto`` message maps to a
``BEGIN_PROTO`` / ``FIELD_*`` / ``END_PROTO`` block in ``onnx.h``, and the
macros expand into the storage, accessors, and the ``SERIALIZATION_METHOD``
core (``ParseFromStream``, ``SerializeToStream``, ``SerializeSize``, and
``PrintToStringStream``).  The per-message parse and serialize bodies live in
``onnx.cc``.  This keeps the ``.proto`` field numbers next to the C++ code and
removes the build-time dependency on the protobuf compiler.

Protobuf-compatible API
^^^^^^^^^^^^^^^^^^^^^^^^

Consumers expect the protobuf message API (``ParseFromString``,
``SerializeToString``, ``ByteSizeLong``, ...).  These convenience entry points
are provided as thin wrappers around the type-specific serialization core, so
a message parses and serializes as if it were protobuf-generated while the wire
work stays in one place.  ``google_protobuf_compat.h`` supplies the handful of
``google::protobuf`` symbols (repeated fields, stream types) that callers still
name directly.

Helpers and tools
^^^^^^^^^^^^^^^^^

Around the messages, ``onnx_helper.h`` / ``onnx_helper.cc`` add the utilities
model tools depend on (attribute lookup, scalar reading such as
``ReadScalarAsDouble``, external-data handling), and ``onnx_verify.cc`` performs
model and tensor validation.

What worked
^^^^^^^^^^^

* Building on the wire primitives first meant the message layer only had to
  describe fields, not encoding rules.  Byte-compatibility with ``onnx`` was
  verified by round-tripping real models.
* The macro approach kept the schema declaration compact and reviewable, and
  made it cheap to keep the ``.proto`` comments and field numbers next to the
  generated C++.
* Owning field storage with move-aware setters avoided a class of accidental
  deep copies while still exposing a protobuf-like API.
* Removing the ``libprotobuf`` dependency dropped the 2 GB limit and enabled
  native external-data, zero-copy parsing, and parallel load / save.

What remains
^^^^^^^^^^^^

The library initially bundled verification, hashing, encryption, and text
printing next to the parser.  Trimming that surface, hiding internal symbols,
and reducing the generated per-message wrappers were later measured and
delivered as a separate workstream
(:ref:`l-next-steps-proto-binary-size`).
