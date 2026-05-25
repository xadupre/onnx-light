.. _l-design-protobuf-format:

Protobuf format applied to ONNX
================================

ONNX models are serialized using `Protocol Buffers
<https://protobuf.dev/>`_ (protobuf), Google's compact binary encoding
format.  An ``.onnx`` file is just the binary serialization of a
``ModelProto`` message defined in `onnx.proto
<https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>`_.  This page
explains the relevant subset of the protobuf wire format and how it maps
to ONNX message types.  It is meant to help readers understand the
low-level layout that :epkg:`onnx` produces and that ``onnx_light``
parses and writes without depending on ``libprotobuf``.

----

The wire format in a nutshell
-----------------------------

A protobuf-encoded message is a flat concatenation of *(tag, value)*
pairs.  There is no framing, no length prefix at the top level, and no
field ordering requirement.  A consumer reads bytes from start to end,
decodes each tag, and dispatches the following bytes according to the
*wire type* embedded in the tag.

Tag encoding
~~~~~~~~~~~~

Each field starts with a tag encoded as a *varint* (see below).  The
tag combines the field number declared in the ``.proto`` file with a
3-bit wire type:

.. code-block:: text

    tag = (field_number << 3) | wire_type

The wire types relevant to ONNX are:

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Value
     - Name
     - Meaning
   * - 0
     - ``VARINT``
     - Variable-length integer (``int32``, ``int64``, ``uint32``,
       ``uint64``, ``bool``, ``enum``, ``sint32``, ``sint64``).
   * - 1
     - ``I64``
     - Fixed 8 bytes, little-endian (``fixed64``, ``sfixed64``,
       ``double``).
   * - 2
     - ``LEN``
     - Length-prefixed payload: a varint *length* followed by *length*
       bytes (``string``, ``bytes``, embedded messages, packed
       repeated fields).
   * - 5
     - ``I32``
     - Fixed 4 bytes, little-endian (``fixed32``, ``sfixed32``,
       ``float``).

Wire types 3 and 4 (start-group / end-group) are deprecated and not
used by ONNX.

The decoding side appears in ``onnx_light/onnx_proto/stream.cc``: the
function ``BinaryStream::next_field()`` reads the tag varint and splits
it into ``field_number = tag >> 3`` and ``wire_type = tag & 0x07``.

Varints
~~~~~~~

A *varint* (variable-length integer) encodes an unsigned integer using
1 to 10 bytes.  Each byte stores 7 payload bits in its low bits and
uses the most significant bit (``0x80``) as a *continuation* flag:

* ``0x80`` set: more bytes follow.
* ``0x80`` clear: this is the last byte.

The payload bits are stored *little-endian* (least-significant 7 bits
first).  For example the integer ``300`` encodes as ``0xAC 0x02``:

.. code-block:: text

    byte 0: 1010 1100   -> continuation=1, payload=0101100  (low 7 bits)
    byte 1: 0000 0010   -> continuation=0, payload=0000010  (next 7 bits)
    value = (0000010 << 7) | 0101100 = 300

Because a 64-bit value contains at most ``ceil(64 / 7) = 10`` payload
groups, a varint never exceeds 10 bytes.  Field numbers from 1 to 15
fit in a single tag byte, which is why frequently used fields in
ONNX are assigned small numbers.

ZigZag encoding
~~~~~~~~~~~~~~~

The protobuf types ``sint32`` and ``sint64`` apply *ZigZag* mapping
before varint encoding so that small negative numbers do not require
the full 10 bytes.  The mapping interleaves positive and negative
values:

.. code-block:: text

      0 -> 0
     -1 -> 1
      1 -> 2
     -2 -> 3
      2 -> 4
      ...

It is implemented in ``onnx_light/onnx_proto/stream.h`` by
``encodeZigZag64`` / ``decodeZigZag64``.  Note that the plain
``int32`` and ``int64`` ONNX fields do **not** use ZigZag; they use the
two's-complement representation directly, which is why a negative
``int64`` always takes 10 bytes.

Length-prefixed values
~~~~~~~~~~~~~~~~~~~~~~

For wire type ``LEN`` the encoder writes a varint *length* followed by
*length* raw bytes.  This single mechanism is reused for:

* UTF-8 strings (``string``);
* arbitrary byte blobs (``bytes``), including tensor ``raw_data``;
* nested messages (such as ``GraphProto`` inside ``ModelProto``);
* packed repeated fields of scalar types.

Embedded messages are simply written out as their own bytestream and
prefixed by their total size, so the parser can either descend into
the substream or skip the whole region.  ``onnx_light`` exposes this
pattern through ``BinaryStream::LimitToNext()`` and ``Restore()``,
which push and pop a temporary read limit corresponding to the
length-prefixed substream.

Packed repeated fields
~~~~~~~~~~~~~~~~~~~~~~

Repeated scalar fields can be encoded in two ways:

* **Unpacked** – each element is written with its own tag, repeating
  the field number once per value.  This is the only legal encoding
  for repeated message fields and the legacy encoding for proto2
  scalar fields.
* **Packed** – all elements are concatenated into a single
  length-prefixed block (wire type ``LEN``) with a single tag.

ONNX uses packed encoding for scalar arrays such as
``TensorProto.float_data``, ``TensorProto.int32_data``, and
``TensorProto.dims``.  A conformant parser must support both
representations on read, even when it only emits the packed form on
write.

Default values and unknown fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protobuf (proto3) treats every field as *optional*: fields that hold
their default value (``0``, empty string, empty message) are omitted
from the wire format.  Unknown fields encountered during parsing must
be skipped according to their wire type and not treated as errors,
which allows files produced by a newer ONNX version to still be read
by older tools.  ``onnx_light`` skips unknown fields by consulting the
wire type byte and reading the appropriate number of bytes (varint,
4 bytes, 8 bytes, or *length* bytes for ``LEN``).

How ONNX uses the wire format
-----------------------------

ONNX defines its messages in ``onnx.proto``.  The top-level
``ModelProto`` aggregates metadata (``ir_version``,
``producer_name``, ...), the opset imports, and the model graph.
A simplified view of the layout for a tiny model is:

.. code-block:: text

    ModelProto (root)
      field 1  (ir_version)        VARINT
      field 2  (producer_name)     LEN  "..."
      field 7  (graph)             LEN  -> GraphProto
        field 1 (node)             LEN  -> NodeProto    [repeated]
          field 1 (input)          LEN  "..."           [repeated]
          field 2 (output)         LEN  "..."           [repeated]
          field 4 (op_type)        LEN  "..."
          field 5 (attribute)      LEN  -> AttributeProto [repeated]
        field 5 (initializer)      LEN  -> TensorProto   [repeated]
        field 11 (input)           LEN  -> ValueInfoProto [repeated]
        field 12 (output)          LEN  -> ValueInfoProto [repeated]

Each nested message is a self-contained length-prefixed substream, so
``onnx_light`` can parse them independently and even read tensor
payloads in parallel using a thread pool.

Tensors and ``raw_data``
~~~~~~~~~~~~~~~~~~~~~~~~

For large numeric initializers, ONNX recommends using the
``raw_data`` field of ``TensorProto`` (field 9, wire type ``LEN``)
rather than the typed scalar arrays.  ``raw_data`` is a byte blob
containing the tensor values stored contiguously in little-endian
order using their native binary representation (for example, four
bytes per ``float32`` value).  Because ``raw_data`` is a single
length-prefixed payload, a parser knows up front how many bytes to
read, can map the bytes into memory without copying them (see
``no_copy=True`` in :ref:`l-howto-load-save-onnx-files`), and can dispatch the
read to a worker thread while it continues parsing other parts of the
file.

External data
~~~~~~~~~~~~~

For models larger than a few gigabytes, ONNX supports storing tensor
payloads in companion files referenced from ``TensorProto`` via the
``external_data`` field (a repeated ``StringStringEntryProto``) and
the ``data_location`` enum.  The ``.onnx`` file then only carries the
metadata (shape, type, ``location``, ``offset``, ``length``, ...) for
those tensors; the actual bytes live in one or more separate files
referenced by ``location``.  ``onnx_light`` implements both sides of
this convention and can additionally split very large initializers
across multiple data files automatically (see the ``location`` and
``max_external_file_size`` options on :func:`onnx_light.onnx.save`).

The 2 GB protobuf limit
~~~~~~~~~~~~~~~~~~~~~~~

``libprotobuf`` enforces a hard 2 GB limit on a single message,
because internal offsets are stored in 32-bit signed integers.
This is the main reason large ONNX models must use external data when
serialized through the standard ``onnx`` package.  ``onnx_light``
uses 64-bit offsets throughout its reader and writer, so it can
produce and consume single ``.onnx`` files larger than 2 GB while
remaining wire-compatible with the protobuf format.

Further reading
---------------

* `Protocol Buffers wire format
  <https://protobuf.dev/programming-guides/encoding/>`_ – the
  authoritative description of varints, wire types, and packed
  encoding.
* `onnx.proto schema
  <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>`_ – the
  ``.proto`` schema that defines every ONNX message and field number.
* :ref:`l-design-differences` – how ``onnx_light`` implements this
  format without depending on ``libprotobuf``.
