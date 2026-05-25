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

``TensorProto`` is the central message used everywhere a tensor value
appears: graph initializers, attribute defaults of type ``TENSOR``,
constant nodes, and external-data references.  Its on-wire layout
illustrates almost every feature of the protobuf format described
above.  The fields most commonly seen are:

.. list-table::
   :header-rows: 1
   :widths: 8 22 12 58

   * - #
     - Field
     - Wire type
     - Description
   * - 1
     - ``dims`` (``repeated int64``)
     - ``LEN`` (packed)
     - Tensor shape.  Encoded as a single length-prefixed block of
       varints; an unranked tensor has no ``dims`` field at all
       (different from a scalar, which has zero ``dims`` entries
       inside an empty packed block).
   * - 2
     - ``data_type`` (``int32`` enum)
     - ``VARINT``
     - Element type from ``TensorProto.DataType``
       (1 = ``FLOAT``, 7 = ``INT64``, 11 = ``DOUBLE``, ...).  Required
       for any tensor that carries data.
   * - 3
     - ``segment`` (``Segment``)
     - ``LEN``
     - Optional ``{begin, end}`` pair used by legacy chunked tensors;
       rarely populated by modern producers.
   * - 4
     - ``float_data`` (``repeated float``)
     - ``LEN`` (packed)
     - Typed payload for ``FLOAT``.  Each element is 4 little-endian
       bytes; the packed block size therefore equals
       ``4 * numel(tensor)``.
   * - 5
     - ``int32_data`` (``repeated int32``)
     - ``LEN`` (packed)
     - Typed payload for ``INT32``, ``UINT8``, ``INT8``, ``UINT16``,
       ``INT16``, ``BOOL``, ``FLOAT16``, ``BFLOAT16`` and the small
       ``FLOAT8`` / ``INT4`` / ``UINT4`` types (each element widened
       to a varint).
   * - 6
     - ``string_data`` (``repeated bytes``)
     - ``LEN``
     - Typed payload for ``STRING``.  Each element is its own
       length-prefixed block, so this field is **unpacked** (one tag
       per element); the order matches the row-major iteration of
       ``dims``.
   * - 7
     - ``int64_data`` (``repeated int64``)
     - ``LEN`` (packed)
     - Typed payload for ``INT64``.
   * - 8
     - ``name`` (``string``)
     - ``LEN``
     - Tensor name; matched against ``input``/``output`` names in the
       enclosing graph and against ``external_data`` keys.
   * - 9
     - ``raw_data`` (``bytes``)
     - ``LEN``
     - Single contiguous blob of element bytes in little-endian order
       and the native binary representation of ``data_type``
       (4 bytes per ``FLOAT``, 8 bytes per ``INT64``, 2 bytes per
       ``FLOAT16``, packed nibbles for ``INT4`` / ``UINT4``, ...).
       Mutually exclusive with the typed ``*_data`` fields.
   * - 10
     - ``double_data`` (``repeated double``)
     - ``LEN`` (packed)
     - Typed payload for ``DOUBLE`` (8 bytes per element on the wire).
   * - 11
     - ``uint64_data`` (``repeated uint64``)
     - ``LEN`` (packed)
     - Typed payload for ``UINT32`` and ``UINT64``.
   * - 12
     - ``doc_string`` (``string``)
     - ``LEN``
     - Optional human-readable description.
   * - 13
     - ``external_data`` (``repeated StringStringEntryProto``)
     - ``LEN``
     - Key/value pairs (``location``, ``offset``, ``length``,
       ``checksum``, ...) used when ``data_location`` is set to
       ``EXTERNAL``.
   * - 14
     - ``data_location`` (``Location`` enum)
     - ``VARINT``
     - ``DEFAULT`` (0) when the payload is inline (in ``raw_data`` or
       a typed field), or ``EXTERNAL`` (1) when it lives in a
       companion file pointed to by ``external_data``.
   * - 16
     - ``metadata_props`` (``repeated StringStringEntryProto``)
     - ``LEN``
     - Free-form key/value annotations attached to the tensor.

In the simplest case a ``FLOAT`` initializer of shape ``[2, 3]``
containing six values is written, in order, as:

.. code-block:: text

    TensorProto
      field 1  (dims)       LEN  3 bytes  -> packed varints: 2, 3
      field 2  (data_type)  VARINT        -> 1   (FLOAT)
      field 8  (name)       LEN  N bytes  -> "weight"
      field 9  (raw_data)   LEN  24 bytes -> six little-endian float32 values

The typed scalar arrays (``float_data`` / ``int32_data`` /
``int64_data`` / ``double_data`` / ``uint64_data``) and ``raw_data``
are **mutually exclusive**: a parser must read whichever one is
present and use ``data_type`` to interpret the bytes.  Modern
producers almost always use ``raw_data`` because:

* it is a single ``LEN`` payload — its byte size is known up front,
  making memory pre-allocation trivial;
* the bytes are already in the on-disk layout, so they can be
  ``memcpy``-ed (or, with ``no_copy=True`` in
  :ref:`l-howto-load-save-onnx-files`, simply pointed at) into the
  destination buffer;
* ``onnx_light`` can hand the block to a worker thread and keep
  parsing the rest of the message in parallel.

The ``raw_data`` encoding has two notable subtleties: it is always
little-endian regardless of the host byte order, and sub-byte types
(``BOOL``, ``INT4``, ``UINT4``, the ``FLOAT4E2M1`` family) are
packed two values per byte in the order specified by ``onnx.proto``.

Sparse tensors are carried by ``SparseTensorProto`` (used in
``GraphProto.sparse_initializer``), which simply embeds two
``TensorProto`` substreams — one for the non-zero values and one for
the coordinate indices — alongside the dense ``dims``.

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
