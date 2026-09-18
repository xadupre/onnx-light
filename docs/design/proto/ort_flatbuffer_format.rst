.. _l-design-ort-flatbuffer-format:

ORT flatbuffer format: parallelization and alignment
====================================================

:epkg:`onnxruntime` defines a compact :epkg:`FlatBuffers` serialization
(``.ort``) of an ONNX model.  Unlike the protobuf wire format used by
``.onnx`` files (see :ref:`l-design-protobuf-format`), a flatbuffer is a
single, flat, contiguous byte buffer in which tables and vectors are
addressed through relative offsets. A consumer can access the serialized
data without first decoding a protobuf message.

The native C++ writer supports ``SerializeFormat::kOrtFlatbuffers`` and
produces an ``ORTM`` buffer with ORT format version ``4``. It is exposed
through the file, memory, and file-descriptor serialization APIs (see
:ref:`l-howto-save-ort-flatbuffers`). A full ONNX Runtime build can load the
result, resolving the graph and upgrading kernel metadata during loading.
Minimal runtime builds cannot execute these version-4 files directly.
The native onnx-light reader accepts both native version-4 files and
current ORT version-6 files, reconstructing an ONNX model for execution
or reserialization.

Two layout properties matter for large tensor payloads: parallel assembly
and alignment of the serialized tensor bytes.

The relevant subset of the onnxruntime schema (``ort.fbs``) is::

    table Tensor {
      name:string;
      doc_string:string;
      dims:[int64];
      data_type:TensorDataType;
      raw_data:[uint8];
      string_data:[string];
      // offset into an external data file so that data >2GB can be handled;
      // -1 when the bytes are stored inline in raw_data.
      external_data_offset:int64 = -1;
    }

The native writer embeds tensor data inline. Although the format includes
``external_data_offset``, the writer does not support external sidecar
output. Unresolved external tensor payloads are rejected: their bytes must
be loaded into ``raw_data`` before serialization.

Reading and reconstruction
--------------------------

``ModelProto.ParseFromString(data, popts)`` and
``ModelProto.ParseFromFile(path, popts)`` select the native reader when
``popts.format`` is ``SerializeFormat.ORT_FLATBUFFERS``.
The reader reconstructs graph structure and tensor data rather than the
original protobuf wire representation. ORT graph normalization and missing
original names or documentation mean that protobuf byte-for-byte roundtrips
are not guaranteed. Execution results, not serialized protobuf equality,
are the appropriate roundtrip check.

The reader honors the configured tensor-byte and recursion limits and
invokes raw-data and node callbacks. It owns copies of decoded tensor bytes
even if ``ParseOptions.no_copy`` is set. The offset-addressed format does
not imply that this reader offers zero-copy or memory-mapped tensor views.

External tensor offsets are explicitly rejected. The reader does not
automatically open a companion file or guess an external-data filename.

Binary-size budget
------------------

The native reader and writer add serialization code to ``lib_onnx_proto``.
The previous Linux CI limits predated the implemented ORT codec: 1,193,944
installed bytes, 822,634 ``.text`` bytes, and 760 defined dynamic symbols.
The codec receives a bounded allowance of 192 KiB installed, 128 KiB of
``.text``, and 32 symbols above those limits. The checks remain enforced;
the shared-library dependency allowlist is unchanged.

The Linux Release CI measurement for commit ``bbeaeb6b`` is:

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Metric
     - Measured with native ORT
     - CI maximum
   * - Stripped installed bytes
     - 1,363,432
     - 1,390,552
   * - ``.text`` bytes
     - 940,986
     - 953,706
   * - Defined dynamic symbols
     - 782
     - 792

These limits account for the added functionality rather than a toolchain
baseline change. The codec adds no FlatBuffers or ONNX Runtime shared-library
dependency.

Parallelization
---------------

**Native flatbuffer assembly is single-threaded.** The writer constructs a
single buffer and its relative offsets. Setting
``SerializeOptions.num_threads`` does not parallelize this assembly.

The format itself does not prevent parallel preparation of independent
tensor payloads or parallel copying of known payload regions when reading.
These possibilities are distinct from implemented behavior: the native
writer does not provide parallel flatbuffer assembly, and the native reader
decodes sequentially even when ``ParseOptions.num_threads`` is set.

Alignment
---------

**Inline tensor byte offsets can be explicitly aligned.** The schema
declares ``raw_data`` as ``[uint8]`` without a ``force_align`` attribute.
That does not prohibit the writer from requesting stronger alignment while
building the vector. The native writer honors
``SerializeOptions.alignment`` for inline tensor payload offsets; an
external sidecar is not required.

The alignment is relative to the start of the serialized buffer. When the
buffer is written at file offset zero, the tensor's file offset has the
requested alignment. When writing through a descriptor positioned elsewhere,
the starting file offset must also be suitably aligned to preserve it.

**Offset alignment does not guarantee pointer alignment.** If a payload
starts at an aligned offset but the buffer's base address is unaligned, the
payload's address is still unaligned. In particular, a Python ``bytes``
allocation returned by ``SerializeToString`` is not guaranteed to have the
requested base alignment. Memory mapping or another allocation strategy must
provide an appropriately aligned base before a consumer can rely on aligned
in-memory access. Alignment alone does not promise that ONNX Runtime will
use every tensor without copying.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Property
     - Native onnx-light support
   * - Serialization targets
     - File, memory, and file descriptor
   * - Tensor storage
     - Inline; unresolved external data and sidecar output rejected
   * - Parallel assembly
     - No; ``num_threads`` does not parallelize assembly
   * - Inline payload offset alignment
     - Yes, through ``SerializeOptions.alignment``
   * - Buffer base pointer alignment
     - Not guaranteed by serialization
   * - ORT reading
     - Version 4 and version 6; reconstructs an ONNX model
   * - Parallel decoding
     - No; ``num_threads`` does not parallelize decoding
   * - Zero-copy decoding
     - No; tensor bytes are copied even with ``no_copy``
   * - External tensor reading
     - Unsupported offsets are explicitly rejected

See also
--------

* :ref:`l-howto-save-ort-flatbuffers` — how to produce a ``.ort`` file today.
* :ref:`l-example-plot-save-ort-flatbuffers` — file-size comparison example.
* :ref:`l-design-protobuf-format` — the protobuf format used by ``.onnx``.
* :ref:`l-design-no-copy-ownership` — buffer alignment and no-copy ownership.
* :ref:`l-design-loading-saving-scenarios` — parallel I/O and alignment recipes.
