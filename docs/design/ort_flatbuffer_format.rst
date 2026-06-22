.. _l-design-ort-flatbuffer-format:

ORT flatbuffer format: parallelization and alignment
====================================================

:epkg:`onnxruntime` defines a compact :epkg:`FlatBuffers` serialization
(``.ort``) of an ONNX model.  Unlike the protobuf wire format used by
``.onnx`` files (see :ref:`l-design-protobuf-format`), a flatbuffer is a
single, flat, contiguous byte buffer in which every table and vector is
addressed through relative offsets, so it can be memory-mapped and read
without an up-front parsing pass.

The C++ reader and writer for ``SerializeFormat::kOrtFlatbuffers`` are not
implemented yet (calls raise ``RuntimeError`` — see
:ref:`l-howto-save-ort-flatbuffers`).  Before implementing them, two
properties that ``onnx_light`` already provides for the ``.onnx`` path had
to be checked against the format itself:

* **parallelization** — can the large tensor payloads be read or written by
  a thread pool, the way the protobuf ``StringStream`` parser
  parallelizes large ``raw_data`` blocks (``ParseOptions::is_parallel``,
  ``StartThreadPool`` / ``WaitForDelayedBlock``)?
* **alignment** — can a tensor's bytes be placed on a power-of-two boundary
  (element-size, SIMD, or page) so that a downstream consumer can mmap or
  zero-copy them, the way :attr:`SerializeOptions.alignment` aligns external
  weights for ``.onnx`` (see :ref:`l-design-no-copy-ownership` and
  :ref:`l-howto-align-external-data-streaming`)?

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

The bulk of a model's bytes live in ``Tensor.raw_data`` (or, for tensors
routed to a companion file, at ``Tensor.external_data_offset``).

Parallelization
---------------

**Reading is parallelizable.**  A flatbuffer never has to be parsed: the
table/vtable graph is walked through offsets, which is cheap and copies no
payload.  Each ``raw_data`` vector is an independent, length-prefixed,
contiguous region at a known offset in the buffer.  Once the offsets of the
initializers have been collected — a lightweight walk — the large blocks are
mutually disjoint and can be dispatched to a thread pool to be copied (or, in
a no-copy load, page-touched) in parallel.  This mirrors exactly how the
protobuf parser submits large ``raw_data`` ``LEN`` fields to its worker pool,
so the future ORT reader can reuse the same
``ParseOptions::is_parallel`` / ``StartThreadPool`` /
``WaitForDelayedBlock`` machinery.  A pure zero-copy / mmap load needs no
threads at all, because the bytes are already in memory at their final
addresses.

**Writing is fundamentally single-threaded.**  ``flatbuffers::FlatBufferBuilder``
builds the buffer bottom-up through a single growing cursor: every child
(vector or table) must be finished before the parent table that references it
is started.  There is no API to append two independent ``raw_data`` vectors to
the same buffer from two threads.  The parallelism available to the writer is
therefore limited to:

* preparing each tensor's bytes (consolidation, dtype conversion, copying)
  on worker threads and then feeding the already-materialized spans to the
  single-threaded builder; and
* routing large tensors to the companion external file through
  ``external_data_offset``, where the external writer can stream bytes in
  parallel exactly like the ``.onnx`` external-data path.

Alignment
---------

**Inline ``raw_data`` is not aligned beyond 4 bytes.**  FlatBuffers aligns a
vector to the size of its element type; ``raw_data`` is declared ``[uint8]``,
so its elements are 1-byte aligned and the vector's data start is only
guaranteed to follow its 32-bit length prefix (4-byte alignment).  The schema
does **not** apply the ``force_align`` attribute to ``raw_data``, so float or
double weights stored inline are not aligned to their natural element size,
let alone to a SIMD (e.g. 64-byte) or page (e.g. 4096-byte) boundary.  A
consumer that needs aligned access to inline ``raw_data`` must therefore
tolerate unaligned reads or copy the bytes out; the inline path cannot offer
the mmap-friendly zero-copy guarantee that ``.onnx`` external data provides.

**The external-data path can be aligned.**  ``Tensor.external_data_offset``
lets a tensor's bytes live in a companion file at a writer-chosen offset.
That offset is controlled entirely by the external writer, which can zero-pad
to any power-of-two boundary — the same technique
:attr:`SerializeOptions.alignment` and
:func:`onnx_light.onnx.align_external_data_streaming` already use for ``.onnx``
external weights.  Alignment for ``.ort`` is therefore achievable, but only
through ``external_data_offset``, not for bytes embedded inside the flatbuffer.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Property
     - Inline ``raw_data``
     - External (``external_data_offset``)
   * - Parallel read
     - Yes — disjoint offset-addressed blocks, reuse the thread pool
     - Yes — independent file regions
   * - Parallel write
     - No — ``FlatBufferBuilder`` has a single cursor
     - Yes — external writer can stream in parallel
   * - Alignment
     - No — ``[uint8]`` is 1-byte aligned, no ``force_align``
     - Yes — writer pads each offset to the requested boundary

Consequences for the (future) onnx-light implementation:

* the reader can reuse the existing parallel-block machinery for ``raw_data``
  and offers true zero-copy / mmap loads;
* the writer is single-threaded at the flatbuffer-assembly step, so any
  parallelism and any alignment guarantee (to honour
  :attr:`SerializeOptions.alignment`) must come from routing large or
  alignment-sensitive tensors through ``external_data_offset``, mirroring the
  ``.onnx`` external-data path.

See also
--------

* :ref:`l-howto-save-ort-flatbuffers` — how to produce a ``.ort`` file today.
* :ref:`l-example-plot-save-ort-flatbuffers` — file-size comparison example.
* :ref:`l-design-protobuf-format` — the protobuf format used by ``.onnx``.
* :ref:`l-design-no-copy-ownership` — buffer alignment and no-copy ownership.
* :ref:`l-design-loading-saving-scenarios` — parallel I/O and alignment recipes.
