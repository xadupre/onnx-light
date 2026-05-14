.. _l-design-no-copy-ownership:

ModelProto creation and no-copy ownership
==========================================

This page explains exactly who owns tensor ``raw_data`` when ``no_copy=True`` is
enabled, when ownership is transferred, and when memory is released.  It also
documents the in-place tensor consolidation function
:func:`onnx_light.onnx.consolidate_tensors_to_buffer`
(C++: ``ConsolidateTensorsToBuffer``) that produces the same kind of shared-buffer
ownership after loading.

Options class hierarchy
-----------------------

Buffer-related options (alignment, size threshold) are shared across several
operations and are factored into a common base class:

* ``TensorBufferOptions`` – base class with ``raw_data_threshold`` (default: 0)
  and ``alignment`` (default: 0).
* ``ParseOptions`` inherits ``TensorBufferOptions``; its ``raw_data_threshold``
  defaults to 1024 bytes.
* ``SerializeOptions`` inherits ``TensorBufferOptions``; its ``raw_data_threshold``
  defaults to ``kSmallTensorDataThresholdBytes`` (64 bytes).

Any function that accepts a ``TensorBufferOptions`` reference also accepts
``ParseOptions`` or ``SerializeOptions`` objects.

Core objects and where ownership lives
--------------------------------------

Every tensor stores bytes in ``TensorProto::raw_data`` (type ``utils::ByteSpan``).
That ``ByteSpan`` object is a member of the ``TensorProto`` instance, so its
lifetime is tied to the model object graph (``ModelProto -> GraphProto -> TensorProto``).

``ByteSpan`` has two storage modes:

* **Owned mode**: it owns an internal byte buffer.
* **Borrowed mode**: it stores a pointer plus an optional ``std::shared_ptr<void>``
  owner token.

When the borrowed mode also carries a shared owner token, the backing storage
remains alive as long as the corresponding ``TensorProto`` (or a copy of it) is
alive.

When ownership is assigned during parsing
-----------------------------------------

Ownership is assigned while parsing each tensor:

* Inline ``raw_data`` in the protobuf payload:

  * ``no_copy=False``: bytes are copied into ``ByteSpan`` owned mode.
  * ``no_copy=True`` (from in-memory bytes): ``ByteSpan`` borrows from the input bytes buffer.
    No shared owner token is attached, so the caller owns the input bytes lifetime.

* External-data tensors (``data_location=EXTERNAL``):

  * ``no_copy=False``: bytes are copied into ``ByteSpan`` owned mode.
  * ``no_copy=True``: ``TwoFilesStream`` memory-maps (or file-maps on Windows) the
    weights file once, returns a slice pointer and a ``shared_ptr`` owner, and
    ``ByteSpan`` stores both in borrowed mode.

In other words, external-data no-copy transfers lifetime management to shared
ownership held by each tensor, while inline-bytes no-copy keeps lifetime
management with the caller.

In-place consolidation with ConsolidateTensorsToBuffer
------------------------------------------------------

The function ``ConsolidateTensorsToBuffer(ModelProto &model, const TensorBufferOptions &opts)``
(Python: ``onnx_light.onnx.consolidate_tensors_to_buffer``) takes an already-loaded
model and moves all qualifying tensor payloads into a single contiguous buffer,
reproducing the shared-buffer ownership pattern of the no-copy external-data
loading scenario:

1. All tensors whose ``raw_data.size() >= opts.raw_data_threshold`` are selected.
2. A single buffer is allocated.  If ``opts.alignment > 0``, each tensor's offset
   within the buffer is rounded up to the nearest multiple of ``alignment`` bytes,
   and the buffer start itself is aligned to the same boundary.
3. Each tensor's bytes are copied into the buffer at the computed offset.
4. Each tensor's ``raw_data`` is switched to **borrowed mode** with a shared owner
   token pointing to the new buffer, so the buffer stays alive as long as any
   tensor references it.

The function returns a ``std::shared_ptr<uint8_t[]>`` (Python: the function returns
``None``; the buffer lifetime is managed by the tensors).  Tensors that are smaller
than the threshold remain in their original owned or borrowed state.

This is useful for:

* Reducing memory fragmentation after loading a model that was parsed without the
  no-copy option.
* Enabling memory-mapping of all tensor weights after the fact.
* Preparing a model for inference runtimes that benefit from a single contiguous
  tensor weight region.

Usage example (C++)::

    #include "onnx_helper.h"

    // Load a model normally.
    ModelProto model;
    utils::FileStream stream("model.onnx");
    ParseOptions parse_opts;
    ParseProtoFromStream(model, stream, parse_opts);

    // Consolidate all tensors into a single 64-byte-aligned buffer.
    TensorBufferOptions buf_opts;
    buf_opts.alignment = 64;
    auto buf = ConsolidateTensorsToBuffer(model, buf_opts);
    // buf is now the shared buffer; the model's tensors borrow from it.

Usage example (Python)::

    import onnx_light.onnx as onnxl

    model = onnxl.load("model.onnx")

    opts = onnxl.TensorBufferOptions()
    opts.alignment = 64        # optional: align each tensor to 64-byte boundaries
    opts.raw_data_threshold = 1024  # optional: only consolidate tensors >= 1 KB
    onnxl.consolidate_tensors_to_buffer(model, opts)
    # After this call, large tensors borrow from a single shared buffer.

Loading scenarios summary
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Load scenario
     - ``no_copy``
     - ``TensorProto::raw_data`` storage
     - Who must keep backing memory alive
   * - ``onnxl.load("model.onnx")`` (single-file)
     - ``False`` (default)
     - Owned copy
     - ``TensorProto`` / model
   * - ``onnxl.load("model.onnx", no_copy=True)`` (single-file)
     - ``True``
     - Owned copy (file stream cannot borrow inline payload)
     - ``TensorProto`` / model
   * - ``onnxl.load(model_bytes, no_copy=False)``
     - ``False``
     - Owned copy
     - ``TensorProto`` / model
   * - ``onnxl.load(model_bytes, no_copy=True)``
     - ``True``
     - Borrowed pointer into ``model_bytes``
     - **Caller** (must keep ``model_bytes`` alive)
   * - ``onnxl.load("model.onnx", load_external_data=True, no_copy=False)``
     - ``False``
     - Owned copy
     - ``TensorProto`` / model
   * - ``onnxl.load("model.onnx", load_external_data=True, no_copy=True)``
     - ``True``
     - Borrowed pointer + shared owner token
     - Shared ownership via ``ByteSpan`` in model tensors
   * - ``onnxl.consolidate_tensors_to_buffer(model)`` (post-load)
     - n/a
     - Borrowed pointer + shared owner token
     - Shared ownership via ``ByteSpan`` in model tensors

When memory is released
-----------------------

* Owned mode memory is released when ``ByteSpan`` is destroyed.
* **Copy scenarios** (``no_copy=False``) always use owned storage; memory is
  released when each ``ByteSpan`` is destroyed with the model/tensor object.
* **No-copy + external-data** stores borrowed pointers with a shared owner
  token; mapped/shared weights are released only when the last referencing
  ``ByteSpan`` is destroyed.
* **No-copy + inline bytes** stores borrowed pointers without owner token;
  tensors are valid only while the caller-managed input bytes object exists.
* **ConsolidateTensorsToBuffer** creates a single shared buffer and stores a
  shared owner token in each tensor's ``ByteSpan``; the buffer is released when
  all referencing ``ByteSpan`` objects (and any external ``shared_ptr`` returned
  by the C++ function) are destroyed.

Model copy/move behavior
------------------------

Moving model/tensor objects preserves ``ByteSpan`` ownership state:

* owned buffers remain owned by the destination object,
* borrowed pointers remain borrowed,
* shared owner tokens (when present in no-copy external-data) move with the tensors.

This means:

* In **copy scenarios**, model data remains owned by model objects.
* In **no-copy external-data scenarios**, data remains valid after the
  ``TwoFilesStream`` parser object is destroyed because each tensor keeps a
  shared owner token for the mapped buffer.
* In **no-copy inline-bytes scenarios**, tensors still depend on the original
  caller-provided bytes object lifetime.
* After **ConsolidateTensorsToBuffer**, tensors remain valid regardless of
  whether the caller retains the returned ``shared_ptr`` because each tensor
  holds its own owner token.
