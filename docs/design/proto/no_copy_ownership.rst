.. _l-design-no-copy-ownership:

ModelProto creation and no-copy ownership
==========================================

This page explains exactly who owns tensor ``raw_data`` when ``no_copy=True`` is
enabled, when ownership is transferred, and when memory is released.  It also
documents the in-place tensor consolidation function
:func:`onnx_light.onnx.consolidate_tensors_to_buffer`
(C++: :cpp:func:`~onnx_light::ConsolidateTensorsToBuffer`) that produces the same kind of shared-buffer
ownership after loading.

Options class hierarchy
-----------------------

Buffer-related options (alignment, size threshold) are shared across several
operations and are factored into a common base class:

* :cpp:struct:`~onnx_light::TensorBufferOptions` – base class with
  ``raw_data_threshold`` (default: ``kSmallTensorDataThresholdBytes``, or 64 bytes)
  and ``alignment`` (default: 0).
* :cpp:struct:`~onnx_light::ParseOptions` and
  :cpp:struct:`~onnx_light::SerializeOptions` inherit
  :cpp:struct:`~onnx_light::TensorBufferOptions` without changing these defaults.

Any function that accepts a :cpp:struct:`~onnx_light::TensorBufferOptions` reference also accepts
:cpp:struct:`~onnx_light::ParseOptions` or :cpp:struct:`~onnx_light::SerializeOptions` objects.

Core objects and where ownership lives
--------------------------------------

When a tensor payload is represented by ``TensorProto::raw_data``, its bytes are
stored in a :cpp:class:`~onnx_light::utils::ByteSpan`.
That :cpp:class:`~onnx_light::utils::ByteSpan` object is a member of the :class:`~onnx_light.onnx_lib.TensorProto` instance, so its
lifetime is tied to the tensor, whether standalone or in the model object graph
(``ModelProto -> GraphProto -> TensorProto``).
:cpp:class:`~onnx_light::utils::ByteSpan` has two storage modes:

* **Owned mode**: it owns an internal byte buffer.
* **Borrowed mode**: it stores a pointer plus an optional ``std::shared_ptr<void>``
  owner token.

When the borrowed mode also carries a shared owner token, the backing storage
remains alive as long as the corresponding :class:`~onnx_light.onnx_lib.TensorProto` (or a copy of it) is
alive.

Attaching a custom deleter to a TensorProto
-------------------------------------------

Rather than managing lifetime through a raw ``std::shared_ptr<void>`` owner token,
you can attach an arbitrary cleanup function directly to a tensor via
:cpp:func:`~onnx_light::TensorProto::set_raw_data_with_deleter` or the lower-level
:cpp:func:`~onnx_light::utils::ByteSpan::assign_with_deleter`.  The deleter is called
exactly once when the last copy of the owner token inside the ByteSpan is destroyed —
that is, after every tensor sharing that token goes out of scope or its ``raw_data``
is overwritten or cleared.

The deleter is a zero-argument callable (lambda, function pointer, or functor)
returning ``void``.  Internally it is wrapped in a ``std::shared_ptr<void>`` with a
custom deleter and stored as the ``owner_`` token, so all the copy/move/clear
semantics of :cpp:class:`~onnx_light::utils::ByteSpan` apply without any change.

C++ example::

    #include "onnx_proto/onnx.h"
    #include <cstdlib>  // for std::malloc / std::free

    // Allocate tensor data outside of TensorProto's normal allocators.
    const size_t n_bytes = 4 * sizeof(float);
    uint8_t *buf = static_cast<uint8_t *>(std::malloc(n_bytes));
    // … fill buf …

    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.ref_dims().push_back(4);

    // Hand the buffer to the tensor.  free() will be called when the tensor is
    // destroyed (or when the raw_data is overwritten/cleared).
    tensor.set_raw_data_with_deleter(buf, n_bytes, [buf]() { std::free(buf); });

    // The tensor now owns buf's lifetime through the deleter.  buf must not be
    // freed elsewhere.

A no-op deleter is valid, but if no cleanup is required, directly borrowing the
buffer avoids creating a shared owner token::

    tensor.ref_raw_data().assign_borrowed(ptr, sz);

The lower-level :cpp:func:`~onnx_light::utils::ByteSpan::assign_with_deleter` works the same way::

    span.assign_with_deleter(ptr, sz, []() { /* custom cleanup */ });

To attach a deleter to data that is already stored (without replacing the bytes or
changing the storage mode), use
:cpp:func:`~onnx_light::TensorProto::attach_raw_data_deleter` or the lower-level
:cpp:func:`~onnx_light::utils::ByteSpan::attach_deleter`::

    // Preserve any current owner when raw_data is borrowed.
    auto owner = tensor.ref_raw_data().owner();
    tensor.attach_raw_data_deleter([owner]() { /* custom cleanup */ });

Attaching a deleter replaces any existing shared owner token.  For borrowed data,
the new deleter must therefore keep the backing storage valid until it runs.

Taking ownership of tensor data while parsing
---------------------------------------------

:cpp:struct:`~onnx_light::ParseOptions` exposes a ``raw_data_callback`` hook that is
invoked for every :class:`~onnx_light.onnx_lib.TensorProto` that has ``raw_data`` once
that data has been parsed (inline or external).  The callback receives the freshly
parsed tensor and returns a deleter (a zero-argument callable); when the returned
deleter is non-empty it is attached to the tensor's ``raw_data`` via
:cpp:func:`~onnx_light::utils::ByteSpan::attach_deleter`, so it fires once when the buffer
is released.  Return an empty ``std::function`` to leave ownership unchanged.

The callback works with owned or borrowed bytes without moving them.  A non-empty
deleter replaces the current owner token, so it must retain or release any backing
storage needed by borrowed bytes::

    ParseOptions options;
    options.raw_data_callback = [](TensorProto &tensor,
                                   GraphProto *graph) -> std::function<void()> {
      // Inspect tensor.ref_raw_data() and the parent graph (``graph`` is nullptr for a
      // standalone tensor); optionally relocate it (e.g. to a device) and return the matching
      // cleanup. Preserve the current owner when the bytes remain borrowed.
      auto owner = tensor.ref_raw_data().owner();
      return [owner, name = tensor.ref_name()]() {
        /* release resources */
      };
    };
    model.ParseFromString(bytes, options);

By default ``raw_data_callback`` is empty and parsing behaves exactly as before.

The same hook is available from Python as
:attr:`onnx_light.onnx.ParseOptions.raw_data_callback`.  The callback is called as
``fn(tensor, graph)`` with the freshly parsed :class:`~onnx_light.onnx.TensorProto` and its
parent :class:`~onnx_light.onnx.GraphProto` (or ``None`` for a standalone tensor) and must return
either ``None`` (ownership unchanged) or a zero-argument callable used as the deleter::

    import onnx_light.onnx as onnx

    options = onnx.ParseOptions()
    # print() returns None, so the tensor keeps its default ownership.
    options.raw_data_callback = lambda tensor, graph: print(tensor.name, len(tensor.raw_data))

    model = onnx.ModelProto()
    model.ParseFromString(serialized, options)

For the common case of *only* reporting progress while keeping the default
allocation, assign a :class:`onnx_light.onnx.RawDataCallback` instead.  It
forwards every parsed tensor to its ``on_tensor`` callable and always returns
``None``, so tensor ownership is left to the default allocator::

    options = onnx.ParseOptions()
    options.raw_data_callback = onnx.RawDataCallback(
        lambda tensor: print(tensor.name, len(tensor.raw_data))
    )

See the :ref:`l-example-plot-raw-data-callback` gallery example for a complete walk-through.

No-copy ownership while parsing
-------------------------------

Ownership is assigned while parsing each tensor:

* Inline ``raw_data`` in the protobuf payload:

  * ``no_copy=False``: bytes are copied into :cpp:class:`~onnx_light::utils::ByteSpan` owned mode.
  * ``no_copy=True`` (from in-memory bytes): :cpp:class:`~onnx_light::utils::ByteSpan` borrows from the input bytes buffer.
    No shared owner token is attached, so the caller owns the input bytes lifetime.

* External-data tensors (``data_location=EXTERNAL``):

  * ``no_copy=False``: bytes are copied into :cpp:class:`~onnx_light::utils::ByteSpan` owned mode.
  * ``no_copy=True``: :cpp:class:`~onnx_light::utils::TwoFilesStream` memory-maps (or file-maps on Windows) the
    weights file once, returns a slice pointer and a ``shared_ptr`` owner, and
    :cpp:class:`~onnx_light::utils::ByteSpan` stores both in borrowed mode.

In other words, external-data no-copy transfers lifetime management to shared
ownership held by each tensor, while inline-bytes no-copy keeps lifetime
management with the caller.

In-place consolidation with ConsolidateTensorsToBuffer
------------------------------------------------------

The function ``ConsolidateTensorsToBuffer(ModelProto &model, const TensorBufferOptions &opts)``
(Python: :func:`onnx_light.onnx.consolidate_tensors_to_buffer`) takes an already-loaded
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
     - :class:`~onnx_light.onnx_lib.TensorProto` / model
   * - ``onnxl.load("model.onnx", no_copy=True)`` (single-file)
     - ``True``
     - Owned copy (file stream cannot borrow inline payload)
     - :class:`~onnx_light.onnx_lib.TensorProto` / model
   * - ``onnxl.load(model_bytes, no_copy=False)``
     - ``False``
     - Owned copy
     - :class:`~onnx_light.onnx_lib.TensorProto` / model
   * - ``onnxl.load(model_bytes, no_copy=True)``
     - ``True``
     - Borrowed pointer into ``model_bytes``
     - **Caller** (must keep ``model_bytes`` alive)
   * - ``onnxl.load("model.onnx", load_external_data=True, no_copy=False)``
     - ``False``
     - Owned copy
     - :class:`~onnx_light.onnx_lib.TensorProto` / model
   * - ``onnxl.load("model.onnx", load_external_data=True, no_copy=True)``
     - ``True``
     - Borrowed pointer + shared owner token
     - Shared ownership via :cpp:class:`~onnx_light::utils::ByteSpan` in model tensors
   * - ``onnxl.consolidate_tensors_to_buffer(model)`` (post-load)
     - n/a
     - Borrowed pointer + shared owner token
     - Shared ownership via :cpp:class:`~onnx_light::utils::ByteSpan` in model tensors

When memory is released
-----------------------

* Owned mode memory is released when :cpp:class:`~onnx_light::utils::ByteSpan` is destroyed.
* **Copy scenarios** (``no_copy=False``) always use owned storage; memory is
  released when each :cpp:class:`~onnx_light::utils::ByteSpan` is destroyed with the model/tensor object.
* **No-copy + external-data** stores borrowed pointers with a shared owner
  token; mapped/shared weights are released only when the last referencing
  :cpp:class:`~onnx_light::utils::ByteSpan` is destroyed.
* **No-copy + inline bytes** stores borrowed pointers without owner token;
  tensors are valid only while the caller-managed input bytes object exists.
* **ConsolidateTensorsToBuffer** creates a single shared buffer and stores a
  shared owner token in each tensor's :cpp:class:`~onnx_light::utils::ByteSpan`; the buffer is released when
  all referencing :cpp:class:`~onnx_light::utils::ByteSpan` objects (and any external ``shared_ptr`` returned
  by the C++ function) are destroyed.

Model copy/move behavior
------------------------

Moving model/tensor objects preserves :cpp:class:`~onnx_light::utils::ByteSpan` ownership state:

* owned buffers remain owned by the destination object,
* borrowed pointers remain borrowed,
* shared owner tokens (when present in no-copy external-data) move with the tensors.

This means:

* In **copy scenarios**, model data remains owned by model objects.
* In **no-copy external-data scenarios**, data remains valid after the
  :cpp:class:`~onnx_light::utils::TwoFilesStream` parser object is destroyed because each tensor keeps a
  shared owner token for the mapped buffer.
* In **no-copy inline-bytes scenarios**, tensors still depend on the original
  caller-provided bytes object lifetime.
* After **ConsolidateTensorsToBuffer**, tensors remain valid regardless of
  whether the caller retains the returned ``shared_ptr`` because each tensor
  holds its own owner token.
