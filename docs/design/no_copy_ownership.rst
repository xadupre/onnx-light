.. _l-design-no-copy-ownership:

ModelProto creation and no-copy ownership
==========================================

This page explains exactly who owns tensor ``raw_data`` when ``no_copy=True`` is
enabled, when ownership is transferred, and when memory is released.

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

When memory is released
-----------------------

* Owned mode memory is released when ``ByteSpan`` is destroyed.
* Borrowed + shared-owner mode releases mapped/shared weights when the last
  referencing ``ByteSpan`` is destroyed.
* Borrowed-without-owner mode (inline bytes no-copy) is valid only while the
  caller-managed input bytes object still exists.

Model copy/move behavior
------------------------

Copying or moving model/tensor objects preserves ``ByteSpan`` ownership state:

* owned buffers remain owned by the destination object,
* borrowed pointers remain borrowed,
* shared owner tokens are copied/moved with the tensors.

Therefore, external-data no-copy mappings remain valid across model copies while
references exist, and are released automatically when the last referencing model
object is destroyed.
