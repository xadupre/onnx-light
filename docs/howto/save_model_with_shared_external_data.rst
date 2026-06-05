.. _l-howto-save-model-with-shared-external-data:

How to save a model that shares weights with another on-disk model
==================================================================

This page documents the recipe for saving a new model that **reuses
already-external weights** of a previously saved model, without copying
those weights a second time on disk.  The driver is
:func:`onnx_light.onnx.save_model_with_shared_external_data`.

When to use it
--------------

Use this how-to when **all** of the following are true:

* A first model has already been written to disk as a
  ``(first.onnx, first.onnx.data[, ...])`` pair.
* The first model is then loaded **without** external data
  (``load_external_data=False``), so its initializers still carry their
  original ``external_data`` metadata (``location``, ``offset``,
  ``length``).
* A second model is then built that mixes some of those reused
  initializers with brand-new initializers carrying inline ``raw_data``.
* You want to save the second model next to the first one without
  duplicating the bytes of the reused initializers.

A typical use case is producing a *variant* of an existing model (for
example with a different head, a quantized branch, or extra adapters)
while keeping a single physical copy of the shared weights on disk.

Recipe
------

Initializers already marked ``EXTERNAL`` are written out **as-is**: their
``external_data`` entries are kept unchanged, so they keep referencing the
weights file they already pointed at.  No byte is copied from that file.
New initializers carrying inline ``raw_data`` are written to a single
secondary weights file at ``<dst_onnx_path>.data``, at aligned offsets.

.. code-block:: python

    import numpy as np
    import onnx_light.onnx as onnxl
    import onnx_light.onnx.helper as oh
    from onnx_light.onnx_lib import SerializeOptions

    # 1) Load the first model WITHOUT external data so its initializers keep
    #    their external_data metadata pointing at first.onnx.data.
    first = onnxl.load("first.onnx", load_external_data=False)

    # 2) Build a second model that reuses every initializer of the first
    #    one and adds a brand-new inline initializer.
    new_arr = np.full((5,), 7.0, dtype=np.float32)
    new_init = oh.make_tensor(
        name="new_weight",
        data_type=onnxl.TensorProto.FLOAT,
        dims=new_arr.shape,
        vals=new_arr.tobytes(),
        raw=True,
    )
    graph = oh.make_graph(
        [], "g", [], [],
        initializer=[*first.graph.initializer, new_init],
    )
    second = oh.make_model(graph, producer_name="second")

    # 3) Save the second model next to the first one.  Reused initializers
    #    keep pointing at first.onnx.data; new initializers land in
    #    second.onnx.data at 4096-aligned offsets.
    opts = SerializeOptions()
    opts.alignment = 4096
    bytes_written = onnxl.save_model_with_shared_external_data(
        model=second,
        dst_onnx_path="second.onnx",
        options=opts,
    )
    print(f"Wrote {bytes_written} bytes to second.onnx.data")

After the call:

* ``second.onnx`` references the reused initializers through their
  original ``external_data`` entries (pointing at ``first.onnx.data``) and
  the new initializers through ``second.onnx.data``.
* ``second.onnx.data`` contains *only* the new weights and is not created
  at all when every initializer is reused (``bytes_written`` is ``0``).
* ``first.onnx`` and ``first.onnx.data`` are not modified.
* The ``model`` passed in is mutated in place: new initializers no longer
  carry inline ``raw_data`` after the call; they reference
  ``second.onnx.data`` instead.

Important constraints
---------------------

* The caller is responsible for the recorded ``location`` of each reused
  initializer remaining resolvable relative to ``dst_onnx_path``'s parent
  directory.  In the recipe above, ``second.onnx`` is saved next to
  ``first.onnx``, so the original ``location="first.onnx.data"`` still
  resolves correctly.  Saving the destination in a different directory
  would require either pre-rewriting the ``location`` entries to a
  relative/absolute path that still resolves, or keeping the destination
  next to the source.
* Only ``SerializeOptions.alignment`` (inherited from
  :class:`onnx_light.onnx_lib.TensorBufferOptions`) is honored.  Use a
  power of two such as ``4096`` for mmap-friendly pages; ``0`` disables
  alignment.

Comparison with the streaming alignment recipe
----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Typical scenario
   * - :func:`onnx_light.onnx.align_external_data_streaming`
     - Re-align an existing two-file (or multi-file) model into a new
       ``(dst.onnx, dst.data)`` pair without ever loading the weights in
       RAM.  Use it as a one-shot post-processing step.
   * - :func:`onnx_light.onnx.save_model_with_shared_external_data`
     - Save a *new* model that mixes already-external initializers
       (kept on disk as-is) with brand-new inline initializers (written
       to a fresh ``.data`` file next to the destination ``.onnx``).

See also
--------

* :func:`onnx_light.onnx.save_model_with_shared_external_data` – API
  reference.
* :ref:`l-design-loading-saving-scenarios` – design notes on complex
  load/save scenarios this function was designed for.
* :ref:`l-howto-align-external-data-streaming` – companion how-to for
  re-aligning an existing model without loading its weights.
* :ref:`l-howto-load-save-onnx-files` – common load/save patterns.
