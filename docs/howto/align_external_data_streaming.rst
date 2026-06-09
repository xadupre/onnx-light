.. _l-howto-align-external-data-streaming:

How to re-align external weights without loading them in memory
===============================================================

This page documents the recommended recipe for re-aligning the tensor
offsets of an existing two-file (or multi-file) ONNX model **without ever
loading the weights in RAM**.  The driver is
:func:`onnx_light.onnx.align_external_data_streaming`.

When to use it
--------------

Use this how-to when **all** of the following are true:

* The model is already saved on disk as a ``(.onnx, .data[, .data.1, ...])``
  pair where the initializers point at the weights file(s) through their
  ``external_data`` entries.
* You want every tensor offset in the destination weights file to be aligned
  to a power-of-two boundary (for example ``4096`` for page-aligned mmap or
  accelerator-friendly loading).
* The model is too large — or the host too memory-constrained — to load the
  full set of weights into memory just to re-serialize them.

Recipe
------

The function parses only the proto metadata and streams each tensor's bytes
from the source weights file to the destination weights file in chunks of
``chunk_size`` bytes, zero-padding to the requested alignment between
tensors.  Peak heap usage is therefore bounded by *(metadata size +
chunk_size)*, independent of the total weights size.

.. code-block:: python

    import onnx_light.onnx as onnxl

    total_bytes = onnxl.align_external_data_streaming(
        src_onnx_path="src.onnx",
        dst_onnx_path="dst.onnx",
        dst_weights_path="dst.data",
        alignment=4096,
        chunk_size=4 * 1024 * 1024,
    )
    print(f"Wrote {total_bytes} bytes (including alignment padding) to dst.data")

After the call:

* ``dst.onnx`` is a fresh ``.onnx`` file whose initializers all reference
  ``dst.data`` with aligned offsets.
* ``dst.data`` is a **single** consolidated weights file — even when the
  source spread tensors across multiple ``external_data.location`` files.
* The source files (``src.onnx`` and its ``.data`` companions) are not
  modified.

Comparison with the in-memory variant
-------------------------------------

The "usual" path is to ``load(load_external_data=True)`` the model and then
re-serialize it with :attr:`SerializeOptions.alignment` set to the desired
value.  Functionally the two approaches produce byte-equivalent tensor
payloads at the same aligned offsets; the only difference is the peak
memory footprint:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - :func:`~onnx_light.onnx.align_external_data_streaming`
     - :func:`~onnx_light.onnx.load` + ``SerializeToFile``
   * - Peak heap
     - O(metadata + ``chunk_size``)
     - O(total weights size)
   * - Source files
     - Untouched
     - Untouched
   * - Destination weights
     - Single consolidated file
     - Single consolidated file (with ``use_external_data_location=False``)
   * - Recommended when
     - Weights do not fit in RAM
     - Weights comfortably fit in RAM

See :ref:`l-example-plot-align-external-data-streaming` for a runnable
benchmark of both approaches.

See also
--------

* :func:`onnx_light.onnx.align_external_data_streaming` – API reference.
* :ref:`l-design-loading-saving-scenarios` – design notes on complex
  load/save scenarios this function was designed for.
* :ref:`l-howto-save-model-with-shared-external-data` – companion how-to
  for sharing weights across multiple models.
* :ref:`l-howto-load-save-onnx-files` – common load/save patterns.
