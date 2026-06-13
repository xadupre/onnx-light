.. _l-howto-save-ort-flatbuffers:

:html_theme.sidebar_secondary.remove:

How to save a model in the ORT flatbuffer format
================================================

`onnxruntime <https://onnxruntime.ai/>`_ defines a compact flatbuffer
serialization (``.ort``) commonly used in size-constrained deployments
because it can be memory-mapped directly into the runtime and avoids the
protobuf parsing step.

*onnx-light* exposes the format through
:py:class:`onnx_light.onnx.SerializeFormat` (value ``ORT_FLATBUFFERS``).
The reader and writer for that format are not implemented in the C++ core
yet: calls today raise ``RuntimeError``. Until they land, use
:epkg:`onnxruntime` itself to produce the ``.ort`` file.

Convert an ``.onnx`` file to ``.ort`` with onnxruntime
------------------------------------------------------

.. code-block:: python

    import onnxruntime as ort

    so = ort.SessionOptions()
    # Keep the graph structurally identical to the input model.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.optimized_model_filepath = "model.ort"
    so.add_session_config_entry("session.save_model_format", "ORT")

    # Creating the session triggers the optimized-model dump in ORT format.
    ort.InferenceSession("model.onnx", so, providers=["CPUExecutionProvider"])

Planned onnx-light API
----------------------

Once the C++ writer ships, the equivalent one-liner in onnx-light will be:

.. code-block:: python

    import onnx_light.onnx as onnxl

    sopts = onnxl.SerializeOptions()
    sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    model.SerializeToFile("model.ort", sopts)

See also
--------

* :ref:`l-example-plot-save-ort-flatbuffers` - end-to-end example that saves
  the same model in both formats and compares the resulting file sizes.
* :ref:`l-howto-load-save-onnx-files` - load/save recipes for the regular
  ``.onnx`` protobuf format.
