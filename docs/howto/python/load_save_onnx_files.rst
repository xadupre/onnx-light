.. _l-howto-load-save-onnx-files:
.. _l-howto-load-save-onnx-files-python:

:html_theme.sidebar_secondary.remove:

How to load and save ONNX files
===============================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/load_save_onnx_files`.

This page covers the *onnx-light* Python load/save API for one-file
``.onnx`` models, two-file models with external tensor data, split external
data across multiple files, alignment options, and the thread-count options for
larger models.

Common load/save patterns
-------------------------

Load one file
^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    model = onnxl.load("model.onnx")

Load two files
^^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    model = onnxl.load(
        "model.onnx",
        location="model.onnx.data",
        load_external_data=True,
    )

Save one file
^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    onnxl.save(model, "out.onnx")

Save two files
^^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    onnxl.save(model, "out.onnx", location="out.onnx.data")

Load/save split external files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    onnxl.save(
        model,
        "out.onnx",
        location="out.onnx.data",
        max_external_file_size=2 * 1024 ** 3,
    )
    model = onnxl.load("out.onnx", load_external_data=True)

Parallel load
^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    model = onnxl.load(
        "model.onnx",
        num_threads=4,
    )

Parallel save
^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    onnxl.save(
        model,
        "out.onnx",
        location="out.onnx.data",
        num_threads=4,
    )

Load/save with aligned external data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    save_options = onnxl.SerializeOptions()
    save_options.alignment = 4096
    model.SerializeToFile("out.onnx", save_options, "out.onnx.data")

    loaded = onnxl.ModelProto()
    load_options = onnxl.ParseOptions()
    load_options.alignment = 4096
    loaded.ParseFromFile(
        "out.onnx",
        load_options,
        external_data_file="out.onnx.data",
    )

Save and load the text format (textproto)
-----------------------------------------

The Python helpers also support the protobuf *text format* (``.textproto``),
a human-readable, pure-Python (de)serialization that does not require the
``protobuf`` runtime.  The format is selected automatically from the
``.textproto`` extension, or explicitly with ``format="textproto"``.


.. code-block:: python

    import onnx_light.onnx as onnxl

    # Inferred from the .textproto extension.
    onnxl.save(model, "model.textproto")
    loaded = onnxl.load("model.textproto")

    # Explicit format, any extension.
    onnxl.save(model, "model.onnx", format="textproto")
    loaded = onnxl.load("model.onnx", format="textproto")

The textproto format keeps the full model structure inline; saving with
external tensor data is not supported for this format.


Notes
-----

* Python switches from one-file to two-file save when ``location=...`` is
  provided.  C++ uses :cpp:class:`onnx_light::utils::FileWriteStream` for one file
  and :cpp:class:`onnx_light::utils::TwoFilesWriteStream` for two files.
* Python loads external tensor data with ``load_external_data=True``.  When
  the weights file lives next to the ``.onnx`` file and the stored location is
  still valid, the explicit ``location=...`` override can be omitted.
* Split external-data saves use ``max_external_file_size`` to cap each weights
  file.  The first file keeps the requested base name (for example
  ``out.onnx.data``), then additional files append numbered suffixes such as
  ``out.onnx.data.1`` and ``out.onnx.data.2``.  During load, both Python and
  C++ follow the per-tensor ``external_data.location`` entries stored in the
  model.
* Alignment lives on :py:class:`onnx_light.onnx.SerializeOptions`,
  :py:class:`onnx_light.onnx.ParseOptions`, :cpp:class:`onnx_light::SerializeOptions`,
  and :cpp:class:`onnx_light::ParseOptions` rather than the high-level
  :func:`~onnx_light.onnx.save` / :func:`~onnx_light.onnx.load` helper keywords.  Use a power of two such as
  ``4096`` when you want page-aligned external-data offsets.
* The same parallel options apply to one-file and two-file I/O.  In C++, set
  ``num_threads`` on :cpp:class:`onnx_light::ParseOptions` or
  :cpp:class:`onnx_light::SerializeOptions` before calling the helper functions.

See also
--------

* :ref:`l-example-plot-load-save-external` - Python walkthrough for external
  data round-trips.
* :ref:`l-cpp-load-onnx-light-time-example` - standalone C++ load example.
