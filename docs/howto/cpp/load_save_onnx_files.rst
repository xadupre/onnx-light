.. _l-howto-load-save-onnx-files-cpp:

:html_theme.sidebar_secondary.remove:

How to load and save ONNX files
===============================

This guide covers the C++ interface. For the other language, see
:doc:`../python/load_save_onnx_files`.

This page covers the *onnx-light* C++ load/save API for one-file
``.onnx`` models, two-file models with external tensor data, split external
data across multiple files, alignment options, and the thread-count options for
larger models.

Common load/save patterns
-------------------------

Load one file
^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::ModelProto model;
    onnx::utils::FileStream stream("model.onnx");
    onnx::ParseOptions options;
    onnx::ParseModelProtoFromStream(model, stream, options);

Load two files
^^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::ModelProto model;
    onnx::utils::TwoFilesStream stream("model.onnx", "model.onnx.data");
    onnx::ParseOptions options;
    onnx::ParseModelProtoFromStream(model, stream, options);

Save one file
^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::SerializeOptions options;
    onnx::utils::FileWriteStream stream("out.onnx");
    onnx::SerializeModelProtoToStream(model, stream, options);

Save two files
^^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::SerializeOptions options;
    onnx::utils::TwoFilesWriteStream stream("out.onnx", "out.onnx.data");
    onnx::SerializeModelProtoToStream(model, stream, options);

Load/save split external files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::SerializeOptions options;
    options.max_external_file_size = 2LL * 1024 * 1024 * 1024;
    onnx::utils::TwoFilesWriteStream out("out.onnx", "out.onnx.data");
    onnx::SerializeModelProtoToStream(model, out, options);

    onnx::ModelProto loaded;
    // Additional files such as out.onnx.data.1 are opened automatically
    // from the external_data.location values stored in the model.
    onnx::utils::TwoFilesStream in("out.onnx", "out.onnx.data");
    onnx::ParseOptions parse_options;
    onnx::ParseModelProtoFromStream(loaded, in, parse_options);

Parallel load
^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::ModelProto model;
    onnx::utils::FileStream stream("model.onnx");
    onnx::ParseOptions options;
    options.num_threads = 4;
    onnx::ParseModelProtoFromStream(model, stream, options);

Parallel save
^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::SerializeOptions options;
    options.num_threads = 4;
    onnx::utils::TwoFilesWriteStream stream("out.onnx", "out.onnx.data");
    onnx::SerializeModelProtoToStream(model, stream, options);

Load/save with aligned external data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::SerializeOptions save_options;
    save_options.alignment = 4096;
    onnx::utils::TwoFilesWriteStream out("out.onnx", "out.onnx.data");
    onnx::SerializeModelProtoToStream(model, out, save_options);

    onnx::ModelProto loaded;
    onnx::utils::TwoFilesStream in("out.onnx", "out.onnx.data");
    onnx::ParseOptions load_options;
    load_options.alignment = 4096;
    onnx::ParseModelProtoFromStream(loaded, in, load_options);


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
