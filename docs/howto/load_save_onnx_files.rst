.. _l-howto-load-save-onnx-files:

How to load and save ONNX files
===============================

This page aligns the most common *onnx-light* load/save recipes side by side
for Python and C++.  Each row shows the equivalent API for one-file
``.onnx`` models, two-file models with external tensor data, split external
data across multiple files, and the parallel options for larger models.

Common load/save patterns
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Scenario
     - Python
     - C++
   * - Load one file
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          model = onnxl.load("model.onnx")
     - .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::ModelProto model;
          onnx::utils::FileStream stream("model.onnx");
          onnx::ParseOptions options;
          onnx::ParseModelProtoFromStream(model, stream, options);
   * - Load two files
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          model = onnxl.load(
              "model.onnx",
              location="model.onnx.data",
              load_external_data=True,
          )
     - .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::ModelProto model;
          onnx::utils::TwoFilesStream stream("model.onnx", "model.onnx.data");
          onnx::ParseOptions options;
          onnx::ParseModelProtoFromStream(model, stream, options);
   * - Save one file
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          onnxl.save(model, "out.onnx")
     - .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::SerializeOptions options;
          onnx::utils::FileWriteStream stream("out.onnx");
          onnx::SerializeModelProtoToStream(model, stream, options);
   * - Save two files
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          onnxl.save(model, "out.onnx", location="out.onnx.data")
     - .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::SerializeOptions options;
          onnx::utils::TwoFilesWriteStream stream("out.onnx", "out.onnx.data");
          onnx::SerializeModelProtoToStream(model, stream, options);
   * - Load/save split external files
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          onnxl.save(
              model,
              "out.onnx",
              location="out.onnx.data",
              max_external_file_size=2 * 1024 ** 3,
          )
          model = onnxl.load("out.onnx", load_external_data=True)
     - .. code-block:: cpp

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
   * - Parallel load
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          model = onnxl.load(
              "model.onnx",
              parallel=True,
              num_threads=4,
          )
     - .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::ModelProto model;
          onnx::utils::FileStream stream("model.onnx");
          onnx::ParseOptions options;
          options.parallel = true;
          options.num_threads = 4;
          onnx::ParseModelProtoFromStream(model, stream, options);
   * - Parallel save
     - .. code-block:: python

          import onnx_light.onnx as onnxl

          onnxl.save(
              model,
              "out.onnx",
              location="out.onnx.data",
              parallel=True,
              num_threads=4,
          )
     - .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::SerializeOptions options;
          options.parallel = true;
          options.num_threads = 4;
          onnx::utils::TwoFilesWriteStream stream("out.onnx", "out.onnx.data");
          onnx::SerializeModelProtoToStream(model, stream, options);

Notes
-----

* Python switches from one-file to two-file save when ``location=...`` is
  provided.  C++ uses :cpp:class:`onnx::utils::FileWriteStream` for one file
  and :cpp:class:`onnx::utils::TwoFilesWriteStream` for two files.
* Python loads external tensor data with ``load_external_data=True``.  When
  the weights file lives next to the ``.onnx`` file and the stored location is
  still valid, the explicit ``location=...`` override can be omitted.
* Split external-data saves use ``max_external_file_size`` to cap each weights
  file.  The first file keeps the requested base name (for example
  ``out.onnx.data``), then additional files append numbered suffixes such as
  ``out.onnx.data.1`` and ``out.onnx.data.2``.  During load, both Python and
  C++ follow the per-tensor ``external_data.location`` entries stored in the
  model.
* The same parallel options apply to one-file and two-file I/O.  In C++, set
  ``parallel`` and ``num_threads`` on :cpp:class:`onnx::ParseOptions` or
  :cpp:class:`onnx::SerializeOptions` before calling the helper functions.

See also
--------

* :ref:`l-example-plot-load-save-external` - Python walkthrough for external
  data round-trips.
* :ref:`l-cpp-load-onnx-light-time-example` - standalone C++ load example.
