.. _l-cpp-export-nnef-example:

Standalone C++ example: export an ONNX model to NNEF
====================================================

This page documents ``examples/export_nnef``, a self-contained CMake project
that demonstrates linking with *onnx-light* and exporting an ONNX
:cpp:class:`ModelProto` to the
`Khronos NNEF v1.0 <https://www.khronos.org/nnef>`_ representation from C++
using :cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::ToNNEFText` and
:cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::SaveNNEF`.

Step 1 – Install the C++ library
---------------------------------

From the *onnx-light* repository root, build and install the static library
and its public headers. The Python extension is not required:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install
    cmake --install build-install

Step 2 – Build the example
---------------------------

Point ``CMAKE_PREFIX_PATH`` at the install prefix chosen above:

.. code-block:: bash

    cmake -S examples/export_nnef -B build-export-nnef \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-export-nnef

Step 3 – Run the example
-------------------------

.. code-block:: bash

    ./build-export-nnef/export_nnef path/to/model.onnx path/to/out_nnef

The output directory is created if needed and is populated with a
``graph.nnef`` text file plus one ``<label>.dat`` file per initializer,
encoded with the standard NNEF binary tensor format (128-byte header
followed by raw data).  An optional third argument overrides the
``graph_name`` written into ``graph.nnef``.

CMakeLists.txt
--------------

The example uses ``find_package`` and links against the exported
``onnx_light::onnx_nnef`` target, which transitively pulls
``lib_onnx_proto``:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(export_nnef LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(export_nnef main.cc)
    target_link_libraries(export_nnef PRIVATE onnx_light::onnx_nnef)

main.cc
--------

The program loads an ONNX model with ``FileStream`` /
``ParseModelProtoFromStream``, prints the ``graph.nnef`` text and writes
the NNEF directory.  ``NNEFExportError`` is raised when an ONNX construct
cannot be expressed in NNEF (for instance an op with no registered
converter, or a ``Reshape`` with a non-constant ``shape`` input).

.. code-block:: cpp

    #include "nnef/exporter.h"
    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    #include <filesystem>
    #include <iostream>
    #include <string>

    int main(int argc, char *argv[]) {
      if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <out_dir> [graph_name]\n";
        return 1;
      }

      const std::string model_path = argv[1];
      const std::string out_dir = argv[2];
      const std::string graph_name = (argc == 4) ? argv[3] : "";

      ONNX_LIGHT_NAMESPACE::ModelProto model;
      ONNX_LIGHT_NAMESPACE::utils::FileStream stream(model_path);
      ONNX_LIGHT_NAMESPACE::ParseOptions parse_opts;
      ONNX_LIGHT_NAMESPACE::ParseModelProtoFromStream(model, stream, parse_opts);

      try {
        std::cout << ONNX_LIGHT_NAMESPACE::nnef::ToNNEFText(model, graph_name);
        const std::string absolute_out_dir =
            ONNX_LIGHT_NAMESPACE::nnef::SaveNNEF(model, out_dir, graph_name, /*overwrite=*/true);
        std::cout << "Wrote NNEF directory: " << absolute_out_dir << "\n";
      } catch (const ONNX_LIGHT_NAMESPACE::nnef::NNEFExportError &e) {
        std::cerr << "NNEF export error: " << e.what() << "\n";
        return 2;
      }
      return 0;
    }

Registering a custom op converter from C++
-------------------------------------------

Operators without a builtin converter raise
:cpp:class:`ONNX_LIGHT_NAMESPACE::nnef::NNEFExportError`.  New converters
can be plugged in by calling
:cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::RegisterOpConverter` before
``ToNNEFText`` / ``SaveNNEF``:

.. code-block:: cpp

    #include "nnef/exporter.h"

    using namespace ONNX_LIGHT_NAMESPACE::nnef;

    RegisterOpConverter("MyOp",
        [](ExportContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &,
           const std::map<std::string, AttributeValue> &,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs) {
          ctx.AddStatement(outputs[0] + " = my_op(" + inputs[0] + ");");
        });

See also
--------

* :ref:`l-howto-export-nnef` – higher-level overview of the NNEF exporter.
* :doc:`../api/cpp/onnx/checker` – checker API reference (useful to
  validate a model before exporting it).
