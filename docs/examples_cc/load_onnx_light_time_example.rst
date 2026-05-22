.. _l-cpp-load-onnx-light-time-example:

Standalone C++ example: load an ONNX file with onnx_light
=========================================================

This page documents ``examples/load_onnx_light_time``, a self-contained CMake
project that demonstrates how to consume *onnx-light* as an installed C++
library, repeatedly load an ONNX file, and print timing statistics together
with a summary of the model.

Step 1 – Install the C++ library
---------------------------------

From the *onnx-light* repository root, build and install the static library
and its public headers.  The Python extension is not required:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build  build-install
    cmake --install build-install

The install step places:

* ``liblib_onnx_proto.a`` and ``liblib_onnx_lib.a`` into ``<prefix>/lib``
* All public C++ headers under ``<prefix>/include/onnx_light``
* CMake package config files under ``<prefix>/lib/cmake/onnx_light``

Step 2 – Build the example
---------------------------

Point ``CMAKE_PREFIX_PATH`` at the install prefix chosen above:

.. code-block:: bash

    cmake -S examples/load_onnx_light_time -B build-load-onnx-light-time \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-load-onnx-light-time

Step 3 – Run the example
-------------------------

.. code-block:: bash

    ./build-load-onnx-light-time/load_onnx_light_time path/to/model.onnx 10 4

To measure the shared-buffer external-data path directly from C++, pass the
optional ``nocopy`` mode on a model that uses external tensor data:

.. code-block:: bash

    ./build-load-onnx-light-time/load_onnx_light_time path/to/model.onnx 10 1 nocopy

Example output:

.. code-block:: text

    Loaded: path/to/model.onnx
      Average load (ms): 5.321
      Min load (ms)    : 5.002
      Max load (ms)    : 5.889
      IR version       : 9
      Producer name    : my_framework
      Graph name       : my_graph
      Nodes            : 42
      Inputs           : 2
      Outputs          : 1
      Initializers     : 10

CMakeLists.txt
--------------

The example CMake project uses ``find_package`` to locate the installed
library and links against the exported ``onnx_light::lib_onnx_proto`` target.
That is enough here because the example only parses protobuf-compatible model
messages and does not need operator-aware APIs:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(load_onnx_light_time LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(load_onnx_light_time main.cc)
    target_link_libraries(load_onnx_light_time PRIVATE onnx_light::lib_onnx_proto)

main.cc
--------

The program opens the ONNX file with :cpp:class:`onnx::utils::FileStream`,
parses it with :cpp:func:`onnx::ParseModelProtoFromStream`, reports parse-time
statistics from repeated in-process iterations, and prints model metadata.
File-not-found and parse errors are caught and reported to ``stderr``.
``FileStream`` reads the file sequentially using a buffered read-ahead approach:

.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    #include <iostream>
    #include <stdexcept>
    #include <string>

    int main(int argc, char *argv[]) {
      if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        return 1;
      }

      const std::string file_path = argv[1];

      onnx::ModelProto model;
      try {
        onnx::utils::FileStream stream(file_path);
        onnx::ParseOptions opts;
        onnx::ParseModelProtoFromStream(model, stream, opts);
      } catch (const std::exception &e) {
        std::cerr << "Error loading '" << file_path << "': " << e.what() << "\n";
        return 1;
      }

      std::cout << "Loaded: " << file_path << "\n";

      if (model.has_ir_version())
        std::cout << "  IR version   : " << model.ref_ir_version() << "\n";
      if (model.has_producer_name())
        std::cout << "  Producer     : " << model.ref_producer_name().as_string() << "\n";

      if (model.has_graph()) {
        const onnx::GraphProto &graph = model.ref_graph();
        std::cout << "  Graph name   : " << graph.ref_name().as_string() << "\n";
        std::cout << "  Nodes        : " << graph.ref_node().size() << "\n";
        std::cout << "  Inputs       : " << graph.ref_input().size() << "\n";
        std::cout << "  Outputs      : " << graph.ref_output().size() << "\n";
        std::cout << "  Initializers : " << graph.ref_initializer().size() << "\n";
      }

      return 0;
    }

Key API types
-------------

:cpp:class:`onnx::utils::FileStream`
    Buffered binary input stream.  Constructed with the path to the ``.onnx``
    file; throws ``std::runtime_error`` if the file cannot be opened.  Uses a
    read-ahead buffer and supports optional parallel tensor loading via an
    internal thread pool.  Also serves as the base class for
    :cpp:class:`onnx::utils::TwoFilesStream`.

:cpp:class:`onnx::ParseOptions`
    Controls parsing behaviour.  Set ``num_threads = N`` (with ``N > 1``,
    or a negative value to use the number of CPU cores) to enable parallel
    tensor loading across *N* threads
    (useful for large models with many initializers).

:cpp:func:`onnx::ParseModelProtoFromStream`
    Parses the binary protobuf stream into a :cpp:class:`onnx::ModelProto`.
    Handles both single-file models and models with external data (via
    :cpp:class:`onnx::utils::TwoFilesStream`).

:cpp:class:`onnx::ModelProto`
    Top-level ONNX model container.  Access the embedded graph with
    ``model.ref_graph()`` (returns :cpp:class:`onnx::GraphProto`).

See also
--------

* :ref:`l-api-onnx-onnx-proto-stream` – full reference for ``FileStream``,
  ``StringStream``, and write streams.
* :ref:`l-api-onnx-onnx-proto-onnx-helper` – ``ParseModelProtoFromStream`` and related helpers.
