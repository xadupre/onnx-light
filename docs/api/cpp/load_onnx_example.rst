.. _l-cpp-load-onnx-example:

Standalone C++ example: load an ONNX file
==========================================

This page documents ``examples/load_onnx``, a self-contained CMake project
that demonstrates how to consume *onnx-light* as an installed C++ library,
open an ONNX file, and print a summary of the model.

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

* ``liblib_onnx_cpp.a`` into ``<prefix>/lib``
* All public C++ headers under ``<prefix>/include/onnx_light``
* CMake package config files under ``<prefix>/lib/cmake/onnx_light``

Step 2 – Build the example
---------------------------

Point ``CMAKE_PREFIX_PATH`` at the install prefix chosen above:

.. code-block:: bash

    cmake -S examples/load_onnx -B build-load-onnx \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-load-onnx

Step 3 – Run the example
-------------------------

.. code-block:: bash

    ./build-load-onnx/load_onnx path/to/model.onnx

Example output:

.. code-block:: text

    Loaded: path/to/model.onnx
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
library and links against the exported ``onnx_light::onnx_light`` target:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(load_onnx LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(load_onnx main.cpp)
    target_link_libraries(load_onnx PRIVATE onnx_light::onnx_light)

main.cpp
--------

The program opens the ONNX file with :cpp:class:`onnx::utils::MmapStream`,
parses it with :cpp:func:`onnx::ParseModelProtoFromStream`, and prints the
model metadata.  File-not-found and parse errors are caught and reported to
``stderr``.  ``MmapStream`` memory-maps the file so the OS page cache is
accessed as contiguous memory without per-byte buffering or system calls:

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
        onnx::utils::MmapStream stream(file_path);
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

:cpp:class:`onnx::utils::MmapStream`
    Binary input stream backed by a memory-mapped file.  Constructed with the
    path to the ``.onnx`` file; throws ``std::runtime_error`` if the file
    cannot be opened or mapped.  Inherits from
    :cpp:class:`onnx::utils::StringStream` so all fast in-memory parsing
    primitives are available without buffering or system calls per byte.

:cpp:class:`onnx::utils::FileStream`
    Buffered binary input stream.  Still available for use cases where mmap
    is not applicable (e.g. very large files on memory-constrained systems or
    as the base class for :cpp:class:`onnx::utils::TwoFilesStream`).

:cpp:class:`onnx::ParseOptions`
    Controls parsing behaviour.  Set ``parallel = true`` and
    ``num_threads = N`` to enable parallel tensor loading across *N* threads
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

* :doc:`stream` – full reference for ``MmapStream``, ``FileStream``,
  ``StringStream``, and write streams.
* :doc:`onnx_helper` – ``ParseModelProtoFromStream`` and related helpers.
