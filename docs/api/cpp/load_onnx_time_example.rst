.. _l-cpp-load-onnx-time-example:

Standalone C++ example: measure ONNX loading time
=================================================

This page documents ``examples/load_onnx_time``, a self-contained CMake
project that demonstrates how to consume *onnx-light* as an installed C++
library, repeatedly load an ONNX file, and report wall-clock timing
statistics.

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

Step 2 – Build the example
---------------------------

Point ``CMAKE_PREFIX_PATH`` at the install prefix chosen above:

.. code-block:: bash

    cmake -S examples/load_onnx_time -B build-load-onnx-time \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-load-onnx-time

Step 3 – Run the example
-------------------------

The optional second argument controls the number of timed iterations.
When omitted, the example runs five load passes.

.. code-block:: bash

    ./build-load-onnx-time/load_onnx_time path/to/model.onnx 10

Example output:

.. code-block:: text

    Loaded: path/to/model.onnx
      File size (MB)   : 12.345
      Iterations       : 10
      Total load (ms)  : 123.456
      Average load (ms): 12.346
      Min load (ms)    : 11.876
      Max load (ms)    : 13.420
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
    project(load_onnx_time LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(load_onnx_time main.cpp)
    target_link_libraries(load_onnx_time PRIVATE onnx_light::onnx_light)

main.cpp
--------

The program opens the ONNX file with :cpp:class:`onnx::utils::FileStream`,
parses it with :cpp:func:`onnx::ParseModelProtoFromStream`, measures each
iteration with ``std::chrono::steady_clock``, and prints aggregate timing
statistics together with a short model summary.

Key API types
-------------

:cpp:class:`onnx::utils::FileStream`
    Binary input stream backed by a file.  Constructed with the path to the
    ``.onnx`` file; throws ``std::runtime_error`` if the file cannot be
    opened.

:cpp:class:`onnx::ParseOptions`
    Controls parsing behaviour for each timed load pass.

:cpp:func:`onnx::ParseModelProtoFromStream`
    Parses the binary protobuf stream into a :cpp:class:`onnx::ModelProto`.

:cpp:class:`onnx::ModelProto`
    Top-level ONNX model container.  The example reuses the parsed model from
    the last iteration to print graph metadata next to the timing results.

See also
--------

* :doc:`load_onnx_example` – standalone example that loads a model and prints
  metadata without timing it.
* :doc:`stream` – full reference for ``FileStream``, ``StringStream``,
  and write streams.
* :doc:`onnx_helper` – ``ParseModelProtoFromStream`` and related helpers.
