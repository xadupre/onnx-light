.. _l-cpp-load-onnx-time-example:

Standalone C++ example: measure ONNX loading time
=================================================

This page documents ``examples/load_onnx_time``, a self-contained CMake
project that benchmarks ONNX model loading using the standard ``onnx``
C++ library (protobuf-based).  It is intended as a reference comparison
against the :doc:`load_onnx_light_time_example`.

Step 1 – Install the standard onnx C++ library
-----------------------------------------------

On Ubuntu / Debian:

.. code-block:: bash

    sudo apt-get install -y libonnx-dev libprotobuf-dev

On other platforms (e.g. via `vcpkg <https://vcpkg.io/>`_ on Windows):

.. code-block:: bat

    vcpkg install onnx

Step 2 – Build the example
---------------------------

.. code-block:: bash

    cmake -S examples/load_onnx_time -B build-load-onnx-time \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build-load-onnx-time

Or use the helper script:

.. code-block:: bash

    bash examples/load_onnx_time/build.sh

.. code-block:: bat

    examples\load_onnx_time\build.bat

The helper scripts build the executable under ``build/load-onnx-time-example``
(``build\load-onnx-time-example\Release`` on Windows with a multi-config
generator).

Step 3 – Run the example
-------------------------

The optional second argument controls the number of timed iterations.
When omitted, the example runs five load passes.

.. code-block:: bash

    ./build-load-onnx-time/load_onnx_time path/to/model.onnx 10

When using the helper script defaults:

.. code-block:: bash

    ./build/load-onnx-time-example/load_onnx_time path/to/model.onnx 10

Example output:

.. code-block:: text

    Loaded: path/to/model.onnx
      File size (MB)   : 12.345
      Iterations       : 10
      Num threads      : 1
      Total load (ms)  : 123.456
      Average load (ms): 12.346
      Min load (ms)    : 11.876
      Max load (ms)    : 13.420
      IR version       : 9
      Producer name    : my_framework
      Graph name       : my_graph
      Nodes            : 42
      Inputs           : 2
      Outputs          : 1
      Initializers     : 10

CMakeLists.txt
--------------

The example CMake project finds the standard onnx library via
``find_package(ONNX)`` and links against the ``onnx`` and ``onnx_proto``
targets:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(load_onnx_time LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(Protobuf REQUIRED)
    find_package(ONNX REQUIRED)

    add_executable(load_onnx_time main.cpp)
    target_compile_definitions(load_onnx_time PRIVATE ONNX_ML=1)
    target_link_libraries(load_onnx_time PRIVATE onnx onnx_proto)

main.cpp
--------

The program opens the ONNX file with ``std::ifstream``, parses it with
``onnx::ModelProto::ParseFromIstream``, measures each iteration with
``std::chrono::steady_clock``, and prints aggregate timing statistics
together with a short model summary.

Key API types
-------------

:cpp:class:`onnx::ModelProto`
    Top-level ONNX model container from the standard protobuf-based onnx
    library.  Loaded via ``ParseFromIstream``.

:cpp:class:`onnx::GraphProto`
    Graph container accessed via ``onnx::ModelProto::graph()``.

See also
--------

* :doc:`load_onnx_light_time_example` – standalone example that loads a model
  with the onnx_light C++ API (no protobuf dependency) and reports timing.
