.. _l-cpp-load-onnx-time-example:

Standalone C++ example: measure ONNX loading time
=================================================

This page documents ``examples/load_onnx_time``, a self-contained CMake
project that benchmarks ONNX model loading using the standard ``onnx``
C++ library (protobuf-based).  It is intended as a reference comparison
against the :ref:`l-cpp-load-onnx-light-time-example`.

Step 1 – Obtain the standard ONNX C++ library
---------------------------------------------

**Option A – system package** (Ubuntu / Debian):

.. code-block:: bash

    sudo apt-get install -y libonnx-dev libprotobuf-dev

On Windows via `vcpkg <https://vcpkg.io/>`_:

.. code-block:: bat

    vcpkg install onnx

**Option B – build from source** (explicit):
Set ``ONNX_GIT_TAG`` to the desired git tag or branch.  The script pre-clones
onnx, then cmake builds onnx **and all its transitive dependencies** (protobuf,
abseil, utf8_range, …) inline via ``FetchContent`` — no separate protobuf
install step is needed:

.. code-block:: bash

    ONNX_GIT_TAG=v1.17.0 bash examples/load_onnx_time/build.sh

On Windows:

.. code-block:: bat

    set ONNX_GIT_TAG=v1.17.0
    examples\load_onnx_time\build.bat

**Option C – automatic** (no variables needed):
The helper script probes for the onnx CMake package at startup.  If not found,
it automatically switches to a from-source build.  The onnx git tag is derived
from the Python ``onnx`` package in site-packages (``import onnx``) when
available, falling back to ``ONNX_DEFAULT_GIT_TAG`` (default ``v1.17.0``).
All transitive dependencies are handled by cmake automatically.

.. code-block:: bash

    # Just run; the script figures out what to build
    bash examples/load_onnx_time/build.sh

The onnx source is pre-cloned to ``build/load-onnx-time-lib/onnx-src``.
Subsequent invocations reuse the existing clone and skip the git step.

Step 2 – Build the example
---------------------------

When using the system onnx library directly with CMake:

.. code-block:: bash

    cmake -S examples/load_onnx_time -B build-load-onnx-time \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build-load-onnx-time

To build onnx from source (cmake fetches it automatically), pass
``ONNX_GIT_TAG`` and optionally point ``FETCHCONTENT_SOURCE_DIR_ONNX`` at a
pre-cloned source tree to skip the download:

.. code-block:: bash

    cmake -S examples/load_onnx_time -B build-load-onnx-time \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_GIT_TAG=v1.17.0 \
          -DFETCHCONTENT_SOURCE_DIR_ONNX=/path/to/onnx-src
    cmake --build build-load-onnx-time

Or use the helper scripts (fully automatic):

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

The example CMake project tries ``find_package(ONNX)`` first (system package).
When not found, it falls back to ``FetchContent`` to build onnx and all its
transitive dependencies (protobuf, abseil, utf8_range, …) inline:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(load_onnx_time LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(protobuf CONFIG QUIET)
    if(NOT TARGET protobuf::libprotobuf)
        find_package(Protobuf QUIET)
    endif()
    find_package(ONNX QUIET)

    if(NOT TARGET onnx)
        if(NOT DEFINED ONNX_GIT_TAG)
            set(ONNX_GIT_TAG "v1.17.0")
        endif()
        if(NOT DEFINED ONNX_GIT_URL)
            set(ONNX_GIT_URL "https://github.com/onnx/onnx.git")
        endif()
        set(ONNX_ML ON CACHE BOOL "" FORCE)
        set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        include(FetchContent)
        FetchContent_Declare(onnx
            GIT_REPOSITORY "${ONNX_GIT_URL}"
            GIT_TAG        "${ONNX_GIT_TAG}"
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(onnx)
    endif()

    add_executable(load_onnx_time main.cc)
    target_compile_definitions(load_onnx_time PRIVATE ONNX_ML=1)
    target_link_libraries(load_onnx_time PRIVATE onnx onnx_proto)

main.cc
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

* :ref:`l-cpp-load-onnx-light-time-example` – standalone example that loads a model
  with the onnx_light C++ API (no protobuf dependency) and reports timing.
