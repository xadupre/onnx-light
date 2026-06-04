.. _l-cpp-aidge-onnx-light-example:

Standalone C++ example: combine onnx-light with Eclipse Aidge
=============================================================

This page documents ``examples/aidge_onnx_light``, a self-contained CMake
project that demonstrates how to use *onnx-light* together with the
`Eclipse Aidge <https://projects.eclipse.org/projects/technology.aidge>`_
deep-learning framework from a single C++ program.

Eclipse Aidge's built-in ONNX importer is layered on top of the official
ONNX protobuf bindings, which means it inherits the 2 GB message-size limit
and a full-copy memory pattern. *onnx-light* bypasses protobuf entirely, so a
common deployment workflow is:

#. open the (potentially huge, encrypted, or split) ONNX file with
   :cpp:class:`onnx::utils::MmapFileStream` and parse it with
   :cpp:func:`onnx::ParseModelProtoFromStream`;
#. inspect, patch, or decrypt the in-memory :cpp:class:`onnx::ModelProto`;
#. re-serialize a clean, protobuf-compatible model to a temporary file
   with :cpp:func:`onnx::SerializeModelProtoToStream` and
   :cpp:class:`onnx::utils::FileWriteStream`;
#. hand that file to Aidge's ``Aidge::loadONNX`` to instantiate the
   computation graph.

Step 1 -- Install onnx-light
----------------------------

From the *onnx-light* repository root, build and install the static library
and its public headers.  The Python extension is not required:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build  build-install
    cmake --install build-install

Step 2 -- (Optional) install Eclipse Aidge
------------------------------------------

Install at least ``aidge_core`` and ``aidge_onnx`` following the upstream
instructions, for example into ``/opt/aidge``.  See
`Aidge -- Get started <https://eclipse.dev/aidge/source/GetStarted/install.html>`_
for the recommended procedure.

The example also builds without Aidge: in that case the program performs
the onnx-light load and re-serialize steps but skips the Aidge import.

Step 3 -- Build the example
---------------------------

Point ``CMAKE_PREFIX_PATH`` at the install prefix(es) chosen above
(the Aidge prefix can be appended with ``;`` on all platforms):

.. code-block:: bash

    cmake -S examples/aidge_onnx_light -B build-aidge-onnx-light \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="/usr/local;/opt/aidge"
    cmake --build build-aidge-onnx-light

Pass ``-DAIDGE_ONNX_LIGHT_REQUIRE_AIDGE=ON`` to make Aidge mandatory and
fail the configure step if it is missing.

The companion ``build.sh`` / ``build.bat`` scripts automate steps 1 and 3
(install ``onnx_light`` locally, then build the example). They look up Aidge
through the optional ``AIDGE_PREFIX`` environment variable.

Step 4 -- Run the example
-------------------------

.. code-block:: bash

    ./build-aidge-onnx-light/aidge_onnx_light path/to/model.onnx

Example output without Aidge:

.. code-block:: text

    Loaded with onnx-light: path/to/model.onnx
      Load time (ms)   : 1.674
      IR version       : 7
      Producer name    : backend-test
      Graph name       : test_softmax_example_expanded
      Nodes            : 6
      Inputs           : 1
      Outputs          : 1
      Initializers     : 0
    Re-serialized with onnx-light: path/to/model.onnx.onnxlight.12345.tmp
      Save time (ms)   : 0.079
    Aidge integration disabled at build time (rebuild with the Eclipse Aidge
    CMake packages on CMAKE_PREFIX_PATH to enable it).

When Aidge is enabled, the program additionally prints the number of nodes,
inputs and outputs of the resulting ``Aidge::GraphView``.

CMakeLists.txt
--------------

The example CMake project uses ``find_package`` to locate the installed
libraries.  ``onnx_light`` is required; ``aidge_core`` and ``aidge_onnx`` are
optional unless ``-DAIDGE_ONNX_LIGHT_REQUIRE_AIDGE=ON`` is passed:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(aidge_onnx_light LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    option(AIDGE_ONNX_LIGHT_REQUIRE_AIDGE
           "Fail the configure step if the Eclipse Aidge packages cannot be found"
           OFF)

    find_package(onnx_light REQUIRED)
    if(AIDGE_ONNX_LIGHT_REQUIRE_AIDGE)
      find_package(aidge_core CONFIG REQUIRED)
      find_package(aidge_onnx CONFIG REQUIRED)
    else()
      find_package(aidge_core CONFIG QUIET)
      find_package(aidge_onnx CONFIG QUIET)
    endif()

    add_executable(aidge_onnx_light main.cc)
    target_link_libraries(aidge_onnx_light PRIVATE onnx_light::lib_onnx_proto)
    if(aidge_core_FOUND AND aidge_onnx_FOUND)
      target_compile_definitions(aidge_onnx_light PRIVATE AIDGE_ONNX_LIGHT_HAS_AIDGE=1)
      target_link_libraries(aidge_onnx_light PRIVATE _aidge_core _aidge_onnx)
    endif()

main.cc
-------

The program loads the model with :cpp:class:`onnx::utils::MmapFileStream`
and :cpp:func:`onnx::ParseModelProtoFromStream`, re-serializes it through
:cpp:class:`onnx::utils::FileWriteStream` and
:cpp:func:`onnx::SerializeModelProtoToStream`, and -- when compiled with
Aidge -- calls ``Aidge::loadONNX`` on the re-serialized file:

.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    #ifdef AIDGE_ONNX_LIGHT_HAS_AIDGE
    #include <aidge/graph/GraphView.hpp>
    #include <aidge/onnx/ONNX.hpp>
    #endif

    int main(int argc, char *argv[]) {
      const std::string input_path = argv[1];

      onnx::ModelProto model;
      onnx::utils::MmapFileStream in_stream(input_path);
      onnx::ParseOptions parse_opts;
      onnx::ParseModelProtoFromStream(model, in_stream, parse_opts);

      const std::string output_path = input_path + ".onnxlight.tmp";
      onnx::utils::FileWriteStream out_stream(output_path);
      onnx::SerializeOptions ser_opts;
      onnx::SerializeModelProtoToStream(model, out_stream, ser_opts);

    #ifdef AIDGE_ONNX_LIGHT_HAS_AIDGE
      std::shared_ptr<Aidge::GraphView> graph = Aidge::loadONNX(output_path);
    #endif
      return 0;
    }

Key API types
-------------

:cpp:class:`onnx::utils::MmapFileStream`
    Memory-mapped binary input stream.  Used here to load the ONNX file
    without copying it into a protobuf message, which is what enables the
    multi-gigabyte and zero-copy properties of *onnx-light*.

:cpp:func:`onnx::ParseModelProtoFromStream`
    Parses the binary protobuf stream into an :cpp:class:`onnx::ModelProto`.

:cpp:class:`onnx::utils::FileWriteStream`
    Buffered binary output stream backing the re-serialization step.

:cpp:func:`onnx::SerializeModelProtoToStream`
    Writes a :cpp:class:`onnx::ModelProto` back to a binary protobuf stream,
    producing an artefact byte-compatible with the standard ONNX bindings
    (and therefore consumable by Aidge's ``loadONNX``).

``Aidge::loadONNX`` (Aidge)
    Reads an on-disk ONNX file and returns a shared pointer to an
    ``Aidge::GraphView`` that can subsequently be scheduled on any Aidge
    backend (CPU, CUDA, ...).  See the Aidge documentation for advanced
    usage such as device placement, quantization, or graph transformations.

See also
--------

* :ref:`l-cpp-load-onnx-light-time-example` -- a simpler example that only
  loads an ONNX file with *onnx-light* and reports timing statistics.
* :ref:`l-cpp-build-save-load-onnx-proto-example` -- demonstrates building
  and saving a :cpp:class:`onnx::ModelProto` from scratch.
