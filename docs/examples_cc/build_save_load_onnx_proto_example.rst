.. _l-cpp-build-save-load-onnx-proto-example:

Standalone C++ example: build, save and load an ONNX model with only ``lib_onnx_proto``
=======================================================================================

This page documents ``examples/build_save_load_onnx_proto``, a self-contained
CMake project that builds a tiny ONNX :cpp:class:`onnx::ModelProto` entirely
in C++, serializes it to disk, parses it back, and checks the round-trip.

The example purposely links against ``onnx_light::lib_onnx_proto`` **only**:
no operator schemas, no shape inference, no checker, and no backend kernels
are needed. It is the minimum surface required to produce and consume an
``.onnx`` file from C++.

Step 1 – Install the C++ library
---------------------------------

From the *onnx-light* repository root, build and install the static library
and its public headers. The Python extension is not required:

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

    cmake -S examples/build_save_load_onnx_proto -B build-build-save-load \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-build-save-load

Step 3 – Run the example
-------------------------

.. code-block:: bash

    ./build-build-save-load/build_save_load_onnx_proto /tmp/example_add.onnx

Example output:

.. code-block:: text

    Saved: /tmp/example_add.onnx
      IR version       : 7
      Producer name    : build_save_load_onnx_proto
      Graph name       : add_graph
      Nodes            : 1
      Inputs           : 1
      Outputs          : 1
      Initializers     : 1
    Loaded: /tmp/example_add.onnx
      ...
      Initializer B    : [1, 2, 3]
    Round-trip OK

The generated file is a valid ONNX model and can be opened with the
reference :mod:`onnx` Python package
(``onnx.load(...)`` followed by ``onnx.checker.check_model(...)``).

CMakeLists.txt
--------------

The example CMake project uses ``find_package`` to locate the installed
library and links against the exported ``onnx_light::lib_onnx_proto`` target
only:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(build_save_load_onnx_proto LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(build_save_load_onnx_proto main.cc)
    target_link_libraries(build_save_load_onnx_proto PRIVATE onnx_light::lib_onnx_proto)

Key API types
-------------

:cpp:class:`onnx::ModelProto`, :cpp:class:`onnx::GraphProto`, :cpp:class:`onnx::NodeProto`, :cpp:class:`onnx::TensorProto`, :cpp:class:`onnx::ValueInfoProto`
    Plain proto-message classes used to assemble the model in memory.

:cpp:class:`onnx::utils::FileWriteStream`
    Buffered binary output stream used to write the serialized model.

:cpp:class:`onnx::utils::FileStream`
    Buffered binary input stream used to read the model back from disk.

:cpp:func:`onnx::SerializeModelProtoToStream`
    Writes a :cpp:class:`onnx::ModelProto` to a binary stream using the
    configured :cpp:class:`onnx::SerializeOptions`.

:cpp:func:`onnx::ParseModelProtoFromStream`
    Parses a binary protobuf stream into a :cpp:class:`onnx::ModelProto`,
    optionally honouring :cpp:class:`onnx::ParseOptions` (threading, no-copy,
    etc.).

See also
--------

* :ref:`l-cpp-load-onnx-light-time-example` – companion example that loads a
  pre-existing ``.onnx`` file and reports timing statistics.
* :ref:`l-api-onnx-onnx-proto-stream` – full reference for ``FileStream``,
  ``FileWriteStream``, ``StringStream``, and other stream classes.
* :ref:`l-api-onnx-onnx-proto-onnx-helper` – ``SerializeModelProtoToStream``,
  ``ParseModelProtoFromStream``, and related helpers.
