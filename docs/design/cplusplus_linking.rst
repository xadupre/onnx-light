.. _l-design-cpp-linking:

Linking *onnx-light* in C++
===========================

This page summarizes the design used to consume *onnx-light* as a standalone
C++ library from another project.

Runnable examples are available in :epkg:`C++ onnx-light examples`, including
``examples/load_onnx_light_time`` and ``examples/check_onnx_light_model``.

Install and link model
----------------------

From the repository root, install the C++ library with CMake:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install
    cmake --install build-install

Then downstream projects can rely on the exported CMake targets:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::onnx_light)

Use ``onnx_light::onnx_light`` when the code needs higher-level ONNX features
implemented in ``onnx_light/onnx_lib`` such as operator schemas, checker, shape
inference, or version conversion.

For protobuf-compatible message parsing/serialization only, downstream code can
link just the lighter proto target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_proto)

That is sufficient when the program only manipulates ``ModelProto`` /
``GraphProto`` data and does not need any notion of operators.

For manual registration of lightweight math operator schemas without shape
inference support, downstream code can link:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_op)

This keeps downstream CMake files independent from hardcoded include paths and
library file names. If *onnx-light* is installed to a non-standard prefix,
configure the downstream project with ``-DCMAKE_PREFIX_PATH=<prefix>``.

Alternative without install
---------------------------

For monorepos or local development, a downstream CMake project can also include
*onnx-light* directly:

.. code-block:: cmake

    set(ONNX_LIGHT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory(path/to/onnx-light)
    target_link_libraries(my_target PRIVATE lib_onnx_lib)

Use the in-tree ``lib_onnx_proto`` target instead when only proto
parsing/serialization is needed. This uses the in-tree build targets instead of
``find_package``.

Excerpt from the example project
--------------------------------

The example CMake project in ``examples/load_onnx_light_time`` uses exactly
that pattern:

.. literalinclude:: ../../examples/load_onnx_light_time/CMakeLists.txt
    :language: cmake
    :lines: 28-37

See also
--------

* :ref:`l-cpp-load-onnx-light-time-example`
