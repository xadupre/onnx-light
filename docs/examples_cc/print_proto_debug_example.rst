.. _l-cpp-print-proto-debug-example:

Standalone C++ example: print a proto for debugging
===================================================

This page documents ``examples/print_proto_debug``
(`view on GitHub <https://github.com/xadupre/onnx-light/tree/main/examples/print_proto_debug>`_),
a self-contained CMake project that demonstrates how to dump any ONNX proto as
readable text from C++.

The example purposely links against ``onnx_light::lib_onnx_proto`` **only**:
no operator schemas, no shape inference, no checker, and no backend kernels
are needed to print a generated protobuf message for debugging.

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

    cmake -S examples/print_proto_debug -B build-print-proto-debug \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-print-proto-debug

Step 3 – Run the example
-------------------------

.. code-block:: bash

    ./build-print-proto-debug/print_proto_debug

.. code-block:: cpp

    #include "onnx.h"
    #include "simple_string.h"

    #include <iostream>
    #include <vector>

    int main() {
      onnx::NodeProto node;
      node.set_name("relu1");
      node.set_op_type("Relu");
      *node.add_input() = "X";
      *node.add_output() = "Y";
      node.set_doc_string("Simple ReLU activation");

      onnx::utils::PrintOptions options;
      std::vector<std::string> lines = node.PrintToVectorString(options);
      // join_string is declared in simple_string.h.
      std::cout << onnx::utils::join_string(lines, "\n") << "\n";
      return 0;
    }

CMakeLists.txt
--------------

The example CMake project uses ``find_package`` to locate the installed
library and links against the exported ``onnx_light::lib_onnx_proto`` target.
That is the minimum dependency set required here because the example only
formats generated protobuf-compatible messages:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(print_proto_debug LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(print_proto_debug main.cc)
    target_link_libraries(print_proto_debug PRIVATE onnx_light::lib_onnx_proto)

Example output:

.. code-block:: text

    input: "X"
    output: "Y"
    name: "relu1"
    op_type: "Relu"
    doc_string: "Simple ReLU activation"

See also
--------

* ``onnx::ProtoDebugString`` (from ``proto_utils.h``) is a convenience helper
  that internally calls ``PrintToVectorString`` and returns a single
  ``std::string``.
