.. _l-cpp-check-onnx-light-model-example:

Standalone C++ example: validate an ONNX model with onnx_light checker
=======================================================================

This page documents ``examples/check_onnx_light_model``, a self-contained
CMake project that demonstrates linking with *onnx-light* and running
:cpp:func:`onnx::checker::check_model` from C++.

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

    cmake -S examples/check_onnx_light_model -B build-check-onnx-light-model \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local
    cmake --build build-check-onnx-light-model

Step 3 – Run the example
-------------------------

.. code-block:: bash

    ./build-check-onnx-light-model/check_onnx_light_model path/to/model.onnx 1

The optional ``full_check`` argument accepts ``0`` (default) or ``1``.
When ``full_check=1``, checker runs additional shape-inference validation.

Example output:

.. code-block:: text

    Model is valid: path/to/model.onnx
      full_check: true

CMakeLists.txt
--------------

The example uses ``find_package`` and links against the exported
``onnx_light::onnx_light`` target:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(check_onnx_light_model LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(check_onnx_light_model main.cc)
    target_link_libraries(check_onnx_light_model PRIVATE onnx_light::onnx_light)

main.cc
--------

The program calls the path-based checker API and handles validation failures
using :cpp:class:`onnx::checker::ValidationError`.

.. code-block:: cpp

    #include "onnx_lib/checker.h"

    #include <iostream>

    int main(int argc, char *argv[]) {
      if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [full_check]\n";
        return 1;
      }

      try {
        ONNX_LIGHT_NAMESPACE::checker::check_model(argv[1], false);
        std::cout << "Model is valid: " << argv[1] << "\n";
      } catch (const ONNX_LIGHT_NAMESPACE::checker::ValidationError &e) {
        std::cerr << "Validation error:\n" << e.what() << "\n";
        return 2;
      }
      return 0;
    }

See also
--------

* :ref:`l-cpp-load-onnx-light-time-example` – standalone example that loads a
  model and reports timing statistics.
* :doc:`../api/cpp/onnx/checker` – checker API reference.
