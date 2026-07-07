.. _l-cpp-check-onnx-light-model-example:

Standalone C++ example: validate an ONNX model with onnx_light checker
=======================================================================

This page documents ``examples/check_onnx_light_model``
(`view on GitHub <https://github.com/xadupre/onnx-light/tree/main/examples/check_onnx_light_model>`_),
a self-contained
CMake project that demonstrates linking with *onnx-light* and running
:cpp:func:`onnx::checker::check_model` from C++. The same program also
demonstrates calling :cpp:func:`onnx_optim::shapes::InferShapesModel` —
the onnx_optim shape-inference entry point — on the loaded model.

Step 1 – Install the C++ library
---------------------------------

From the *onnx-light* repository root, build and install the static library
and its public headers. The Python extension is not required:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DONNX_LIGHT_BUILD_KERNELS=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install
    cmake --install build-install

``-DONNX_LIGHT_BUILD_KERNELS=OFF`` skips building ``lib_onnx_kernels`` and
``lib_onnx_backend_test`` (the operator-kernel runtime and the backend-test
case registry). They are not needed by this example, which only links the
checker / shape-inference layer exposed by ``onnx_light::lib_onnx_lib``.

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

    ./build-check-onnx-light-model/check_onnx_light_model path/to/model.onnx 1 1

The optional ``full_check`` argument accepts ``0`` (default) or ``1``.
When ``full_check=1``, checker runs additional shape-inference validation.

The optional :func:`~onnx_light.onnx_lib.shape_inference.infer_shapes` argument accepts ``0`` (default) or ``1``.
When ``infer_shapes=1``, the example loads the model into a :class:`~onnx_light.onnx_lib.ModelProto`
and calls :cpp:func:`onnx_optim::shapes::InferShapesModel` to populate
``graph.value_info`` and refine ``graph.output`` shapes in place, then
reports how many entries each list contains.

Example output:

.. code-block:: text

    Model is valid: path/to/model.onnx
      full_check: true
      shape inference: ok
        graph.value_info entries: 12
        graph.output entries:     1

CMakeLists.txt
--------------

The example uses ``find_package`` and links against the exported
``onnx_light::lib_onnx_lib`` target. ``onnx_light::lib_onnx_optim`` is also
linked so the program can call onnx_optim shape inference:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(check_onnx_light_model LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(onnx_light REQUIRED)

    add_executable(check_onnx_light_model main.cc)
    target_link_libraries(check_onnx_light_model
      PRIVATE onnx_light::lib_onnx_lib onnx_light::lib_onnx_optim)

main.cc
--------

The program calls the path-based checker API and handles validation failures
using :cpp:class:`onnx::checker::ValidationError`. When ``infer_shapes=1`` it
also loads the model with :cpp:func:`LoadProtoFromPath` and runs
:cpp:func:`onnx_optim::shapes::InferShapesModel` on the resulting
``ModelProto``.

.. literalinclude:: ../../examples/check_onnx_light_model/main.cc
    :language: cpp

See also
--------

* :ref:`l-cpp-load-onnx-light-time-example` – standalone example that loads a
  model and reports timing statistics.
* :doc:`../api/cpp/onnx/checker` – checker API reference.
