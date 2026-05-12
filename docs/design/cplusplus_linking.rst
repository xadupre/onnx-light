.. _l-design-cpp-linking:

Linking *onnx-light* in C++
===========================

This page summarizes the design used to consume *onnx-light* as a standalone
C++ library from another project.

The full runnable example is available in
:epkg:`C++ onnx-light examples` (folder ``examples/load_onnx_light_time``).

Install and link model
----------------------

After installing *onnx-light* with CMake, downstream projects can rely on the
exported CMake target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::onnx_light)

This keeps downstream CMake files independent from hardcoded include paths and
library file names.

Excerpt from the example project
--------------------------------

The example CMake project in ``examples/load_onnx_light_time`` uses exactly
that pattern:

.. literalinclude:: ../../examples/load_onnx_light_time/CMakeLists.txt
    :language: cmake
    :lines: 28-37

See also
--------

* :doc:`../api/cpp/load_onnx_light_time_example`
