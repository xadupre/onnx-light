.. _l-design-cpp-linking:

Linking *onnx-light* in C++
===========================

The installed CMake package exports small targets under the ``onnx_light::``
namespace. Public dependencies are propagated, so a consumer links only the
highest-level feature it needs. See :ref:`l-design-library-split` for the
dependency graph and :epkg:`C++ onnx-light examples` for complete projects.

Install and consume the package
-------------------------------

Build a static, pure C++ installation:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install --parallel
    cmake --install build-install

A downstream project can then select one or more exported targets:

.. code-block:: cmake

    find_package(onnx_light CONFIG REQUIRED)

    add_executable(my_reader main.cc)
    target_link_libraries(my_reader PRIVATE onnx_light::lib_onnx_proto)

Common choices are:

* ``onnx_light::lib_onnx_proto`` for model parsing and serialization;
* ``onnx_light::lib_onnx_lib`` for schemas, checking, inlining, standard shape
  inference, and version conversion;
* ``onnx_light::lib_onnx_shape`` for the standalone shape and peak-memory
  dispatch;
* ``onnx_light::lib_onnx_patterns`` for standard graph rewrites;
* ``onnx_light::lib_onnx_kernels`` for the reference runtime;
* ``onnx_light::lib_onnx_backend_test`` for backend-test cases;
* ``onnx_light::lib_onnx_gradient`` for gradient generation.

``lib_onnx_backend_test`` propagates ``lib_onnx_kernels``. The shape, pattern,
kernel, and gradient targets propagate ``lib_onnx_core`` and
``lib_onnx_proto``. Consumers do not need to repeat these dependencies.

For a non-standard installation prefix, set
``-DCMAKE_PREFIX_PATH=<installation-prefix>`` when configuring the consumer.

.. _l-design-cpp-linking-cmake-options:

Configure-time options
----------------------

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Option
     - Default
     - Effect
   * - ``ONNX_LIGHT_BUILD_PYTHON``
     - ``ON``
     - Builds the nanobind modules and shared C++ libraries. ``OFF`` builds
       static libraries for pure C++ consumers.
   * - ``ONNX_LIGHT_BUILD_KERNELS``
     - ``ON``
     - Builds kernels, backend tests, gradients, and their Python modules.
       ``OFF`` produces the reduced proto/schema/shape/pattern package.
   * - ``ONNX_LIGHT_INSTALL``
     - ``ON``
     - Installs headers, libraries, and CMake package files.
   * - ``ONNX_LIGHT_PROVIDE_ONNX_TARGETS``
     - ``OFF``
     - Adds in-tree compatibility targets named ``onnx``, ``onnx_proto``,
       ``onnx::onnx``, and ``onnx::onnx_proto``.
   * - ``ONNX_LIGHT_BUILD_TESTS``
     - ``OFF``
     - Builds C++ tests. Reduced tests remain available when kernels are off.
   * - ``ONNX_LIGHT_BUILD_BENCHMARKS``
     - ``OFF``
     - Builds C++ benchmarks.
   * - ``ONNX_LIGHT_BUILD_FUZZERS``
     - ``OFF``
     - Builds Clang/libFuzzer harnesses.
   * - ``ONNX_ML``
     - ``ON``
     - Enables the ``ai.onnx.ml`` operator domain.
   * - ``ONNX_LIGHT_WERROR``
     - ``ON``
     - Treats warnings from onnx-light targets as errors.
   * - ``ONNX_HARDENING``
     - ``OFF``
     - Enables supported OpenSSF compiler and linker hardening flags.

.. _l-design-cpp-linking-no-kernels:

Reduced build without runtime kernels
-------------------------------------

``ONNX_LIGHT_BUILD_KERNELS=OFF`` omits ``lib_onnx_kernels``,
``lib_onnx_backend_test``, ``lib_onnx_gradient``, and their Python modules.
The remaining Python modules and reduced C++ tests are supported:

.. code-block:: bash

    cmake -S . -B build-reduced \
          -DONNX_LIGHT_BUILD_KERNELS=OFF \
          -DONNX_LIGHT_BUILD_PYTHON=OFF
    cmake --build build-reduced --parallel

The exported package still contains ``lib_onnx_proto``, ``lib_onnx_core``,
``lib_onnx_manipulations``, ``lib_onnx_lib``, ``lib_onnx_op``,
``lib_onnx_shape``, and ``lib_onnx_patterns``.

Drop-in upstream ONNX targets
-----------------------------

An installed package also provides an ``onnx`` compatibility package:

.. code-block:: cmake

    find_package(onnx CONFIG REQUIRED)
    target_link_libraries(my_target PRIVATE onnx::onnx)

``onnx::onnx`` aggregates the proto, schema, manipulation, and shape targets;
``onnx::onnx_proto`` provides only proto-compatible messages. For
``add_subdirectory`` or ``FetchContent`` consumers, set
``ONNX_LIGHT_PROVIDE_ONNX_TARGETS=ON`` to create the equivalent compatibility
targets in the build tree.

.. _l-design-cpp-linking-patterns:

Registering graph patterns
--------------------------

The optimizer interface is in ``lib_onnx_core``; standard ONNX rewrites are in
``lib_onnx_patterns`` and are registered explicitly:

.. code-block:: cmake

    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_patterns)

.. code-block:: cpp

    #include "onnx_core/builder/graph_graph.h"
    #include "onnx_extensions/patterns/dispatch_table.h"

    onnx_light::onnx_patterns::RegisterPatterns();
    onnx_light::core::builder::GraphGraph optimizer(builder);
    optimizer.Optimize();

Applications using only custom patterns may link
``onnx_light::lib_onnx_core`` and pass their pattern instances directly to
``GraphGraph``.

Registering runtime kernels
---------------------------

Link ``onnx_light::lib_onnx_kernels`` and register the built-in dispatch table
before creating a runtime session:

.. code-block:: cpp

    #include "onnx_extensions/kernels/kernel_dispatch_table.h"

    onnx_light::onnx_kernels::RegisterKernelFunctions();

See :ref:`l-design-runtime` for session creation, execution pools, prepared
execution, tuning, and profiling.

Using the source tree directly
------------------------------

For monorepos, use the non-namespaced in-tree targets:

.. code-block:: cmake

    set(ONNX_LIGHT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory(path/to/onnx-light)
    target_link_libraries(my_target PRIVATE lib_onnx_lib)

The examples under ``examples/`` use both installed and in-tree forms and are
kept as executable linking tests.
