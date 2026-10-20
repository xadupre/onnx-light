.. _l-howto-install-onnx-light-cpp:

How to install *onnx-light*
===========================

This guide covers the C++ interface. For the other language, see
:doc:`../python/install_onnx_light`.

This page covers *onnx-light* installation for C++, including local
development, faster builds, and basic post-install validation.

Common install patterns
-----------------------

Install for local development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install
    cmake --install build-install

Speed up the build
^^^^^^^^^^^^^^^^^^


.. code-block:: bash

    cmake --build build-install --parallel 8

Local builds use private precompiled headers (PCHs) to avoid repeatedly parsing
common headers. The shape and pattern libraries precompile only their shared
``shapes_context.h`` and ``graph_graph.h`` interfaces, respectively; these PCHs
are not propagated to consumers. Source files must still include their own
dependencies and build without PCHs.

Configure with ``-DONNX_LIGHT_USE_PCH=OFF`` to disable PCHs. When ``ccache`` or
``sccache`` is configured as a compiler launcher, PCHs are disabled by default;
``-DONNX_LIGHT_USE_PCH=ON`` explicitly overrides that default.

Shape and pattern precompiled headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single clean-build comparison for the shape and pattern PCHs used GCC 13.3,
Unix Makefiles, Release ``-O3 -DNDEBUG``, kernels, Python bindings and tests
enabled, and ``--parallel 6``, without compiler caches or reusable build
outputs. Complete-build timing included PCH construction:

.. list-table:: Single clean-build pair
   :header-rows: 1

   * - Shape and pattern PCHs
     - Elapsed
     - Maximum RSS (KiB)
   * - Disabled
     - 22m 46.43s
     - 2,923,976
   * - Enabled
     - 21m 19.36s
     - 2,923,840

This pair reduced complete-build elapsed time by 87.07s (6.37%). GNU
``time`` reports maximum process RSS, not aggregate concurrent memory.
This is a single observation, not a statistical or cross-toolchain speedup
claim, and predates the separate kernels PCH extension below. The two
percentages must not be added to predict a combined improvement.

Kernels precompiled headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The private ``lib_onnx_kernels`` PCH includes ``test_case.h`` and
``node_helpers.h`` (which includes ``runtime_context.h``), plus standard
headers. ``ONNX_LIGHT_USE_PCH=OFF`` disables it through the existing PCH
wrapper; compiler-cache launchers still disable PCH by default.

A local GCC 13.3 comparison used Unix Makefiles, ``/usr/bin/c++``, Release
``-O3 -DNDEBUG``, kernels, Python bindings and tests enabled, and
``cmake --build <build-dir> --parallel 6``. The build directory was deleted
and configured afresh before each complete default-target build. Compiler
launchers were empty and ``CCACHE_DISABLE=1 SCCACHE_DISABLE=1`` were set.
Timing covered all compilation and linking, including PCH construction,
but excluded configuration.

.. list-table:: Single clean-build pair
   :header-rows: 1

   * - Kernels PCH
     - Elapsed
     - Maximum RSS (KiB)
     - Kernels PCH size (bytes)
   * - Existing headers
     - 23m 17.31s
     - 2,923,372
     - 152,606,520
   * - With ``node_helpers.h``
     - 22m 50.47s
     - 2,923,420
     - 179,695,275

This pair improved complete-build elapsed time by 26.84s (1.92%), with
essentially unchanged maximum RSS reported by GNU ``time -v``. That RSS is
not the aggregate memory of six concurrent compiler processes. Filesystem
caches were not flushed; this is not a statistical or cross-toolchain
speedup claim. Generated ``cmake_pch.hxx.gch.d`` dependencies contained
both runtime headers only after the extension. Target compiler flags and
definitions were unchanged.

Verify the install
^^^^^^^^^^^^^^^^^^


.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_lib)

Load a model with the proto-only target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build only the C++ protos, configure the independent ``onnx_proto/`` project
instead of the repository root:

.. code-block:: bash

    cmake -S onnx_proto -B build-proto -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/install
    cmake --build build-proto --parallel 6
    cmake --install build-proto

This installation exports ``find_package(onnx_proto CONFIG REQUIRED)`` and
``onnx_proto::lib_onnx_proto`` without building Python, schemas, or runtime
kernels. See :doc:`../../api/cpp/onnx_proto/index` for its build options.

The full *onnx-light* installation continues to expose the existing target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    add_executable(load_model main.cc)
    target_link_libraries(load_model PRIVATE onnx_proto::lib_onnx_proto)

The quoted headers come from the installed ``onnx_light`` target's
exported include directories.

.. code-block:: cpp

    #include "onnx.h"
    #include "stream.h"

    #include <exception>
    #include <iostream>

    int main() {
      try {
        onnx::ModelProto model;
        onnx::utils::FileStream stream("model.onnx");
        onnx::ParseOptions options;
        options.num_threads = 1;
        onnx::ParseModelProtoFromStream(model, stream, options);
        if (model.has_graph()) {
          // ref_graph() is safe after confirming the graph is present.
          std::cout << "Loaded " << model.ref_graph().node_size() << " nodes\n";
        }
        return 0;
      } catch (const std::exception &e) {
        std::cerr << "Error loading model.onnx: " << e.what() << "\n";
        return 1;
      }
    }


Notes
-----

* Python editable installs pull build dependencies from ``pyproject.toml`` and
  build the extension in place for local development.
* The C++ install flow exports CMake targets under ``onnx_light::...`` and does
  not require building the Python extension when
  ``-DONNX_LIGHT_BUILD_PYTHON=OFF`` is set.
* If the C++ library is installed to a non-standard prefix, configure the
  downstream project with ``-DCMAKE_PREFIX_PATH=<prefix>`` before calling
  ``find_package(onnx_light REQUIRED)``.
* Use ``onnx_light::lib_onnx_lib`` for higher-level ONNX features such as checker,
  shape inference, or version conversion. Use
  ``onnx_proto::lib_onnx_proto`` when the downstream code only needs proto
  parsing and serialization.

See also
--------

* :ref:`l-design-cpp-linking`
* :ref:`l-howto-load-save-onnx-files-cpp`
* :ref:`l-cpp-load-onnx-light-time-example`
