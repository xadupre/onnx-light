.. _l-howto-install-onnx-light:

How to install *onnx-light*
===========================

This page aligns the most common *onnx-light* installation workflows side by
side for Python and C++. Each row shows the equivalent commands for local
development, faster builds, and basic post-install validation.

Common install patterns
-----------------------

Install for local development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

          pip install -e .[dev] -v

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: bash

          cmake -S . -B build-install \
                -DCMAKE_BUILD_TYPE=Release \
                -DONNX_LIGHT_BUILD_PYTHON=OFF \
                -DCMAKE_INSTALL_PREFIX=/usr/local
          cmake --build build-install
          cmake --install build-install

Speed up the build
^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

          CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .[dev]

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: bash

          cmake --build build-install --parallel 8

Verify the install
^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

          python -c "import onnx_light; print(onnx_light.__version__)"

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cmake

          find_package(onnx_light REQUIRED)
          target_link_libraries(my_target PRIVATE onnx_light::onnx_light)

Load a model with the proto-only target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx as onnxl

          model = onnxl.load("model.onnx")

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cmake

          find_package(onnx_light REQUIRED)
          add_executable(load_model main.cc)
          target_link_libraries(load_model PRIVATE onnx_light::lib_onnx_proto)

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
* Use ``onnx_light::onnx_light`` for higher-level ONNX features such as checker,
  shape inference, or version conversion. Use
  ``onnx_light::lib_onnx_proto`` when the downstream code only needs proto
  parsing and serialization.

See also
--------

* :ref:`l-design-cpp-linking`
* :ref:`l-howto-load-save-onnx-files`
* :ref:`l-cpp-load-onnx-light-time-example`
