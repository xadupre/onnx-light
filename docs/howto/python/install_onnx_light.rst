.. _l-howto-install-onnx-light:
.. _l-howto-install-onnx-light-python:

How to install *onnx-light*
===========================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/install_onnx_light`.

This page covers *onnx-light* installation for Python, including local
development, faster builds, and basic post-install validation.

Common install patterns
-----------------------

Install for local development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash

    pip install -e .[dev] -v

Speed up the build
^^^^^^^^^^^^^^^^^^


.. code-block:: bash

    CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .[dev]

Verify the install
^^^^^^^^^^^^^^^^^^


.. code-block:: bash

    python -c "import onnx_light; print(onnx_light.__version__)"

Load a model with the proto-only target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    import onnx_light.onnx as onnxl

    model = onnxl.load("model.onnx")

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
* :ref:`l-howto-load-save-onnx-files-python`
* :ref:`l-cpp-load-onnx-light-time-example`
