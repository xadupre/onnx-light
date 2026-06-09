Getting Started
===============

Install the package in editable mode:

.. code-block:: bash

    pip install -e .[dev] -v

or

.. code-block:: bash

    python setup.py build_ext --inplace

To speed up compilation with multiple threads, pass ``--parallel`` (or ``-j``)
with the number of jobs:

.. code-block:: bash

    python setup.py build_ext --inplace --parallel 8

By default, ``python setup.py build_ext`` auto-enables parallel builds
(``--parallel <cpu_count>``) unless ``CMAKE_BUILD_PARALLEL_LEVEL`` is already set.

Alternatively, when installing with pip, control parallel builds using the
``CMAKE_BUILD_PARALLEL_LEVEL`` environment variable:

.. code-block:: bash

    CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .[dev]

Run a quick check:

.. code-block:: bash

    python -c "import onnx_light; print(onnx_light.__version__)"

Build and run the C++ unit tests from the editable build:

With ``pip install``:

.. code-block:: bash

    pip install -C build-dir=build -C cmake.build-type=Debug -C cmake.define.ONNX_LIGHT_BUILD_TESTS=ON -e .[dev]
    ctest --test-dir build --output-on-failure

With ``setup.py``:

.. code-block:: bash

    python setup.py build_ext --inplace --build-temp build --cpp-tests
    ctest --test-dir build --output-on-failure

On multi-config generators such as Visual Studio, add the matching
configuration to ``ctest``: use ``-C Debug`` when the build was configured with
``cmake.build-type=Debug``, and ``-C Release`` after ``python setup.py
build_ext --cpp-tests``.

Load a model with parallel tensor parsing:

.. code-block:: python

    import onnx_light.onnx

    model = onnx_light.onnx.load("model.onnx", num_threads=4)
    print(model.ir_version)

Source code: `https://github.com/xadupre/onnx-light <https://github.com/xadupre/onnx-light>`_
