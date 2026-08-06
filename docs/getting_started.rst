Start
=====

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

    pip install -C build-dir=build -C cmake.build-type=Debug -C cmake.define.ONNX_LIGHT_BUILD_TESTS=ON -e .[dev] -v
    ctest --test-dir build --output-on-failure

With ``setup.py``:

.. code-block:: bash

    python setup.py build_ext --inplace --build-temp build --cpp-tests
    ctest --test-dir build --output-on-failure

Alternatively, ``--run-cpp-tests`` builds the C++ unit tests (it implies
``--cpp-tests``) and runs them with ``ctest`` in one step:

.. code-block:: bash

    python setup.py build_ext --inplace --build-temp build --run-cpp-tests

On multi-config generators such as Visual Studio, add the matching
configuration to ``ctest``: use ``-C Debug`` when the build was configured with
``cmake.build-type=Debug``, and ``-C Release`` after ``python setup.py
build_ext --cpp-tests``.

Load a model with parallel tensor parsing:

.. code-block:: python

    import onnx_light.onnx

    model = onnx_light.onnx.load("model.onnx", num_threads=4)
    print(model.ir_version)

Replace ``onnx`` by ``onnx_light.onnx``
---------------------------------------

``onnx-light`` mirrors the public :epkg:`onnx` API in both Python and C++, so
most code written against the standard ``onnx`` package can be ported by changing
a few imports / includes.  See :ref:`l-howto-replace-onnx` for a complete
side-by-side recipe covering Python, C++ and the matching build setup.

Python
^^^^^^

The :mod:`onnx_light.onnx` module exposes the same protobuf message types
(:class:`~onnx_light.onnx_lib.ModelProto`,
:class:`~onnx_light.onnx_lib.TensorProto`, ...) and submodules
(``helper``, ``numpy_helper``, ``reference``, ``backend``).  Replace
``import onnx`` with ``import onnx_light.onnx as onnx`` and the matching
submodules:

.. code-block:: python

    # before
    import onnx
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    # after
    import onnx_light.onnx as onnx
    from onnx_light.onnx import helper, numpy_helper
    from onnx_light.onnx.reference import ReferenceEvaluator

The rest of the code (``onnx.load``, ``onnx.save``, ``helper.make_node``,
``ReferenceEvaluator``, ...) stays unchanged.  See
:ref:`l-onnx-tutorial` for a full set of examples ported from the upstream ONNX
introduction.

C++
^^^

``onnx-light`` replicates the upstream ``onnx`` C++ API under the
``onnx_light`` namespace.  Two macros make most sources compile unchanged:
``ONNX_NAMESPACE`` resolves to ``onnx_light`` and headers such as ``onnx_pb.h``
and ``checker.h`` keep their familiar names.  Replace the ``onnx/`` include root
with ``onnx_lib/`` (and the ``onnx::`` namespace with ``onnx_light::`` if it is
spelled out explicitly):

.. code-block:: cpp

    // before
    #include "onnx/onnx_pb.h"
    #include "onnx/checker.h"

    onnx::ModelProto model;
    onnx::checker::check_model(model);

    // after
    #include "onnx_lib/onnx_pb.h"
    #include "onnx_lib/checker.h"

    onnx_light::ModelProto model;
    onnx_light::checker::check_model(model);

Code that already uses the ``ONNX_NAMESPACE`` macro instead of a hard-coded
``onnx::`` qualifier needs only the include-path change.  Link against
``onnx_light::lib_onnx_lib`` (schemas / checker / shape inference) or a lighter
target — see :ref:`l-design-cpp-linking` for the full list of CMake targets and
:epkg:`C++ onnx-light examples` for runnable programs.

Build without the backend tests and kernels
--------------------------------------------

The operator-kernel runtime (``lib_onnx_kernels``) and the backend-test case
registry (``lib_onnx_backend_test``) are the largest parts of the build.  When
you only need the schema / checker / shape-inference / version-converter / proto
layer, pass ``-DONNX_LIGHT_BUILD_KERNELS=OFF`` at configure time to skip building
them:

.. code-block:: bash

    cmake -S . -B build-install \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DONNX_LIGHT_BUILD_KERNELS=OFF
    cmake --build build-install

``ONNX_LIGHT_BUILD_KERNELS=OFF`` is incompatible with
``ONNX_LIGHT_BUILD_PYTHON=ON`` and ``ONNX_LIGHT_BUILD_TESTS=ON``, so it is meant
for pure C++ builds.  See :ref:`l-design-cpp-linking-no-kernels` for the
matching CMake workflow and the list of targets that remain available.

Source code: `https://github.com/xadupre/onnx-light <https://github.com/xadupre/onnx-light>`_
