.. _l-design-cpp-linking:

Linking *onnx-light* in C++
===========================

This page summarizes the design used to consume *onnx-light* as a standalone
C++ library from another project.

Runnable examples are available in :epkg:`C++ onnx-light examples`, including
``examples/load_onnx_light_time``, ``examples/save_onnx_light_time``,
``examples/build_save_load_onnx_proto``, ``examples/check_onnx_light_model``,
``examples/export_nnef``, ``examples/run_add_node_test`` and
``examples/run_backend_test_ort``.  Each example's ``CMakeLists.txt`` shows
which subset of the exported targets is needed.

See :ref:`l-design-library-split` for the full dependency graph of the C++
libraries and a per-target description.

Install and link model
----------------------

From the repository root, install the C++ library with CMake:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install
    cmake --install build-install

When downstream code only needs the schema / checker / shape-inference /
version-converter / proto layer (``onnx_light::onnx_light``,
``onnx_light::lib_onnx_op``, ``onnx_light::lib_onnx_optim``,
``onnx_light::lib_onnx_proto``), pass
``-DONNX_LIGHT_BUILD_KERNELS=OFF`` at configure time to skip building and
installing the much larger ``lib_onnx_kernels`` (operator-kernel runtime)
and ``lib_onnx_backend_test`` (backend-test case registry) libraries.
``onnx_light::onnx_kernels`` and ``onnx_light::onnx_backend_test`` are then
not part of the exported CMake package.

Then downstream projects can rely on the exported CMake targets:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::onnx_light)

Use ``onnx_light::onnx_light`` when the code needs higher-level ONNX features
implemented in ``onnx_light/onnx_lib`` such as operator schemas, checker, shape
inference, or version conversion.

For protobuf-compatible message parsing/serialization only, downstream code can
link just the lighter proto target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_proto)

That is sufficient when the program only manipulates ``ModelProto`` /
``GraphProto`` data and does not need any notion of operators.

For manual registration of lightweight math operator schemas without shape
inference support, downstream code can link:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_op)

When shape inference dispatch and graph optimization passes are also needed
(without pulling in the full ``onnx_light::onnx_light`` checker/inliner/version
converter), link the optim target instead, which transitively pulls in
``lib_onnx_op`` and ``lib_onnx_proto``:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_optim)

To evaluate ONNX nodes / graphs / models in-process using the bundled C++
**reference implementation** of the ONNX operators (runtime
``struct Tensor``, ``RunGraph`` / ``RunFunction`` / ``RunModel``, the
``SplitMix64``-based deterministic RNG, ...), link the kernels target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::onnx_kernels)

The kernels live under ``onnx_light/onnx_kernels/kernels/<group>/`` and
form a self-contained runtime that only depends on
``onnx_light::lib_onnx_proto``.  See
:doc:`../api/cpp/onnx_kernels/index` for the full C++ API reference.

To additionally pull in the backend-test infrastructure
(``struct TestCase``, ``Expect()`` helper, per-operator
``RegisterXxxCases`` registries used by every ``CollectTestCases``
call), link:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::onnx_backend_test)

``onnx_light::onnx_backend_test`` publicly depends on
``onnx_light::onnx_kernels`` (which transitively brings in
``onnx_light::lib_onnx_proto``) and is intentionally independent from
``onnx_light::onnx_light`` / ``onnx_light::lib_onnx_op``; it can be
combined with ``onnx_light::onnx_light`` when both schema validation and
execution are needed in the same binary.

This keeps downstream CMake files independent from hardcoded include paths and
library file names. If *onnx-light* is installed to a non-standard prefix,
configure the downstream project with ``-DCMAKE_PREFIX_PATH=<prefix>``.

Alternative without install
---------------------------

For monorepos or local development, a downstream CMake project can also include
*onnx-light* directly:

.. code-block:: cmake

    set(ONNX_LIGHT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory(path/to/onnx-light)
    target_link_libraries(my_target PRIVATE lib_onnx_lib)

Use the in-tree ``lib_onnx_proto`` target instead when only proto
parsing/serialization is needed, or ``lib_onnx_op``, ``lib_onnx_optim``,
``lib_onnx_kernels`` or ``lib_onnx_backend_test`` for the corresponding
feature subset.  This uses the
in-tree build targets directly instead of ``find_package``.

Excerpt from the example project
--------------------------------

The example CMake project in ``examples/load_onnx_light_time`` uses exactly
that pattern:

.. literalinclude:: ../../examples/load_onnx_light_time/CMakeLists.txt
    :language: cmake
    :lines: 28-37

Python extension modules and proto duplication
----------------------------------------------

The Python package ships five nanobind extension modules,
``onnx_light.onnx_py._onnxpyprotoop``,
``onnx_light.onnx_py._onnxpyprotolib``,
``onnx_light.onnx_py._onnxpyoptim``,
``onnx_light.onnx_py._onnxpykernels`` and
``onnx_light.onnx_py._onnxpybackend``.  All five need access to the proto
classes (``ModelProto``, ``NodeProto``, ``TensorProto``, ...) defined in
``onnx_light/onnx_proto``.  How do the extensions agree on a single
``nb::class_<ModelProto>`` registration so that values can flow between
them without a serialise/parse round-trip?

When ``ONNX_LIGHT_BUILD_PYTHON=ON``, ``CMakeLists.txt`` builds
``lib_onnx_proto`` as a **shared** library (``liblib_onnx_proto.so`` /
``.dylib`` / ``.dll``) instead of a static archive.  All five
extensions link against that single shared
object (directly or transitively through ``lib_onnx_lib`` /
``lib_onnx_op`` / ``lib_onnx_optim`` / ``lib_onnx_kernels`` /
``lib_onnx_backend_test``), and
the build installs every file side by side under
``onnx_light/onnx_py/``.  The extensions are linked with an ``$ORIGIN``
runtime path (``@loader_path`` on macOS) so the dynamic loader finds
``liblib_onnx_proto.so`` next to them at import time without any
``LD_LIBRARY_PATH`` setup.

Pure C++ consumers (``ONNX_LIGHT_BUILD_PYTHON=OFF``) keep the lighter
**static** variant they used to ship, so the existing
``find_package(onnx_light) -> onnx_light::lib_onnx_proto`` workflow is
unchanged.

Because ``liblib_onnx_proto.so`` is loaded only once per process, the
proto classes have a single set of out-of-line member definitions and a
single ``std::type_info`` instance.  Consequently
``&typeid(ModelProto)`` evaluates to the same pointer in every
extension, and nanobind's cross-module type registry resolves
``ModelProto`` references coming from ``_onnxpyoptim`` or
``_onnxpybackend`` against the
``nb::class_<ModelProto>`` that ``_onnxpyprotoop`` registered.  In
practice, only ``_onnxpyprotoop`` declares
``nb::class_<NodeProto>`` / ``nb::class_<ModelProto>`` / ...; the
``_onnxpyprotolib``, ``_onnxpyoptim``, ``_onnxpykernels`` and
``_onnxpybackend`` modules return proto values by
reference (for example
``TestCase.model``, see ``onnx_light/onnx_py/_onnxpy_backend_test.cc``)
and let the shared registry produce a Python object backed by the same
binding.  The package's ``onnx_light/onnx_py/_onnxpy.py`` shim imports
``_onnxpyprotoop`` before ``_onnxpyprotolib``, ``_onnxpyoptim``,
``_onnxpykernels`` and ``_onnxpybackend`` to
guarantee that the
``ModelProto`` binding exists by the time any ``_onnxpyprotolib``,
``_onnxpyoptim``, ``_onnxpykernels`` or ``_onnxpybackend`` accessor is used.

See also
--------

* :ref:`l-design-library-split`
* :ref:`l-cpp-load-onnx-light-time-example`
