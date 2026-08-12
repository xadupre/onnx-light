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

.. _l-design-cpp-linking-cmake-options:

CMake configure-time options
----------------------------

The top-level ``CMakeLists.txt`` exposes the following options.  All of them
can be set on the ``cmake`` command line with ``-D<NAME>=<VALUE>``.

.. list-table::
    :header-rows: 1
    :widths: 35 10 55

    * - Option
      - Default
      - Description
    * - ``ONNX_LIGHT_BUILD_PYTHON``
      - ``ON``
      - Build the nanobind Python extensions
        (``_onnxpyprotoop``, ``_onnxpyprotolib``, ``_onnxpycore``,
        ``_onnxpykernels`` and ``_onnxpybackend``).  Turn ``OFF`` for a
        pure C++ build; this also switches ``lib_onnx_proto`` from a
        shared library back to a static archive.
    * - ``ONNX_LIGHT_INSTALL``
      - ``ON``
      - Install the C++ libraries, headers and the exported
        ``onnx_light`` CMake package on ``cmake --install``.  Turn ``OFF``
        when building only for in-tree consumption (for example from a
        parent ``add_subdirectory``).
    * - ``ONNX_LIGHT_BUILD_KERNELS``
      - ``ON``
      - Build the operator-kernel runtime (``lib_onnx_kernels``) and the
        backend-test case registry (``lib_onnx_backend_test``).  Turn
        ``OFF`` to install only the schema / checker / shape-inference /
        version-converter / proto libraries; incompatible with
        ``ONNX_LIGHT_BUILD_PYTHON=ON`` and ``ONNX_LIGHT_BUILD_TESTS=ON``.
    * - ``ONNX_ML``
      - ``ON``
      - Enable ``ai.onnx.ml`` (traditional ML) operator support.
    * - ``ONNX_LIGHT_BUILD_TESTS``
      - ``OFF``
      - Build the C++ unit-test executable ``test_onnx_light`` and
        register the tests with CTest.  Requires
        ``ONNX_LIGHT_BUILD_KERNELS=ON``.
    * - ``ONNX_LIGHT_BUILD_BENCHMARKS``
      - ``OFF``
      - Build the C++ benchmark executables from ``benchmarks/bench_*.cc``.
    * - ``ONNX_LIGHT_BENCH_GPROF``
      - ``OFF``
      - Compile the benchmark executables with ``-pg`` for gprof
        profiling.  Only meaningful with
        ``ONNX_LIGHT_BUILD_BENCHMARKS=ON``.
    * - ``ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX``
      - ``OFF``
      - Fetch and build upstream ``onnx`` (the protobuf-based reference
        implementation) via ``FetchContent`` to enable the
        ``BENCH_HAS_UPSTREAM_ONNX`` side-by-side comparison block in
        ``bench_load_file``.  Only meaningful with
        ``ONNX_LIGHT_BUILD_BENCHMARKS=ON``.
    * - ``ONNX_LIGHT_BUILD_FUZZERS``
      - ``OFF``
      - Build the libFuzzer-instrumented fuzz harnesses
        (``fuzz/fuzz_*.cc``) and the ``make_seed_corpus`` helper.
        Requires Clang and adds ``-fsanitize=fuzzer,address`` by default.
    * - ``ONNX_LIGHT_FUZZER_SANITIZERS``
      - ``address``
      - Comma-separated sanitizer list passed to ``-fsanitize=`` for the
        fuzz harnesses (the ``fuzzer`` sanitizer is always added
        automatically).  Only meaningful with
        ``ONNX_LIGHT_BUILD_FUZZERS=ON``.
    * - ``ONNX_HARDENING``
      - ``OFF``
      - Opt in to the `OpenSSF Compiler Options Hardening Guide for C and
        C++ <https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html>`_
        baseline.  When ``ON``, every onnx-light library, Python extension,
        test, and benchmark target receives the recommended compile and
        link flags (``_FORTIFY_SOURCE=3``, ``_GLIBCXX_ASSERTIONS``,
        ``-fstack-protector-strong``, ``-fstack-clash-protection``,
        ``-fcf-protection=full``, ``-fstrict-flex-arrays=3``,
        ``-ftrivial-auto-var-init=zero``, ``-Wformat=2``,
        ``-Werror=format-security``, ``-z noexecstack``, ``-z relro``,
        ``-z now``, on GCC/Clang and ``/GS``,
        ``/guard:cf``, ``/Qspectre``, ``/sdl``, ``/DYNAMICBASE``,
        ``/NXCOMPAT``, ``/CETCOMPAT`` on MSVC).  When ``/Qspectre`` is
        enabled on MSVC, the configure step also prepends the matching
        ``VCToolsInstallDir/lib/spectre/<arch>`` directory to the linker
        search path so the Spectre-mitigated CRT/STL libraries are used
        when that Visual Studio component is installed.  Each flag is
        probed by the configure step and silently skipped when the active
        toolchain does not accept it.  See ``cmake/Hardening.cmake`` for
        the full list.

.. _l-design-cpp-linking-no-kernels:

Build without the backend tests and kernels
--------------------------------------------

The operator-kernel runtime (``lib_onnx_kernels``) and the backend-test case
registry (``lib_onnx_backend_test``) are by far the largest libraries in the
build.  When downstream code only needs the schema / checker / shape-inference /
version-converter / proto layer, pass ``-DONNX_LIGHT_BUILD_KERNELS=OFF`` at
configure time to skip building and installing them:

.. code-block:: bash

    cmake -S . -B build-install \
          -DCMAKE_BUILD_TYPE=Release \
          -DONNX_LIGHT_BUILD_PYTHON=OFF \
          -DONNX_LIGHT_BUILD_KERNELS=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build build-install
    cmake --install build-install

The exported CMake package then provides only the kernel-free targets —
``onnx_light::lib_onnx_lib``, ``onnx_light::lib_onnx_manipulations``,
``onnx_light::lib_onnx_op``, ``onnx_light::lib_onnx_shape``,
``onnx_light::lib_onnx_patterns`` and ``onnx_light::lib_onnx_proto``.
``onnx_light::lib_onnx_kernels`` and
``onnx_light::lib_onnx_backend_test`` are not built and not part of the package.

``ONNX_LIGHT_BUILD_KERNELS=OFF`` is incompatible with
``ONNX_LIGHT_BUILD_PYTHON=ON`` and ``ONNX_LIGHT_BUILD_TESTS=ON``: the Python
extensions and the C++ unit tests both require the kernels.

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
version-converter / proto layer (``onnx_light::lib_onnx_lib``,
``onnx_light::lib_onnx_manipulations``, ``onnx_light::lib_onnx_op``,
``onnx_light::lib_onnx_shape``, ``onnx_light::lib_onnx_patterns``,
``onnx_light::lib_onnx_proto``), pass
``-DONNX_LIGHT_BUILD_KERNELS=OFF`` at configure time to skip building and
installing the much larger ``lib_onnx_kernels`` (operator-kernel runtime)
and ``lib_onnx_backend_test`` (backend-test case registry) libraries.
``onnx_light::lib_onnx_kernels`` and ``onnx_light::lib_onnx_backend_test`` are then
not part of the exported CMake package.

Then downstream projects can rely on the exported CMake targets:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_lib)

Use ``onnx_light::lib_onnx_lib`` when the code needs higher-level ONNX features
implemented in ``onnx_light/onnx_lib`` such as operator schemas, checker, shape
inference, or version conversion.

For protobuf-compatible message parsing/serialization only, downstream code can
link just the lighter proto target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_proto)

That is sufficient when the program only manipulates :class:`~onnx_light.onnx_lib.ModelProto` /
:class:`~onnx_light.onnx_lib.GraphProto` data and does not need any notion of operators.

For manual registration of lightweight math operator schemas without shape
inference support, downstream code can link:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_op)

To parse / print ONNX text models and manipulate :class:`~onnx_light.onnx_lib.ModelProto` /
:class:`~onnx_light.onnx_lib.GraphProto` (attribute and tensor proto helpers, data-type name
utilities, graph-input collection) without pulling in the operator schemas,
link the manipulations target, which only depends on ``lib_onnx_proto``:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_manipulations)

When the standalone shape-inference dispatch is needed without the full
``onnx_light::lib_onnx_lib`` checker/inliner/version converter, link:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_shape)

The generic graph optimizer and custom-pattern registry are part of
``onnx_light::lib_onnx_core``. To use the standard ONNX rewrite patterns, link
the extension library and register its patterns explicitly:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_patterns)

.. code-block:: cpp

    #include "onnx_core/builder/pattern_registry.h"
    #include "onnx_extensions/patterns/dispatch_table.h"

    onnx_patterns::RegisterPatterns();
    auto patterns = core::builder::CreateRegisteredPatterns();

Applications that provide only custom patterns can instead link
``onnx_light::lib_onnx_core`` and call
``core::builder::RegisterPattern`` directly.

To evaluate ONNX nodes / graphs / models in-process using the bundled C++
**reference implementation** of the ONNX operators (runtime
``struct Tensor``, ``RunGraph`` / ``RunFunction`` / ``RunModel``, the
``SplitMix64``-based deterministic RNG, ...), link the kernels target:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_kernels)

The kernels live under ``onnx_light/onnx_extensions/kernels/kernels/<group>/`` and
form a self-contained runtime that depends on
``onnx_light::lib_onnx_proto`` and ``onnx_light::lib_onnx_manipulations``
(for the graph-manipulation helpers).  See
:doc:`../api/cpp/onnx_extensions/kernels/index` for the full C++ API reference.

To additionally pull in the backend-test infrastructure
(``struct TestCase``, ``Expect()`` helper, per-operator
``RegisterXxxCases`` registries used by every ``CollectTestCases``
call), link:

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_backend_test)

``onnx_light::lib_onnx_backend_test`` publicly depends on
``onnx_light::lib_onnx_kernels`` (which transitively brings in
``onnx_light::lib_onnx_proto``) and is intentionally independent from
``onnx_light::lib_onnx_lib`` / ``onnx_light::lib_onnx_op``; it can be
combined with ``onnx_light::lib_onnx_lib`` when both schema validation and
execution are needed in the same binary.

To compute gradient functions (reverse-mode automatic differentiation of
ONNX graphs via ``GradientOfNodes`` / ``GradientOfFunction``), link the
gradient target, which depends on ``lib_onnx_core`` (and transitively on
``lib_onnx_proto``):

.. code-block:: cmake

    find_package(onnx_light REQUIRED)
    target_link_libraries(my_target PRIVATE onnx_light::lib_onnx_gradient)

See :doc:`../api/cpp/onnx_extensions/gradient/index` for the full C++ API
reference.

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
parsing/serialization is needed, or ``lib_onnx_op``, ``lib_onnx_manipulations``,
``lib_onnx_shape``, ``lib_onnx_patterns``, ``lib_onnx_kernels``,
``lib_onnx_backend_test`` or ``lib_onnx_gradient`` for the
corresponding feature subset.  This uses the
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

The Python package ships up to six nanobind extension modules,
``onnx_light.onnx_py._onnxpyprotoop``,
``onnx_light.onnx_py._onnxpyprotolib``,
``onnx_light.onnx_py._onnxpycore``,
``onnx_light.onnx_py._onnxpykernels``,
``onnx_light.onnx_py._onnxpybackend`` and
``onnx_light.onnx_py._onnxpygradient``.  The first three are always built;
the last three (``_onnxpykernels``, ``_onnxpybackend`` and
``_onnxpygradient``) belong to the extended build variant and are only
present when ``ONNX_LIGHT_BUILD_KERNELS=ON``.  They all need access to the proto
classes (:class:`~onnx_light.onnx_lib.ModelProto`, :class:`~onnx_light.onnx_lib.NodeProto`, :class:`~onnx_light.onnx_lib.TensorProto`, ...) defined in
``onnx_light/onnx_proto``.  How do the extensions agree on a single
``nb::class_<ModelProto>`` registration so that values can flow between
them without a serialise/parse round-trip?

When ``ONNX_LIGHT_BUILD_PYTHON=ON``, ``CMakeLists.txt`` builds
``lib_onnx_proto`` as a **shared** library (``liblib_onnx_proto.so`` /
``.dylib`` / ``.dll``) instead of a static archive.  All the
extensions link against that single shared
object (directly or transitively through ``lib_onnx_lib`` /
``lib_onnx_op`` / ``lib_onnx_shape`` / ``lib_onnx_kernels`` /
``lib_onnx_backend_test`` / ``lib_onnx_gradient``), and
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
:class:`~onnx_light.onnx_lib.ModelProto` references coming from ``_onnxpycore`` or
``_onnxpybackend`` against the
``nb::class_<ModelProto>`` that ``_onnxpyprotoop`` registered.  In
practice, only ``_onnxpyprotoop`` declares
``nb::class_<NodeProto>`` / ``nb::class_<ModelProto>`` / ...; the
``_onnxpyprotolib``, ``_onnxpycore``, ``_onnxpykernels``,
``_onnxpybackend`` and ``_onnxpygradient`` modules return proto values by
reference (for example
``TestCase.model``, see ``onnx_light/onnx_py/_onnxpy_backend_test.cc``)
and let the shared registry produce a Python object backed by the same
binding.  Every Python module that consumes a proto value imports the
proto classes directly from ``_onnxpyprotoop`` (for example
``from ..onnx_py._onnxpyprotoop import ModelProto``), which guarantees that
``_onnxpyprotoop`` is loaded — and its ``nb::class_<ModelProto>`` binding
registered — before any ``_onnxpyprotolib`` / ``_onnxpycore`` /
``_onnxpykernels`` / ``_onnxpybackend`` / ``_onnxpygradient`` accessor is
used.

See also
--------

* :ref:`l-design-library-split`
* :ref:`l-cpp-load-onnx-light-time-example`
