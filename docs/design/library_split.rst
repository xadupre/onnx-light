.. _l-design-library-split:

How the C++ libraries are split
===============================

*onnx-light* is intentionally split into several small static (or shared)
C++ libraries (also referred to as **assemblies**) instead of a single
monolithic archive.  The goal is to let any C++ project link **only**
what it actually needs: a program that just parses a :class:`~onnx_light.onnx_lib.ModelProto` does
not need operator schemas, a program that only registers schemas does
not need shape inference, a program that only evaluates models does not
need the full ONNX schema / checker stack, and so on.

Each library has a well defined responsibility and a minimal set of
dependencies on the other libraries.  The full dependency graph is
shown below (arrows point from a library to the library it depends
on).  The SVG below is generated from
:download:`library_split.dot <_static/library_split.dot>` with
``dot -Tsvg`` (Graphviz); regenerate it after editing the ``.dot`` source.

.. image:: _static/library_split.svg
   :alt: Dependency graph between the onnx-light C++ libraries
   :align: center

The same graph as an ASCII tree (rooted at ``lib_onnx_proto``, the
base dependency)::

    lib_onnx_proto
        └── lib_onnx_core
                ├── lib_onnx_op
                ├── lib_onnx_shape
                ├── lib_onnx_patterns
                ├── lib_onnx_kernels
                │       └── lib_onnx_backend_test
                ├── lib_onnx_gradient
                └── lib_onnx_manipulations
                        └── lib_onnx_lib

``lib_onnx_core`` is a lightweight intermediate library that sits
between ``lib_onnx_proto`` and all higher-level libraries.  It owns the
graph-node input helpers (``CollectExternalInputs``, ``CollectNodeInputs``,
``CollectRemainingInputs``), the symbolic dimension-expression library
(``onnx_core/expressions/expressions.h``), and the ``LightOpSchema``
operator-schema data structures (``onnx_core/light_op_schema/light_op_schema.h``)
so that ``lib_onnx_op`` (operator schemas),
``lib_onnx_manipulations``, ``lib_onnx_shape`` (shape inference), and
``lib_onnx_kernels`` (reference kernels) can share them without any of
them depending on each other.  The ``TensorType`` enumeration and the
``ToTypeString`` converter live one level lower, in ``lib_onnx_proto``
(``onnx_proto/type_helper.h``), so that ``lib_onnx_core`` itself can use
them.
``lib_onnx_op``, ``lib_onnx_manipulations``, ``lib_onnx_shape``,
``lib_onnx_patterns``, and ``lib_onnx_kernels`` are siblings that each depend
on ``lib_onnx_core``
(and transitively on ``lib_onnx_proto``).  ``lib_onnx_shape`` depends on
``lib_onnx_core`` only (it does **not** pull in ``lib_onnx_op`` or
``lib_onnx_lib``).  ``lib_onnx_kernels`` depends on ``lib_onnx_core``
only (it does **not** pull in ``lib_onnx_manipulations``).
``lib_onnx_manipulations`` gathers the schema-independent
``ModelProto`` manipulation helpers (text parser / printer, attribute and
tensor proto helpers, data-type name utilities); ``lib_onnx_lib`` (full
schemas, checker, shape inference, version converter) depends publicly
on it.
``lib_onnx_backend_test`` depends publicly on ``lib_onnx_kernels`` (and
transitively on ``lib_onnx_core`` / ``lib_onnx_proto``) because every
registered test case computes its expected outputs by invoking the
bundled C++ reference kernels.
``lib_onnx_gradient`` provides reverse-mode automatic differentiation for
ONNX graphs (``GradientOfNodes`` / ``GradientOfFunction``); it depends on
``lib_onnx_core`` only (and transitively on ``lib_onnx_proto``).  Like
``lib_onnx_kernels`` and ``lib_onnx_backend_test`` it belongs to the
extended build variant and is only built when
``ONNX_LIGHT_BUILD_KERNELS=ON``.

Core/extension registration pattern (examples)
----------------------------------------------

The same design pattern is reused across shape inference, peak-memory
estimation, kernels, backend tests, and operator schemas:
``onnx_core`` defines the generic types / registries / dispatch APIs, while
the concrete ONNX operator implementations live in sibling libraries
(``onnx_extensions`` and ``onnx_op``) and are wired in explicitly.

.. list-table::
    :header-rows: 1
    :widths: 20 40 40

    * - Feature
      - Core side (mechanism only)
      - Extension side (concrete implementations)
    * - Shape inference
      - ``onnx_core/shapes/dispatch_table.h`` defines
        ``ComputeShapeFn`` + ``RegisterComputeShapeFn`` /
        ``DispatchTable``.
      - ``onnx_extensions/shapes/dispatch_table.cc`` builds
        ``BuiltinShapeFunctions()`` and registers them through
        ``RegisterShapeFunctions()``.
    * - Peak memory
      - ``onnx_core/shapes/dispatch_table.h`` defines
        ``ComputePeakMemoryFn`` + ``RegisterComputePeakMemoryFn`` /
        ``ComputePeakMemory``.
      - ``onnx_extensions/shapes/dispatch_table.cc`` builds
        ``BuiltinPeakMemoryFunctions()`` and registers them through
        ``RegisterPeakMemoryFunctions()``.
    * - Runtime kernels
      - ``onnx_core/runtime/kernels/kernel_dispatch_table.h`` defines
        ``NodeKernelFn`` + ``RegisterKernelFn`` / ``KernelDispatchTable``.
      - ``onnx_extensions/kernels/kernel_dispatch_table.cc`` builds
        ``BuiltinKernelFunctions()`` and registers them through
        ``RegisterKernelFunctions()``.
    * - Graph optimization patterns
      - ``onnx_core/builder/pattern_registry.h`` defines
        ``PatternFactory`` + ``RegisterPattern`` /
        ``CreateRegisteredPatterns``.
      - ``onnx_extensions/patterns/dispatch_table.cc`` registers the concrete
        patterns through ``RegisterPatterns()``.
    * - Backend tests
      - ``onnx_core/backend_test/test_case_registry.h/.cc`` defines
        ``TestCasesCollectorFn`` + ``RegisterTestCasesCollector`` /
        ``GetRegisteredCollectors``.
      - ``onnx_extensions/backend_test/collect_test_cases.cc`` registers
        every category collector and implements ``CollectTestCases``.
    * - Light operator schema
      - ``onnx_core/light_op_schema/light_op_schema.h`` defines
        ``LightOpSchema`` and related schema data structures.
      - ``onnx_op/operator_sets*.cc`` implements per-domain schema history,
        and ``onnx_op/operator_sets.cc`` aggregates them through
        ``GetAllOnnxOpSchemasWithHistory``.

For the first four rows above, explicit registration keeps ``onnx_core``
independent from the extension libraries:

.. code-block:: cpp

    // shape inference + peak memory
    onnx_shapes::RegisterShapeFunctions();
    onnx_shapes::RegisterPeakMemoryFunctions();

    // runtime kernels
    onnx_kernels::RegisterKernelFunctions();

    // graph optimization patterns
    onnx_patterns::RegisterPatterns();

This means a downstream binary can keep linking minimal. If it never calls
shape inference, peak-memory estimation, runtime execution, graph-pattern
registration, or backend-test collection, it does not need to link the
corresponding extension library.

When installed (``cmake --install``) all libraries are exported under the
``onnx_light::`` namespace and can be consumed individually through
``find_package(onnx_light)``.

Summary of each library
-----------------------

.. list-table::
    :header-rows: 1
    :widths: 10 20

    * - Library / CMake target and Sources
      - What it contains
    * - ``onnx_light::lib_onnx_proto``:
        ``onnx_light/onnx_proto/``,
        ``onnx_light/onnx_helpers/onnx_light_helpers.cc``
      - Protobuf-compatible message types (:class:`~onnx_light.onnx_lib.ModelProto`,
        :class:`~onnx_light.onnx_lib.GraphProto`, :class:`~onnx_light.onnx_lib.NodeProto`, :class:`~onnx_light.onnx_lib.TensorProto`, ...), parser /
        serializer, external data and (optional) AES-256 encrypted
        save / load.  Built as **SHARED** when
        ``ONNX_LIGHT_BUILD_PYTHON=ON`` so that every Python extension
        shares the same proto class registrations; built as **STATIC**
        for pure C++ consumers.
    * - ``onnx_light::lib_onnx_core``:
        ``onnx_light/onnx_core/``
      - Intermediate library that sits between ``lib_onnx_proto`` and all
        higher-level libraries.  Owns the graph-node input helpers
        (``CollectExternalInputs``, ``CollectNodeInputs``,
        ``CollectRemainingInputs``), the symbolic dimension-expression
        library, the ``LightOpSchema`` operator-schema data structures, the
        runtime execution engine, and the (initially empty) kernel and
        shape-inference **dispatch tables** so that ``lib_onnx_op``,
        ``lib_onnx_manipulations``, ``lib_onnx_shape``, and
        ``lib_onnx_patterns`` and ``lib_onnx_kernels`` can share them without
        depending on each other.
        It implements *all* the generic mechanisms but registers **no**
        concrete operators or optimization patterns. The actual schemas,
        kernels, shape-inference functions, peak-memory functions, and graph
        patterns are **registered** into those shared registries by the
        extension libraries through their
        ``Register*Functions()`` entry points (``RegisterKernelFunctions``,
        ``RegisterShapeFunctions``, ``RegisterPeakMemoryFunctions``,
        ``RegisterPatterns``, ...).
        Depends publicly on ``lib_onnx_proto`` (which owns the
        ``TensorType`` enumeration and ``ToTypeString`` converter).
    * - ``onnx_light::lib_onnx_op``:``onnx_light/onnx_op/``
      - Lightweight ``LightOpSchema`` registrations for ONNX operator
        domains (math, logical, tensor, sequence, traditional ML, ...),
        built on top of the ``LightOpSchema`` data structures defined in
        ``lib_onnx_core`` (``onnx_core/light_op_schema/light_op_schema.h``,
        ``core::schema`` namespace).  Does not depend on shape
        inference and does not pull in the full ONNX defs.  Useful when
        only the operator catalogue is needed.  Depends publicly on
        ``lib_onnx_core`` (and transitively on ``lib_onnx_proto``).
    * - ``onnx_light::lib_onnx_manipulations`` (in-tree target
        ``lib_onnx_manipulations``):
        ``onnx_light/onnx_manipulations/``,
        ``onnx_light/onnx_lib/common/``
      - Foundational ``common`` utilities (``Status``, assertions, IR /
        proto conversion, ...) plus the schema-independent ``ModelProto`` /
        ``GraphProto`` manipulation helpers: the ONNX text-format parser
        (``parser``) and printer (``printer``), attribute and tensor proto
        construction helpers (``attr_proto_util``, ``tensor_proto_util``,
        ``tensor_util``), the data-type name utilities (``data_type_utils``),
        and the compose helpers.  Depends publicly on ``lib_onnx_core``
        (and transitively on ``lib_onnx_proto``).  ``lib_onnx_lib`` depends
        on it.
    * - ``onnx_light::lib_onnx_lib`` (in-tree target ``lib_onnx_lib``):
        ``onnx_light/onnx_lib/defs/``,
        ``onnx_light/onnx_lib/checker.cc``,
        ``onnx_light/onnx_lib/inliner/``,
        ``onnx_light/onnx_lib/shape_inference/``,
        ``onnx_light/onnx_lib/version_converter/``
      - Full ONNX-compatible operator schemas (with history), checker,
        inliner, shape inference and version converter.  This is the
        target to link for the *complete* ONNX-light experience.
        Depends publicly on ``lib_onnx_manipulations`` (and transitively
        on ``lib_onnx_proto``).
    * - ``lib_onnx_shape`` (exported as ``onnx_light::lib_onnx_shape``):
        ``onnx_light/onnx_extensions/shapes/``
      - Shape-inference dispatch table, expression engine for small
        tensor / backward-propagation shape inference, and graph
        optimization helpers.  Depends publicly on ``lib_onnx_core``
        (and transitively on ``lib_onnx_proto``).  Does **not** depend
        on ``lib_onnx_op``.
    * - ``onnx_light::lib_onnx_patterns`` (in-tree target
        ``lib_onnx_patterns``):
        ``onnx_light/onnx_extensions/patterns/``
      - Concrete ONNX graph-rewriting patterns and the explicit
        ``onnx_patterns::RegisterPatterns`` entry point. The generic
        ``PatternOptimization`` interface, registry, and optimizer context
        remain in ``lib_onnx_core``. Depends publicly on ``lib_onnx_core`` and
        is not linked back into it.
    * - ``onnx_light::lib_onnx_kernels`` (in-tree target ``lib_onnx_kernels``):
        ``onnx_light/onnx_extensions/kernels/``
      - C++ **reference implementation** of the ONNX operators used to
        evaluate models in-process.  Contains the runtime data model
        (``struct Tensor``, ``struct Sequence``, :class:`~onnx_light.onnx.reference.RuntimeContext`,
        ``RunGraph`` / ``RunFunction`` / ``RunModel``) and a kernel for
        each supported operator under ``onnx_extensions/kernels/kernels/<group>/``
        (``math``, ``logical``, ``nn``, ``tensor``, ``sequence``,
        ``controlflow``, ``traditionalml``, ...).  Together with the
        ``SplitMix64``-based deterministic pseudo-random helpers it
        provides everything required to run a node or a full model
        without depending on any third-party runtime.  Intentionally
        **independent** from ``lib_onnx_lib`` / ``lib_onnx_op`` /
        ``lib_onnx_manipulations``; depends publicly on ``lib_onnx_core``
        (and transitively on ``lib_onnx_proto``).
    * - ``onnx_light::lib_onnx_backend_test`` (in-tree target
        ``lib_onnx_backend_test``):
        ``onnx_light/onnx_extensions/backend_test/``
      - Backend test infrastructure (``struct TestCase``, the
        ``Expect`` helper and the ``CollectTestCases`` registry) plus
        the per-operator ``RegisterXxxCases`` functions that build
        single-node models and compute their expected outputs by
        invoking the reference kernels from ``lib_onnx_kernels``.
        Depends publicly on ``lib_onnx_kernels`` (and transitively on
        ``lib_onnx_core`` / ``lib_onnx_proto``); intentionally **independent**
        from ``lib_onnx_lib`` / ``lib_onnx_op`` / ``lib_onnx_manipulations``.
    * - ``onnx_light::lib_onnx_gradient`` (in-tree target
        ``lib_onnx_gradient``):
        ``onnx_light/onnx_extensions/gradient/``
      - Reverse-mode automatic differentiation for ONNX graphs.  Provides
        ``GradientOfNodes`` (from a sequence of ``NodeProto`` plus graph
        metadata) and ``GradientOfFunction`` (from a ``FunctionProto``),
        each returning a gradient ``FunctionProto`` encoding the backward
        computation.  Depends publicly on ``lib_onnx_core`` (and transitively
        on ``lib_onnx_proto``).  Part of the extended build variant, so it is
        only built when ``ONNX_LIGHT_BUILD_KERNELS=ON``.

What to link for a given use case
---------------------------------

The libraries are designed so that a downstream project links **the
smallest set** that covers its needs.  The most common scenarios are:

* **Read / write ONNX models, including > 2 GB files, external data or
  encrypted models, without any operator notion** — link
  ``onnx_light::lib_onnx_proto``.
* **Enumerate or look up ONNX operator schemas** (for example to drive
  code generation or to validate node op types) without paying for
  shape inference — link ``onnx_light::lib_onnx_op``.
* **Parse / print ONNX text models and manipulate ``ModelProto`` /
  ``GraphProto``** (attribute and tensor proto helpers, data-type name
  utilities, graph-input collection) without pulling in the operator
  schemas — link ``onnx_light::lib_onnx_manipulations``.
* **Full ONNX feature set** (schemas with history, checker, inliner,
  shape inference, version conversion) — link ``onnx_light::lib_onnx_lib``.
* **Shape inference** — link
  ``onnx_light::lib_onnx_shape`` (which transitively pulls
  ``lib_onnx_core`` and ``lib_onnx_proto``).
* **Use the standard graph-rewriting patterns** — link
  ``onnx_light::lib_onnx_patterns`` and call
  ``onnx_patterns::RegisterPatterns()`` before creating the registered
  pattern instances. Custom-only optimizers can link ``lib_onnx_core`` and
  either pass patterns directly to ``GraphGraph`` or call
  ``core::builder::RegisterPattern``. See
  :ref:`l-design-cpp-linking-patterns` for complete selection, registration,
  and linking examples.
* **Evaluate ONNX nodes / graphs / models in C++** using the bundled
  reference kernels (runtime ``struct Tensor``, ``RunGraph`` /
  ``RunFunction`` / ``RunModel``, ``SplitMix64`` RNG, ...) without
  pulling in any backend-test fixtures — link
  ``onnx_light::lib_onnx_kernels``.
* **Run backend tests / evaluate models in C++** using the built-in
  reference kernels — link ``onnx_light::lib_onnx_backend_test`` (which
  transitively pulls ``lib_onnx_kernels``, ``lib_onnx_core`` and
  ``lib_onnx_proto``).  It can be combined with ``onnx_light::lib_onnx_lib``
  when both schema validation and execution are needed.
* **Compute gradient functions** (reverse-mode automatic differentiation
  of ONNX graphs) — link ``onnx_light::lib_onnx_gradient`` (which
  transitively pulls ``lib_onnx_core`` and ``lib_onnx_proto``).

The Python extensions follow the same split and each link to the
minimal C++ library that provides their feature.  See
:ref:`l-design-cpp-linking` for the full discussion of the Python /
C++ packaging and of the shared ``lib_onnx_proto`` runtime.
