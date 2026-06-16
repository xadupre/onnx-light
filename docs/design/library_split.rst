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
        ├── lib_onnx_op
        │       └── lib_onnx_optim
        └── lib_onnx_manipulations
                ├── lib_onnx_lib
                └── lib_onnx_kernels
                        └── lib_onnx_backend_test

``lib_onnx_op`` and ``lib_onnx_manipulations`` are independent siblings: each
links directly against ``lib_onnx_proto`` but neither depends on the other.
``lib_onnx_optim`` depends on ``lib_onnx_op`` only (it does **not** pull in
``lib_onnx_lib``).  ``lib_onnx_manipulations`` gathers the schema-independent
``ModelProto`` manipulation helpers (text parser / printer, attribute and
tensor proto helpers, data-type name utilities and the graph-manipulation
helpers); both ``lib_onnx_lib`` (full schemas, checker, shape inference,
version converter) and ``lib_onnx_kernels`` depend publicly on it.
``lib_onnx_backend_test`` depends publicly on ``lib_onnx_kernels`` (and
transitively on ``lib_onnx_manipulations`` / ``lib_onnx_proto``) because every
registered test case computes its expected outputs by invoking the bundled C++
reference kernels.

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
    * - ``onnx_light::lib_onnx_op``:``onnx_light/onnx_op/``
      - Lightweight ``LightOpSchema`` registrations for ONNX operator
        domains (math, logical, tensor, sequence, traditional ML, ...).
        Does not depend on shape inference and does not pull in the
        full ONNX defs.  Useful when only the operator catalogue is
        needed.  Depends publicly on ``lib_onnx_proto``.
    * - ``onnx_light::onnx_manipulations`` (in-tree target
        ``lib_onnx_manipulations``):
        ``onnx_light/onnx_manipulations/``
      - Schema-independent ``ModelProto`` / ``GraphProto`` manipulation
        helpers: the ONNX text-format parser (``parser``) and printer
        (``printer``), attribute and tensor proto construction helpers
        (``attr_proto_util``, ``tensor_proto_util``, ``tensor_util``),
        the data-type name utilities (``data_type_utils``) and the graph
        manipulation helpers (``graph_manipulations``, formerly
        ``onnx_proto/common_functions``).  Depends publicly on
        ``lib_onnx_proto``.  Both ``lib_onnx_lib`` and ``lib_onnx_kernels``
        depend on it.
    * - ``onnx_light::onnx_light`` (in-tree target ``lib_onnx_lib``):
        ``onnx_light/onnx_lib/common/``,
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
    * - ``lib_onnx_optim`` (exported as ``onnx_light::lib_onnx_optim``):
        ``onnx_light/onnx_optim/``
      - Shape-inference dispatch table, expression engine for small
        tensor / backward-propagation shape inference, and graph
        optimization helpers.  Depends publicly on ``lib_onnx_op``.
    * - ``onnx_light::onnx_kernels`` (in-tree target ``lib_onnx_kernels``):
        ``onnx_light/onnx_kernels/``
      - C++ **reference implementation** of the ONNX operators used to
        evaluate models in-process.  Contains the runtime data model
        (``struct Tensor``, ``struct Sequence``, :class:`~onnx_light.onnx.reference.RuntimeContext`,
        ``RunGraph`` / ``RunFunction`` / ``RunModel``) and a kernel for
        each supported operator under ``onnx_kernels/kernels/<group>/``
        (``math``, ``logical``, ``nn``, ``tensor``, ``sequence``,
        ``controlflow``, ``traditionalml``, ...).  Together with the
        ``SplitMix64``-based deterministic pseudo-random helpers it
        provides everything required to run a node or a full model
        without depending on any third-party runtime.  Intentionally
        **independent** from ``lib_onnx_lib`` / ``lib_onnx_op``;
        depends publicly on ``lib_onnx_manipulations`` (for the graph
        manipulation helpers) and transitively on ``lib_onnx_proto``.
    * - ``onnx_light::onnx_backend_test`` (in-tree target
        ``lib_onnx_backend_test``):
        ``onnx_light/onnx_backend_test/``
      - Backend test infrastructure (``struct TestCase``, the
        ``Expect`` helper and the ``CollectTestCases`` registry) plus
        the per-operator ``RegisterXxxCases`` functions that build
        single-node models and compute their expected outputs by
        invoking the reference kernels from ``lib_onnx_kernels``.
        Depends publicly on ``lib_onnx_kernels`` (and transitively on
        ``lib_onnx_proto``); intentionally **independent** from
        ``lib_onnx_lib`` / ``lib_onnx_op``.

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
  schemas — link ``onnx_light::onnx_manipulations``.
* **Full ONNX feature set** (schemas with history, checker, inliner,
  shape inference, version conversion) — link ``onnx_light::onnx_light``.
* **Shape inference and graph optimization passes** — link
  ``onnx_light::lib_onnx_optim`` (which transitively pulls
  ``lib_onnx_op`` and ``lib_onnx_proto``).
* **Evaluate ONNX nodes / graphs / models in C++** using the bundled
  reference kernels (runtime ``struct Tensor``, ``RunGraph`` /
  ``RunFunction`` / ``RunModel``, ``SplitMix64`` RNG, ...) without
  pulling in any backend-test fixtures — link
  ``onnx_light::onnx_kernels``.
* **Run backend tests / evaluate models in C++** using the built-in
  reference kernels — link ``onnx_light::onnx_backend_test`` (which
  transitively pulls ``lib_onnx_kernels`` and ``lib_onnx_proto``).  It
  can be combined with ``onnx_light::onnx_light`` when both schema
  validation and execution are needed.

The Python extensions follow the same split and each link to the
minimal C++ library that provides their feature.  See
:ref:`l-design-cpp-linking` for the full discussion of the Python /
C++ packaging and of the shared ``lib_onnx_proto`` runtime.
