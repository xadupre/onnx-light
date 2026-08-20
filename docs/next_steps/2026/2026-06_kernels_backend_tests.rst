.. _l-next-steps-kernels-backend-tests:

Operator kernels and the C++ backend tests
===========================================

:Date: 2026-06

**complete**

Objective
+++++++++

:epkg:`onnx` ships a large corpus of *backend tests*: one reference model per
operator variant together with the input and expected output tensors.  Upstream
they live in Python (``onnx/backend/test/case``) and are consumed by a Python
runtime.  ``onnx-light`` needs the same coverage, but from C++ and without a
Python dependency, so that the native runtime can be validated in every build
(including builds where Python is disabled).

The objective was twofold:

* provide a native execution engine — a **kernel** per operator plus a
  dispatch table that resolves a :class:`~onnx_light.onnx_lib.NodeProto` to the
  code that runs it;
* provide a native **backend-test** corpus so those kernels are exercised
  against known-good input/output pairs directly from the C++ test binary.

Post-mortem
+++++++++++

The two concerns were kept in separate libraries so the runtime core never
depends on any particular kernel, and so the test corpus can be built or
skipped independently.

Two libraries, one dispatch table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runtime core (``lib_onnx_core``) owns the *mechanism* — the
``core::runtime::KernelDispatchTable`` and the ``KernelBase`` interface — but
knows nothing about individual operators.  The operator *implementations* live
in ``lib_onnx_kernels`` (``onnx_light/onnx_extensions/kernels``).  A single
entry point, ``onnx_kernels::RegisterKernelFunctions()``, populates the dispatch
table with a factory for every built-in operator, keyed by ``"<domain>:<op>"``
(for example ``"ai.onnx:Add"``).  Registration is idempotent, so linking the
kernels does not force them to be registered until the caller asks for it.

Each kernel derives from ``KernelBase`` and follows the same shape:

.. code-block:: cpp

    class Add : public KernelBase {
      void Run(RuntimeContext &rt) override;      // dispatch entry point
      Tensor operator()(const Tensor &x, const Tensor &y, ...) const;
      void operator()(const Tensor &x, const Tensor &y, Tensor &out) const;
      static constexpr bool CanRunInPlace() noexcept { return true; }
    };

``Run`` reads the node inputs from the ``RuntimeContext``, calls the typed
``operator()``, and writes the outputs back.  The two ``operator()`` overloads
— one that allocates and returns, one that fills a pre-allocated tensor — let
the same code serve both the allocating path used by tests and the in-place
path the execution plan prefers when ``CanRunInPlace()`` allows an output to
alias an input.  Kernels are grouped by domain family under
``kernels/kernels/{math,nn,tensor,reduction,logical,...}``, each with an
``include_*.h`` declaration header and one ``kernel_*.cc`` per operator.  Shared
attribute parsing and tensor helpers live in ``kernel_run_helpers.h`` to avoid
per-kernel duplication.

The backend-test corpus
^^^^^^^^^^^^^^^^^^^^^^^^

The C++ corpus mirrors the upstream Python cases but stays lazy.  A
``core::backend_test::TestCase`` stores metadata (name, tag, tolerances,
declared input/output element counts) plus a ``std::function`` that builds the
single-node ``ModelProto`` and its data sets only when the case is actually
run.  Laziness matters because the benchmark variants declare millions of
elements; materializing every case up front would be wasteful.  The ``Expect``
helper captures node, inputs and outputs into that lambda and appends the case
to a per-category registry.

Cases are organized to match the kernels:
``backend_test/cases/{math,nn,tensor,...}`` hold the per-operator numeric
cases, while ``cases_for_shapes`` (constant, empty-shape, inference, in-place,
peak-memory, release, shape-tag), ``cases_numerical`` (NaN/Inf) and
``cases_runtime`` (model-local functions) cover the structural and edge-case
behavior that is not tied to a single operator.  ``collect_test_cases.cc``
registers every category collector through a static initializer and exposes
``CollectTestCases`` / ``CollectTestCasesByName`` / ``GetTestCaseByName`` plus a
``TestMode`` (``TEST`` vs ``BENCHMARK``) so the same definitions serve both
correctness and performance runs.

What worked
^^^^^^^^^^^

* Keeping the dispatch *mechanism* in the runtime core and the *kernels* in a
  separate library meant the core could be built, tested, and reused without
  pulling in hundreds of operator implementations.
* The lazy ``TestCase`` made a very large corpus cheap to enumerate: filtering
  by operator or by the ``_big_`` benchmark flag never materializes the models
  it skips.
* Sharing one node definition between the allocating and in-place ``operator()``
  overloads meant the numeric cases and the memory-reuse execution plan
  validated the *same* kernel code.
* Reusing upstream ONNX's reference outputs (transcribed into C++ cases) kept
  the native runtime honest against a well-known baseline.

Kernels on and off in CI
^^^^^^^^^^^^^^^^^^^^^^^^^

``ONNX_LIGHT_BUILD_KERNELS`` gates both ``lib_onnx_kernels`` and
``lib_onnx_backend_test``.  ``.github/workflows/ci_core.yml`` builds *both*
variants: a fast ``KERNELS=OFF`` preflight that exercises shape inference and
the runtime core without any operator code, and a full ``KERNELS=ON`` job that
links the real kernels and runs the backend corpus through the ``ctest`` /
``test_onnx_light`` binary.  Tests that assert a kernel is *absent* therefore
have to ``GTEST_SKIP`` when the kernels are linked in.

What remains
^^^^^^^^^^^^

Coverage tracks the operators onnx-light implements, not the entire ONNX
opset; unimplemented operators have neither a kernel nor a case.  Benchmark
cases share the correctness definitions but are only run on demand, so
performance regressions are not part of the default matrix.

See also
++++++++

* :ref:`l-next-steps-prepared-execution` — how resolved kernels are cached and
  replayed by the runtime session.
* :ref:`l-next-steps-processor-aware-kernel-tuning` — per-kernel tuning
  built on top of the dispatch table.
