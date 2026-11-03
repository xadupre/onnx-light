.. _l-howto-register-builtins-cpp:

:html_theme.sidebar_secondary.remove:

How to register a built-in kernel, test case, shape inference, light op or peak memory function
===============================================================================================

This guide covers the C++ interface. For the other language, see
:doc:`../python/register_builtin_operators`.

:ref:`l-howto-use-custom-kernel-cpp` and :ref:`l-howto-use-custom-shape-inference-cpp`
show how to plug behaviour into the runtime *from outside* the library. This
page is the contributor-facing counterpart: it explains how to add a new
**built-in** operator to *onnx-light* itself. Every operator is made of a few
independent pieces, each registered in its own static table:

* a **kernel** – the C++ code that computes the outputs, in ``lib_onnx_kernels``;
* a **backend test case** – reference inputs/outputs used to validate the kernel;
* a **shape inference** function – propagates dtypes and symbolic shapes, in
  ``lib_onnx_shape``;
* a **light op schema** – the operator definition (inputs, outputs, type
  constraints, documentation), in ``lib_onnx_op``;
* an optional **peak memory** function – estimates the additional peak memory the
  kernel needs, also in ``lib_onnx_shape``.

Each piece uses ``Abs`` (an element-wise unary math op) as the running example.
All the source directories are compiled with CMake ``GLOB_RECURSE`` patterns, so
adding a new ``.cc`` file is picked up automatically — no ``CMakeLists.txt`` edit
is required. Only the registration tables need a new entry.

Registration always happens in **C++** (each table is a C++ static). There is
**no separate Python registration step**: the extension exposes generic entry
points that read the same C++ dispatch tables, so once the libraries are rebuilt
(``pip install -e .`` or the CMake build) every piece is reachable from Python
automatically. This page covers the native registration steps.

.. contents::
    :local:

Register a kernel
-----------------

Kernels live under
``onnx_light/onnx_extensions/kernels/kernels/<category>/`` (``math``, ``nn``,
``tensor``, ...). A kernel is a class deriving from
:cpp:class:`onnx_light::core::runtime::KernelBase` and overriding ``Run``.


1. **Declare the class** in the category header, e.g.
   ``kernels/math/include_math_kernels.h``:

   .. code-block:: cpp

       /// Element-wise absolute value.
       class Abs : public KernelBase {
       public:
         void Run(RuntimeContext &rt) override;
         using KernelBase::KernelBase;
         Tensor operator()(const Tensor &x, RuntimeContext *rt = nullptr) const;
         void operator()(const Tensor &x, Tensor &output) const;

         /// Element-wise unary kernel: the output buffer may alias the input.
         static constexpr bool CanRunInPlace() noexcept { return true; }
       };

2. **Implement it** in ``kernels/math/kernel_abs.cc``. ``Run`` reads the
   node's inputs from the :cpp:class:`RuntimeContext` and writes the
   outputs back:

   .. code-block:: cpp

       void Abs::Run(RuntimeContext &rt) {
         const NodeProto &node = *node_;
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         SetOutput(node, 0, (*this)(x, &rt), rt);
       }

3. **Register it** by adding one entry to the ``BuiltinKernelFunctions``
   table in
   ``onnx_light/onnx_extensions/kernels/kernel_dispatch_table.cc``. The key
   is ``"<domain>:<op_type>"`` and ``MakeKernel<T>()`` builds the dispatch
   factory:

   .. code-block:: cpp

       {"ai.onnx:Abs", MakeKernel<onnx_kernels::kernel::Abs>()},

The empty default ONNX domain is normalised to ``ai.onnx``. The
registration happens in ``lib_onnx_kernels``; the runtime consults this
table through :cpp:func:`onnx_light::core::runtime::RunNode`.

Register a test case
--------------------

Backend test cases live under
``onnx_light/onnx_extensions/backend_test/cases/<category>/``. A case pairs a
:cpp:class:`NodeProto` with a lambda that produces the expected inputs and
outputs (typically by calling the kernel itself), through the
:cpp:func:`Expect` helper.


1. **Write the case function** in ``cases/math/cases_abs.cc``:

   .. code-block:: cpp

       void RegisterAbsCases(std::vector<TestCase> &registry, TestMode mode) {
         const OpsetId opset = DefaultOpset(13);
         const KernelContext ctx{opset};
         const onnx_kernels::kernel::Abs abs_kernel{ctx};

         NodeProto node;
         node.set_op_type("Abs");
         node.add_input("x");
         node.add_output("y");
         Expect(registry, std::move(node), "test_cc_abs", {opset}, [=]() -> IoData {
           Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
           Tensor y = abs_kernel(x);
           return IoData{{std::move(x)}, {std::move(y)}};
         });
       }

2. **Declare** ``RegisterAbsCases`` in ``cases/math/include_math_cases.h``
   with the visibility macro:

   .. code-block:: cpp

       ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAbsCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

3. **Register** it by adding one entry to the ``OpRegisterModeMap`` in
   ``cases/math/collect_math_cases.cc``:

   .. code-block:: cpp

       {"Abs", &RegisterAbsCases},

The per-category collectors are wired into the global registry (via
:cpp:func:`RegisterTestCasesCollector`) in
``onnx_light/onnx_extensions/backend_test/collect_test_cases.cc``, so the
new case is reachable from
:cpp:func:`onnx_light::core::backend_test::CollectTestCases`.

Register a shape inference
--------------------------

Shape functions live under
``onnx_light/onnx_extensions/shapes/shapes/<category>/``. A shape function has
signature ``void(ShapesContext &, const NodeProto &)`` and sets the symbolic
:cpp:class:`SymTensor` for every output.


1. **Implement it** in ``shapes/math/shape_abs.cc`` (and declare it in
   ``shapes/math/shape_math.h``):

   .. code-block:: cpp

       void ComputeShapeAbs(ShapesContext &ctx, const NodeProto &node, const char *x) {
         CheckNodeOpAndOutput(node, "Abs", "ComputeShapeAbs");
         const SymTensor &input = ctx.Get(x);
         // Abs is element-wise: the output dtype and shape match the input.
         ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
       }

2. **Register it** by adding an entry to the ``BuiltinShapeFunctions``
   table in ``onnx_light/onnx_extensions/shapes/dispatch_table.cc``. The
   lambda checks the input count and forwards to the shape function:

   .. code-block:: cpp

       {"ai.onnx:Abs",
        [](ShapesContext &ctx, const NodeProto &node) {
          RequireInputs(node, 1);
          math::ComputeShapeAbs(ctx, node, node.input(0).c_str());
        }},

``lib_onnx_shape`` is a static archive, so registration is not automatic
on link: :cpp:func:`RegisterShapeFunctions` (in the same file) copies every
builtin entry into the shared ``core::shapes`` dispatch table via
:cpp:func:`onnx_light::core::shapes::RegisterComputeShapeFn`. It is
idempotent and must be called before running shape inference.

Register a light op
-------------------

Operator schemas (the "light" equivalent of ONNX ``OpSchema``) live under
``onnx_light/onnx_op/``, one ``operator_sets_<domain>.cc`` per domain. A schema
is a :cpp:class:`onnx_light::core::LightOpSchema` describing the inputs, outputs,
type constraints and documentation of one ``since_version`` of an operator.


1. **Build the schema(s)** — one function returns every historical version
   of an operator, in ``operator_sets_math.cc``:

   .. code-block:: cpp

       std::vector<LightOpSchema> BuildAbsSchemas() {
         std::vector<LightOpSchema> schemas;
         schemas.push_back(LightOpSchema(
             "Abs", kOnnxDomain, /*since_version=*/13, kAbsDocV13,
             {
                 {"X", "Input tensor", "T"},
             },
             {
                 {"Y", "Output tensor", "T"},
             },
             {
                 {"T", AllNumericTypesIr4(),
                  "Constrain input and output types to all numeric tensors."},
             }));
         // ... earlier revisions (v6, v1) ...
         return schemas;
       }

2. **Register the builder** by adding an entry to the ``builders`` map in
   ``GetAllOnnxOpMathSchemasWithHistory`` (same file), keyed by op type:

   .. code-block:: cpp

       {"Abs", [] { return BuildAbsSchemas(); }},

Each domain's ``GetAllOnnxOp<Domain>SchemasWithHistory`` is aggregated by
``GetAllOnnxOpSchemasWithHistory`` in
``onnx_light/onnx_op/operator_sets.cc``, so a new domain also needs to be
added there; a new op inside an existing domain only needs the
builder-map entry above.

Register a peak memory function
-------------------------------

A peak-memory function estimates the additional peak memory a kernel needs,
excluding its inputs and outputs. Operators without an estimator default to zero.
The registry mirrors shape inference. The signature is
``PeakMemoryEstimate(Device, const NodeProto &, const std::vector<SymShape> &)``.
Every registered estimator receives the node, including its attributes and
optional input names. Shape-only queries supply only its domain and operator type.
The result contains a mandatory ``SymDim worst_case`` and an optional
``SymDim average_case``. All values are bytes excluding inputs and outputs.


1. **Implement it** next to the shape function, e.g.
   ``shapes/nn/shape_attention.cc`` (declared in ``shapes/nn/shape_nn.h``):

   Preserve symbolic dimensions with the checked helpers in
   ``onnx_core/expressions/expressions.h`` and the conversions in
   ``onnx_core/symbolic/symbolic_helper.h``. Use
   ``PeakMemoryEstimate::Exact(bytes)`` only when the workspace requirement is
   deterministic; otherwise return ``{maximum, average}``, leaving the average
   ``std::nullopt`` when unjustified. A concrete ``int64_t`` return is also
   accepted as a concrete upper bound without an average.

   If required shapes or an estimation rule are unavailable, return
   ``core::shapes::UnknownPeakMemory(domain, op_type)``. Its mandatory
   maximum is a valid symbolic upper-bound identifier which must be resolved
   for every invocation and device before checking a memory budget.
   Do not invent a finite bound or substitute
   zero. See :cpp:func:`ComputePeakMemoryAttention` for symbolic score-buffer
   arithmetic and negative-dimension/overflow validation.

2. **Register it** by adding an entry to the ``BuiltinPeakMemoryFunctions``
   table in ``onnx_light/onnx_extensions/shapes/dispatch_table.cc``:

   .. code-block:: cpp

       {"ai.onnx:Attention",
        [](Device device, const NodeProto &node, const std::vector<SymShape> &inputs) {
          return nn::ComputePeakMemoryAttention(device, node, inputs);
        }},

As with shape functions, :cpp:func:`RegisterPeakMemoryFunctions` (same
file) copies these builtins into the shared ``core::shapes`` peak-memory
dispatch table via
:cpp:func:`onnx_light::core::shapes::RegisterComputePeakMemoryFn`.

Decoder peak-memory coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CPU registrations include bounds for the portable native decoder kernels.
They count additional heap buffers, not declared inputs/outputs,
inline shape/stride storage, allocator bookkeeping or shared thread-pool overhead:

* ``MatMul``: ``4 * (elements(A) + elements(B)) + 6 * elements(Y)``, including
  broadcast batches and vector operands. This covers the simultaneously live
  float32 promotions of half-precision inputs and output, plus the temporary
  half-precision result created before copying into a preallocated output.
  Because the estimator receives shapes rather than dtypes, this is also the
  conservative bound for dtype paths that need no promotion; no average is
  asserted.
* ``Gather``: exactly ``8 * elements(indices)`` for the INT64 index copy.
* ``Slice``: ``32 * length(starts)`` for four simultaneously live INT64 arrays,
  including synthesized axes and steps.
* ``RMSNormalization``: ``8 * (max(1, normalized_elements) + rank(scale) +
  normalized_rank) + 4 * (elements(X) + elements(scale))``. This conservatively
  sums the index/coordinate buffers and half-precision promotions even though
  some have disjoint lifetimes.
* ``Mul``, ``Sub``, ``GreaterOrEqual``, ``And``, ``Swish``, ``Where``, ``CumSum``,
  ``Shape``, ``Unsqueeze``, ``Transpose`` and ``Cast`` use the zero default;
  no per-operator registration is needed. Binary broadcast strides use inline
  ``Shape`` storage for every supported rank, including equal-shape and scalar
  fast paths, so they require no additional heap buffers.

Registered estimators can explicitly return unresolved bounds when required
shapes are missing or contain placeholder dimensions. Symbolic dimensions yield
expressions where peak memory depends on them; negative dimensions and concrete
overflow are rejected. Missing estimators, including for other devices or
domains, return zero instead.
Context-based queries distinguish missing named input descriptors from known
scalars; empty optional input slots keep their positions.

An optimized runtime such as ``onnx-light-cpu`` must register its own estimator
**after** calling ``RegisterPeakMemoryFunctions`` if its kernels allocate different
workspace (for example packed MatMul buffers). Registering the same domain,
operator and device replaces the portable estimate; calling the idempotent
builtin registration again does not overwrite that replacement. A runtime must
not reuse a portable bound for an implementation it does not cover. Unknown
phases must remain unknown in both per-node totals and the model peak.

See also
--------

* :ref:`l-howto-use-custom-kernel-cpp` - plug a kernel in at runtime without
  rebuilding the library.
* :ref:`l-howto-use-custom-shape-inference-cpp` - register a shape function from
  Python or C++ at runtime.
* :ref:`l-howto-collect-backend-test-cases-cpp` - enumerate and run backend test
  cases.
* :ref:`l-how-to` - other onnx-light how-to recipes.
