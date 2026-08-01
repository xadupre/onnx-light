.. _l-howto-register-builtins:

:html_theme.sidebar_secondary.remove:

How to register a built-in kernel, test case, shape inference, light op or peak memory function
===============================================================================================

:ref:`l-howto-use-custom-kernel` and :ref:`l-howto-use-custom-shape-inference`
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
* an optional **peak memory** function – estimates the scratch memory the
  kernel needs, also in ``lib_onnx_shape``.

Each piece uses ``Abs`` (an element-wise unary math op) as the running example.
All the source directories are compiled with CMake ``GLOB_RECURSE`` patterns, so
adding a new ``.cc`` file is picked up automatically — no ``CMakeLists.txt`` edit
is required. Only the registration tables listed below need a new entry.

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

2. **Implement it** in ``kernels/math/kernel_abs.cc``. ``Run`` reads the node's
   inputs from the :cpp:class:`RuntimeContext` and writes the outputs back:

   .. code-block:: cpp

       void Abs::Run(RuntimeContext &rt) {
         const NodeProto &node = *node_;
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         SetOutput(node, 0, (*this)(x, &rt), rt);
       }

3. **Register it** by adding one entry to the ``BuiltinKernelFunctions`` table in
   ``onnx_light/onnx_extensions/kernels/kernel_dispatch_table.cc``. The key is
   ``"<domain>:<op_type>"`` and ``MakeKernel<T>()`` builds the dispatch factory:

   .. code-block:: cpp

       {"ai.onnx:Abs", MakeKernel<onnx_kernels::kernel::Abs>()},

The empty default ONNX domain is normalised to ``ai.onnx``. The registration
happens in ``lib_onnx_kernels``; the runtime consults this table through
:cpp:func:`onnx_light::core::runtime::RunNode`.

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

2. **Declare** ``RegisterAbsCases`` in ``cases/math/include_math_cases.h`` with
   the visibility macro:

   .. code-block:: cpp

       ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAbsCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

3. **Register** it by adding one entry to the ``OpRegisterModeMap`` in
   ``cases/math/collect_math_cases.cc``:

   .. code-block:: cpp

       {"Abs", &RegisterAbsCases},

The per-category collectors are wired into the global registry (via
:cpp:func:`RegisterTestCasesCollector`) in
``onnx_light/onnx_extensions/backend_test/collect_test_cases.cc``, so the new
case is reachable from :cpp:func:`onnx_light::core::backend_test::CollectTestCases`
and its Python binding. See :ref:`l-howto-collect-backend-test-cases` for how to
enumerate and run cases.

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

2. **Register it** by adding an entry to the ``BuiltinShapeFunctions`` table in
   ``onnx_light/onnx_extensions/shapes/dispatch_table.cc``. The lambda checks the
   input count and forwards to the shape function:

   .. code-block:: cpp

       {"ai.onnx:Abs",
        [](ShapesContext &ctx, const NodeProto &node) {
          RequireInputs(node, 1);
          math::ComputeShapeAbs(ctx, node, node.input(0).c_str());
        }},

``lib_onnx_shape`` is a static archive, so registration is not automatic on
link: :cpp:func:`RegisterShapeFunctions` (in the same file) copies every builtin
entry into the shared ``core::shapes`` dispatch table via
:cpp:func:`onnx_light::core::shapes::RegisterComputeShapeFn`. It is idempotent and
must be called before running shape inference.

Register a light op
-------------------

Operator schemas (the "light" equivalent of ONNX ``OpSchema``) live under
``onnx_light/onnx_op/``, one ``operator_sets_<domain>.cc`` per domain. A schema
is a :cpp:class:`onnx_light::core::LightOpSchema` describing the inputs, outputs,
type constraints and documentation of one ``since_version`` of an operator.

1. **Build the schema(s)** — one function returns every historical version of an
   operator, in ``operator_sets_math.cc``:

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
``GetAllOnnxOpSchemasWithHistory`` in ``onnx_light/onnx_op/operator_sets.cc``, so
a new domain also needs to be added there; a new op inside an existing domain
only needs the builder-map entry above.

Register a peak memory function
-------------------------------

A peak-memory function estimates the scratch (non-output) memory a kernel needs,
so a scheduler can size buffers. It is optional — operators without one report a
peak of ``0``. The registry mirrors shape inference. The signature is
``int64_t(Device, const std::vector<SymShape> &)``.

1. **Implement it** next to the shape function, e.g.
   ``shapes/nn/shape_attention.cc`` (declared in ``shapes/nn/shape_nn.h``):

   .. code-block:: cpp

       int64_t ComputePeakMemoryAttention(Device device, const std::vector<SymShape> &input_shapes) {
         constexpr int64_t kScoreElementBytes = 4;  // scores held in float32
         (void)device;
         if (input_shapes.size() < 2)
           return 0;
         const SymShape &q_shape = input_shapes[0];
         const SymShape &k_shape = input_shapes[1];
         if (q_shape.Rank() != 4 || k_shape.Rank() != 4)
           return 0;
         const SymDim &batch = q_shape[0];
         const SymDim &q_num_heads = q_shape[1];
         const SymDim &q_seq_len = q_shape[2];
         const SymDim &kv_seq_len = k_shape[2];
         if (!batch.IsInt() || !q_num_heads.IsInt() || !q_seq_len.IsInt() || !kv_seq_len.IsInt())
           return 0;  // no concrete estimate for symbolic dimensions
         return batch.AsInt() * q_num_heads.AsInt() * q_seq_len.AsInt() * kv_seq_len.AsInt() *
                kScoreElementBytes;
       }

2. **Register it** by adding an entry to the ``BuiltinPeakMemoryFunctions`` table
   in ``onnx_light/onnx_extensions/shapes/dispatch_table.cc``:

   .. code-block:: cpp

       {"ai.onnx:Attention", nn::ComputePeakMemoryAttention},

As with shape functions, :cpp:func:`RegisterPeakMemoryFunctions` (same file)
copies these builtins into the shared ``core::shapes`` peak-memory dispatch table
via :cpp:func:`onnx_light::core::shapes::RegisterComputePeakMemoryFn`, and the
result is exposed in Python through ``compute_peak_memory``.

From Python
-----------

There is **no separate Python registration step**. Every piece above lives in a
C++ static table, and the Python extension exposes generic entry points that
consult those same tables at call time. Once you have rebuilt the C++ libraries
(``pip install -e .`` or the CMake build), the new operator is reachable from
Python automatically — each component below simply works with the new
``op_type``:

* **Kernel** — run a model containing the operator with
  :class:`onnx_light.reference.ReferenceEvaluator`, which drives the C++
  ``RuntimeSession`` and resolves kernels through ``BuiltinKernelFunctions``:

  .. code-block:: python

      import numpy as np
      from onnx_light.onnx_lib import parser
      from onnx_light.reference import ReferenceEvaluator

      model = parser.parse_model(
          '<ir_version: 10, opset_import: ["" : 18]>'
          'agraph (float[3] x) => (float[3] y) { y = Abs(x) }'
      )
      sess = ReferenceEvaluator(model)
      (y,) = sess.run(None, {"x": np.array([-1.0, 2.0, -3.5], dtype=np.float32)})

* **Backend test case** — enumerate the C++ cases with
  :func:`onnx_light.onnx.backend.collect_test_cases`; passing the op type
  returns just its cases (see :ref:`l-howto-collect-backend-test-cases`):

  .. code-block:: python

      import onnx_light.onnx.backend as bt

      abs_cases = bt.collect_test_cases("Abs")

* **Shape inference** — run the pipeline with
  :func:`onnx_light.onnx_core.shape_inference.infer_shapes_model`, which reads
  the shared ``core::shapes`` dispatch table populated by
  ``RegisterShapeFunctions``:

  .. code-block:: python

      from onnx_light.onnx_core.shape_inference import infer_shapes_model

      infer_shapes_model(model)  # fills model.graph.value_info / output types

* **Light op schema** — the new schema shows up in
  :mod:`onnx_light.onnx.defs` (``get_schema``, ``get_all_schemas_with_history``)
  and in ``onnx_light.onnx_op.get_all_schemas_with_history``:

  .. code-block:: python

      import onnx_light.onnx.defs as defs

      schema = defs.get_schema("Abs")

* **Peak memory** — query the estimate through
  :func:`onnx_light.onnx_core.shape_inference.compute_peak_memory`, backed by
  ``RegisterPeakMemoryFunctions``:

  .. code-block:: python

      from onnx_light.onnx_core.shape_inference import compute_peak_memory, Device, SymShape

      peak = compute_peak_memory(
          "ai.onnx", "Attention", Device.kCPU,
          [SymShape([2, 4, 8, 16]), SymShape([2, 4, 8, 16])],
      )

See also
--------

* :ref:`l-howto-use-custom-kernel` - plug a kernel in at runtime without
  rebuilding the library.
* :ref:`l-howto-use-custom-shape-inference` - register a shape function from
  Python or C++ at runtime.
* :ref:`l-howto-collect-backend-test-cases` - enumerate and run backend test
  cases.
* :ref:`l-how-to` - other onnx-light how-to recipes.
