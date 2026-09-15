// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 10;

// Domain used for the model-local function. Any non-default domain works;
// ``"local"`` mirrors the convention used by upstream ONNX tests and the
// onnx-light Python function tests (see
// ``unittests/python/bindings/test_onnx_function.py``).
constexpr const char *kLocalDomain = "local";
constexpr const char *kFuncAddName = "func_add";
constexpr const char *kFuncRangeName = "func_range";

} // namespace

// ---------------------------------------------------------------------------
// Single-call to a **model-local function** — exercises the shape-inference
// path for ``FunctionProto`` references declared in ``ModelProto::functions``.
// The function body is a single ``Add`` on two same-shape inputs, so the
// caller-visible output shape must propagate through the function expansion
// and end up equal to the input shape (``[batch, d_model]``).
//
//   X  (batch, d_model)         Y  (batch, d_model)
//        \\                          /
//         \\                        /
//        local:func_add(X, Y)  ──► Z  (batch, d_model)
//
// where ``local:func_add(a, b) -> c { c = Add(a, b) }`` is declared as a
// model-local function. ``onnx_shapes`` shape inference must register the
// :cpp:class:`FunctionProto` in :cpp:class:`ShapesContext` and recursively
// run shape inference on the function body with the function's input/output
// names rebound to the caller's names.
// ---------------------------------------------------------------------------
void RegisterLocalFunctionAddShapeInferenceCases(std::vector<TestCase> &registry,
                                                 TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_inference_local_function_add";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_1{opset};
    const onnx_kernels::kernel::Add kernel_1{ctx_1};
    const KernelContext ctx_2{opset};
    const onnx_kernels::kernel::Abs kernel_2{ctx_2};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset, OpsetId(kLocalDomain, static_cast<int64_t>(1))});

    // Declare the model-local function ``local:func_add`` whose body is a
    // single ``Add`` on two same-shape inputs.
    FunctionProto *func = model.add_functions();
    func->set_name(kFuncAddName);
    func->set_domain(kLocalDomain);
    func->add_input("a");
    func->add_input("b");
    func->add_output("c");
    OperatorSetIdProto *func_opset = func->add_opset_import();
    func_opset->set_domain("");
    func_opset->set_version(static_cast<int64_t>(opset.version));
    NodeProto *body_node = func->add_node();
    body_node->set_op_type("Add");
    body_node->add_input("a");
    body_node->add_input("b");
    body_node->add_output("c");

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // Single node in the main graph: a call into the local function. An ``Abs``
    // node is appended so the model's final output is computed by an Abs of the
    // local-function result, exercising shape propagation through one more
    // built-in op after the function expansion.
    AddNode(*graph, kFuncAddName, {"X", "Y"}, {"Z_pre_abs"}, kLocalDomain);
    AddNode(*graph, "Abs", {"Z_pre_abs"}, {"Z"});

    // Symbolic graph inputs / outputs. Equal symbolic dims across X, Y and Z
    // ensure that the symbolic-dim propagation pass in
    // ``BackendTestCaseShapeInference.AllCollectedCasesPropagateSymbolicDims``
    // also exercises shape propagation through the function expansion.
    const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
    const std::vector<DimSpec> sym_shape = {"batch", "d_model"};
    AppendValueInfo(*graph->add_input(), "X", kFloat, sym_shape);
    AppendValueInfo(*graph->add_input(), "Y", kFloat, sym_shape);
    AppendValueInfo(*graph->add_value_info(), "Z_pre_abs", kFloat, sym_shape);
    AppendValueInfo(*graph->add_output(), "Z", kFloat, sym_shape);

    // Reference DataSet with concrete (``batch=2, d_model=4``) tensors. The
    // function body is just ``Add``, so the runtime expected value is the
    // element-wise sum of the inputs — computed here with the ``Add`` kernel
    // directly (the kernel runtime does not expand local functions, but the
    // shape-inference tests in ``test_backend_shape_inference.cc`` only care
    // about the model topology and the recorded I/O shapes).
    const std::vector<int64_t> data_shape = {2, 4};
    const int64_t input_size = data_shape[0] * data_shape[1];
    std::vector<float> x_values(static_cast<size_t>(input_size));
    std::vector<float> y_values(static_cast<size_t>(input_size));
    for (size_t i = 0; i < x_values.size(); ++i) {
      x_values[i] = static_cast<float>(i) * 0.1f;
      y_values[i] = static_cast<float>(i) * 0.01f + 1.0f;
    }
    Tensor x = Tensor::FromFloat("X", data_shape, x_values);
    Tensor y = Tensor::FromFloat("Y", data_shape, y_values);
    Tensor z_pre_abs = kernel_1(x, y);
    Tensor z = kernel_2(z_pre_abs);
    z.name = "Z";

    AppendDataSet(tc, {x, y}, {z});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

// ---------------------------------------------------------------------------
// Single-call to a **model-local function whose body contains a ``Range``
// node** — exercises the shape-as-value propagation path inside
// ``ExpandLocalFunctionCall``.
//
// The function body produces a 1-D INT64 sequence by calling
// ``Range(start_c, lim, delta_c)`` where:
//   * ``start_c`` = 0 and ``delta_c`` = 1 are ``Constant`` nodes defined
//     inside the function body, so ``onnx_shapes`` shape inference assigns
//     them concrete ``ValueAsShape`` annotations right away.
//   * ``lim`` is the function's **input parameter**, bound at the call site
//     to the graph initializer ``limit_val: int64[] = 5``.
//
// When ``ExpandLocalFunctionCall`` copies the caller's ``SymTensor`` for
// ``limit_val`` (which carries ``ValueAsShape = [5]`` from initializer
// data-propagation) into the function sub-context as ``lim``, the
// ``ValueAsShape`` annotation must survive the copy so that
// ``ComputeShapeRange`` can resolve the output length to the concrete value
// 5 rather than emitting a generic symbolic dim.
//
// Graph topology:
//
//   Initializer: limit_val : int64[] = 5
//       ↓
//   local:func_range(limit_val) → r_out : int64[5]
//       ↓
//   Abs(r_out) → out : int64[5]
//
// Expected output: out = [0, 1, 2, 3, 4]
// ---------------------------------------------------------------------------
void RegisterLocalFunctionRangeShapeInferenceCases(std::vector<TestCase> &registry,
                                                   TestMode /*mode*/) {
  constexpr int64_t kRangeStart = 0;
  constexpr int64_t kRangeDelta = 1;
  constexpr int64_t kRangeLimit = 5;
  const OpsetId opset = DefaultOpset(18);
  const std::string name = "test_cc_shape_inference_local_function_range";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
  lazy_case.build = [name, kRangeStart, kRangeDelta](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(18);

    const KernelContext ctx_3{opset};
    const onnx_kernels::kernel::Range kernel_3{ctx_3};
    const KernelContext ctx_4{opset};
    const onnx_kernels::kernel::Abs kernel_4{ctx_4};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion,
              {opset, OpsetId(std::string(kLocalDomain), static_cast<int64_t>(1))});

    // Declare the model-local function ``local:func_range(lim) -> r``.
    // Body: start_c = Constant(0), delta_c = Constant(1), r = Range(start_c, lim, delta_c).
    FunctionProto *func = model.add_functions();
    func->set_name(kFuncRangeName);
    func->set_domain(kLocalDomain);
    func->add_input("lim");
    func->add_output("r");
    OperatorSetIdProto *func_opset = func->add_opset_import();
    func_opset->set_domain("");
    func_opset->set_version(static_cast<int64_t>(opset.version));

    {
      NodeProto *start_node = func->add_node();
      start_node->set_op_type("Constant");
      start_node->add_output("start_c");
      AddAttribute<int64_t>(*start_node, "value_int", kRangeStart);
    }
    {
      NodeProto *delta_node = func->add_node();
      delta_node->set_op_type("Constant");
      delta_node->add_output("delta_c");
      AddAttribute<int64_t>(*delta_node, "value_int", kRangeDelta);
    }
    {
      NodeProto *range_node = func->add_node();
      range_node->set_op_type("Range");
      range_node->add_input("start_c");
      range_node->add_input("lim");
      range_node->add_input("delta_c");
      range_node->add_output("r");
    }

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // Graph initializer: ``limit_val : int64[] = 5``.
    // ``onnx_shapes`` seeds this with ``ValueAsShape = [5]``.  When the local
    // function is called, ``ExpandLocalFunctionCall`` copies the full
    // ``SymTensor`` (including ``ValueAsShape``) to the sub-context as
    // ``lim``, allowing ``ComputeShapeRange`` to resolve the output length.
    AddInitializer<int64_t>(*graph, "limit_val", {}, {kRangeLimit});

    // Main graph nodes: call the local function then take Abs.
    AddNode(*graph, kFuncRangeName, {"limit_val"}, {"r_out"}, kLocalDomain);
    AddNode(*graph, "Abs", {"r_out"}, {"out"});

    // Intermediate and output value_info with concrete shapes.
    const int32_t kInt64 = static_cast<int32_t>(DataType::INT64);
    AppendValueInfo(*graph->add_value_info(), "r_out", kInt64, {DimSpec(kRangeLimit)});
    AppendValueInfo(*graph->add_output(), "out", kInt64, {DimSpec(kRangeLimit)});

    // Reference DataSet: no graph inputs; output = Abs(Range(0, 5, 1)) = [0,1,2,3,4].
    Tensor limit_t = Tensor::FromInt64("", {}, {kRangeLimit});
    Tensor zero_t = Tensor::FromInt64("", {}, {kRangeStart});
    Tensor one_t = Tensor::FromInt64("", {}, {kRangeDelta});
    Tensor r_t = kernel_3(zero_t, limit_t, one_t);
    Tensor out_t = kernel_4(r_t);
    out_t.name = "out";

    AppendDataSet(tc, {}, {out_t});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
