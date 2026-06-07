// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 10;

// Domain used for the model-local function. Any non-default domain works;
// ``"local"`` mirrors the convention used by upstream ONNX tests and the
// onnx-light Python function tests (see
// ``unittests/python/bindings/test_onnx_function.py``).
constexpr const char *kLocalDomain = "local";
constexpr const char *kFuncAddName = "func_add";

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
// model-local function. ``onnx_optim`` shape inference must register the
// :cpp:class:`FunctionProto` in :cpp:class:`ShapesContext` and recursively
// run shape inference on the function body with the function's input/output
// names rebound to the caller's names.
// ---------------------------------------------------------------------------
void RegisterLocalFunctionAddShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext kctx{opset};

  const std::string name = "test_cc_shape_inference_local_function_add";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion,
            {opset, OpsetId(std::string(kLocalDomain), static_cast<int64_t>(1))});

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

  // Single node in the main graph: a call into the local function.
  AddNode(*graph, kFuncAddName, {"X", "Y"}, {"Z"}, kLocalDomain);

  // Symbolic graph inputs / outputs. Equal symbolic dims across X, Y and Z
  // ensure that the symbolic-dim propagation pass in
  // ``BackendTestCaseShapeInference.AllCollectedCasesPropagateSymbolicDims``
  // also exercises shape propagation through the function expansion.
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  const std::vector<DimSpec> sym_shape = {"batch", "d_model"};
  AppendValueInfo(*graph->add_input(), "X", kFloat, sym_shape);
  AppendValueInfo(*graph->add_input(), "Y", kFloat, sym_shape);
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
  Tensor z = kernel::Add(kctx)(x, y);
  z.name = "Z";

  AppendDataSet(tc, {x, y}, {z});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
