// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Shape → Concat → Add → Sub → Expand → 3 × Add → Add → Add`` —
// mirrors the ``test_value_as_shape`` example from yet-another-onnx-builder
// (https://github.com/xadupre/yet-another-onnx-builder/blob/main/
// unittests/xshape/test_value_as_shape.py#L11). The model is deliberately
// crafted to exercise *value-as-shape* propagation: the ``Expand`` consumer
// receives a ``shape2`` tensor whose contents (``[N, 1]``) are not literal
// initializers but are recovered from the symbolic shape of ``x`` through
// ``Shape``/``Concat``/``Add``/``Sub`` arithmetic. Shape inference must
// therefore propagate the value-as-shape annotation through every
// intermediate INT64 tensor so that the downstream ``Expand`` outputs the
// precise symbolic shape ``(N, 1)``.
//
//   x : float[N, 1], y1/y2/y3 : float[1, B], initializer one : int64[1]={1}
//   n        = Shape(x, start=0, end=1)        # int64[1] = [N]
//   b        = Shape(x, start=1, end=2)        # int64[1] = [1]
//   shape    = Concat([n, b], axis=0)          # int64[2] = [N, 1]
//   shape1   = Add(shape, one)                 # int64[2] = [N+1, 2]
//   shape2   = Sub(shape1, one)                # int64[2] = [N, 1]
//   expanded = Expand(x, shape2)               # float[N, 1]
//   z1/z2/z3 = Add(expanded, y{1,2,3})         # float[N, B]
//   z12      = Add(z1, z2)                     # float[N, B]
//   z        = Add(z12, z3)                    # float[N, B]
//
// The reference DataSet uses concrete sizes (``N=3, B=4``) so the case is
// executable end-to-end by ``BackendTestCaseRunModel``.
// ---------------------------------------------------------------------------
void RegisterValueAsShapeShapeInferenceCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(20);
  const onnx_kernels::kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_value_as_shape";

  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // n = Shape(x, start=0, end=1)
  NodeProto &n_node = AddNode(*graph, "Shape", {"x"}, {"n"});
  AddAttribute<int64_t>(n_node, "start", 0);
  AddAttribute<int64_t>(n_node, "end", 1);

  // b = Shape(x, start=1, end=2)
  NodeProto &b_node = AddNode(*graph, "Shape", {"x"}, {"b"});
  AddAttribute<int64_t>(b_node, "start", 1);
  AddAttribute<int64_t>(b_node, "end", 2);

  // shape = Concat([n, b], axis=0)
  NodeProto &concat_node = AddNode(*graph, "Concat", {"n", "b"}, {"shape"});
  AddAxisAttribute(concat_node, 0);

  AddNode(*graph, "Add", {"shape", "one"}, {"shape1"});
  AddNode(*graph, "Sub", {"shape1", "one"}, {"shape2"});
  AddNode(*graph, "Expand", {"x", "shape2"}, {"expanded"});

  AddNode(*graph, "Add", {"expanded", "y1"}, {"z1"});
  AddNode(*graph, "Add", {"expanded", "y2"}, {"z2"});
  AddNode(*graph, "Add", {"expanded", "y3"}, {"z3"});

  AddNode(*graph, "Add", {"z1", "z2"}, {"z12"});
  AddNode(*graph, "Add", {"z12", "z3"}, {"z_pre_abs"});
  AddNode(*graph, "Abs", {"z_pre_abs"}, {"z"});

  // Initializer ``one`` : int64[1] = [1].
  AddInitializer<int64_t>(*graph, "one", {1}, {1});

  // Graph inputs use the symbolic shapes from the Python test:
  //   x : float[N, 1], y1/y2/y3 : float[1, B].
  AppendValueInfo(*graph->add_input(), "x", DataType::FLOAT, {"N", DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_input(), "y1", DataType::FLOAT, {DimSpec(int64_t{1}), "B"});
  AppendValueInfo(*graph->add_input(), "y2", DataType::FLOAT, {DimSpec(int64_t{1}), "B"});
  AppendValueInfo(*graph->add_input(), "y3", DataType::FLOAT, {DimSpec(int64_t{1}), "B"});

  // Explicit intermediate value_info annotations mirror the ``value_info``
  // list in the Python test. They are stripped by
  // :cpp:func:`SnapshotAndStripValueInfo` in the
  // ``AllCollectedCasesInferOutputShapes`` test and used as the ground
  // truth that shape inference must recover.
  // Shape-computation intermediates (INT64): n/b/shape/shape1/shape2 flow
  // into Expand as value-as-shape tensors and must be listed in value_info
  // so that shape inference does not introduce previously-unknown names.
  AppendValueInfo(*graph->add_value_info(), "n", DataType::INT64, {DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_value_info(), "b", DataType::INT64, {DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_value_info(), "shape", DataType::INT64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "shape1", DataType::INT64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "shape2", DataType::INT64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "expanded", DataType::FLOAT,
                  {"N", DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_value_info(), "z1", DataType::FLOAT, {"N", "B"});
  AppendValueInfo(*graph->add_value_info(), "z2", DataType::FLOAT, {"N", "B"});
  AppendValueInfo(*graph->add_value_info(), "z12", DataType::FLOAT, {"N", "B"});
  AppendValueInfo(*graph->add_value_info(), "z3", DataType::FLOAT, {"N", "B"});
  AppendValueInfo(*graph->add_value_info(), "z_pre_abs", DataType::FLOAT, {"N", "B"});

  // Graph output: z : float[N, B].
  AppendValueInfo(*graph->add_output(), "z", DataType::FLOAT, {"N", "B"});

  // Build the reference DataSet — concrete N=3, B=4 tensors and the kernels
  // chained to materialise the expected ``z`` output.
  constexpr int64_t kN = 3;
  constexpr int64_t kB = 4;
  std::vector<float> x_values(static_cast<size_t>(kN));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f + 1.0f;
  }
  Tensor x = Tensor::FromFloat("x", {kN, 1}, x_values);

  std::vector<float> y1_values(static_cast<size_t>(kB));
  std::vector<float> y2_values(static_cast<size_t>(kB));
  std::vector<float> y3_values(static_cast<size_t>(kB));
  for (size_t i = 0; i < y1_values.size(); ++i) {
    y1_values[i] = static_cast<float>(i) * 0.01f + 2.0f;
    y2_values[i] = static_cast<float>(i) * 0.02f + 3.0f;
    y3_values[i] = static_cast<float>(i) * 0.03f + 4.0f;
  }
  Tensor y1 = Tensor::FromFloat("y1", {1, kB}, y1_values);
  Tensor y2 = Tensor::FromFloat("y2", {1, kB}, y2_values);
  Tensor y3 = Tensor::FromFloat("y3", {1, kB}, y3_values);

  // shape2 evaluates to [N, 1] = [3, 1].
  const Tensor shape2 = Tensor::FromInt64("", {2}, {kN, 1});
  Tensor expanded = onnx_kernels::kernel::Expand(ctx)(x, shape2);
  Tensor z1 = onnx_kernels::kernel::Add(ctx)(expanded, y1);
  Tensor z2 = onnx_kernels::kernel::Add(ctx)(expanded, y2);
  Tensor z3 = onnx_kernels::kernel::Add(ctx)(expanded, y3);
  Tensor z12 = onnx_kernels::kernel::Add(ctx)(z1, z2);
  Tensor z_pre_abs = onnx_kernels::kernel::Add(ctx)(z12, z3);
  Tensor z = onnx_kernels::kernel::Abs(ctx)(z_pre_abs);
  z.name = "z";

  AppendDataSet(tc, {std::move(x), std::move(y1), std::move(y2), std::move(y3)}, {std::move(z)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
