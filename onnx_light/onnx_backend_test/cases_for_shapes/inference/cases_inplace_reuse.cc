// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

} // namespace

void RegisterInPlaceReuseShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::KernelContext ctx{opset};
  const kernel::Abs abs_kernel{ctx};

  const std::string name = "test_cc_shape_inference_inplace_reuse";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Abs", {"X"}, {"A"});
  AddNode(*graph, "Abs", {"A"}, {"B"});
  AddNode(*graph, "Abs", {"B"}, {"Y"});

  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  const std::vector<DimSpec> shape = {"batch", "features"};
  AppendValueInfo(*graph->add_input(), "X", kFloat, shape);
  AppendValueInfo(*graph->add_value_info(), "A", kFloat, shape);
  AppendValueInfo(*graph->add_value_info(), "B", kFloat, shape);
  AppendValueInfo(*graph->add_output(), "Y", kFloat, shape);

  const Tensor x = Tensor::FromFloat(
      "X", {3, 4}, {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, -5.0f, -6.0f, 7.0f, 8.0f});
  Tensor y = abs_kernel(abs_kernel(abs_kernel(x)));
  y.name = "Y";

  AppendDataSet(tc, {x}, {y});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
