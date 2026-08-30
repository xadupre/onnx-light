// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

} // namespace

void RegisterFloorDivOffsetShapeInferenceCase(std::vector<TestCase> &registry) {
  const std::string name("test_cc_shape_inference_floordiv_offset_expression");
  const OpsetId opset = DefaultOpset(18);
  const onnx_kernels::kernel::KernelContext ctx{opset};
  const onnx_kernels::kernel::MaxPool maxpool_kernel{ctx};

  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);
  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});
  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  NodeProto &node = AddNode(*graph, "MaxPool", {"X"}, {"Y"});
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {7});
  AddAttribute<std::vector<int64_t>>(node, "pads", {6, 6});
  AddAttribute<std::vector<int64_t>>(node, "strides", {5});

  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"batch", "channel", "seq"});
  AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {"batch", "channel", "seq//5+2"});

  Tensor x = Tensor::FromFloat("X", {2, 1, 4},
                               {1.0f, 0.0f, 2.0f, 0.0f, //
                                0.0f, 3.0f, 0.0f, 4.0f});
  Tensor y = maxpool_kernel(x, /*kernel_shape=*/{7}, /*strides=*/{5}, /*pads=*/{6, 6});
  y.name = "Y";
  AppendDataSet(tc, {std::move(x)}, {std::move(y)});
  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
