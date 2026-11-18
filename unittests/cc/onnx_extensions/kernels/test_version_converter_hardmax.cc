// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/version_converter/convert.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

TEST(HardmaxConverterExecution, PreservesLegacyFlattenFirstOutputs) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  struct Case {
    std::optional<int64_t> axis;
    core::runtime::Shape shape;
    std::vector<float> expected;
    bool flatten;
  };
  const std::vector<Case> cases{
      {std::nullopt, {1, 2, 2}, {0, 0, 0, 1}, true},
      {1, {1, 2, 2}, {0, 0, 0, 1}, true},
      {0, {2, 2}, {0, 0, 0, 1}, true},
      {-2, {1, 2, 2}, {0, 0, 0, 1}, true},
      {2, {1, 2, 2}, {0, 1, 0, 1}, false},
      {-1, {1, 2, 2}, {0, 1, 0, 1}, false},
  };
  for (size_t index = 0; index < cases.size(); ++index) {
    SCOPED_TRACE(index);
    const auto &test = cases[index];
    ModelProto model;
    model.set_ir_version(IR_VERSION);
    model.add_opset("", 12);
    auto *graph = model.mutable_graph();
    graph->set_name("legacy_hardmax");
    auto *input = graph->add_input();
    input->set_name("X");
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT);
    for (const int64_t dim : test.shape)
      type->mutable_shape()->add_dim()->set_dim_value(dim);
    auto *output = graph->add_output();
    output->set_name("Y");
    output->mutable_type()->CopyFrom(input->type());
    auto *node = graph->add_node();
    node->set_op_type("Hardmax");
    node->add_input("X");
    node->add_output("Y");
    if (test.axis) {
      auto *axis = node->add_attribute();
      axis->set_name("axis");
      axis->set_type(AttributeProto::INT);
      axis->set_i(*test.axis);
    }
    ASSERT_NO_THROW(checker::check_model(model, true));
    const auto converted = version_conversion::ConvertVersion(model, 13);
    ASSERT_NO_THROW(checker::check_model(converted, true));
    ASSERT_EQ(converted.opset_import(0).version(), 13);
    ASSERT_EQ(converted.graph().node_size(), test.flatten ? 4 : 1);
    const auto &hardmax = converted.graph().node(test.flatten ? 2 : 0);
    EXPECT_EQ(hardmax.op_type(), "Hardmax");
    ASSERT_EQ(hardmax.attribute_size(), 1);
    EXPECT_EQ(hardmax.attribute(0).name(), "axis");
    EXPECT_EQ(hardmax.attribute(0).i(), -1);
    const auto outputs = core::runtime::RunModel(
        converted, {core::runtime::Tensor::FromFloat("X", test.shape, {1, 2, 3, 4})});
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].shape, test.shape);
    EXPECT_EQ(outputs[0].data_type, TensorProto::FLOAT);
    for (size_t i = 0; i < test.expected.size(); ++i)
      EXPECT_EQ(outputs[0].AsFloat()[i], test.expected[i]);
  }
}

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
