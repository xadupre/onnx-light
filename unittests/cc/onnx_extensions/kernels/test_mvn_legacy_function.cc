// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/defs/schema.h"
#include "onnx_lib/inliner/inliner.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

TEST(MeanVarianceNormalizationFunctionExecution, PreservesLegacyVarianceBeforeVersion29) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  const auto selected = inliner::FunctionIdSet::Create({{"", "MeanVarianceNormalization"}});
  for (const int version : {13, 18, 28, 29}) {
    for (const auto &values : std::vector<std::vector<float>>{{1, 3}, {10000, 10100}, {1, 1}}) {
      SCOPED_TRACE(version);
      SCOPED_TRACE(values[0]);
      ModelProto model;
      model.set_ir_version(IR_VERSION);
      model.add_opset("", version);
      auto *graph = model.mutable_graph();
      graph->set_name("mvn_legacy_function");
      auto *input = graph->add_input();
      input->set_name("X");
      auto *type = input->mutable_type()->mutable_tensor_type();
      type->set_elem_type(TensorProto::FLOAT);
      type->mutable_shape()->add_dim()->set_dim_value(2);
      auto *output = graph->add_output();
      output->set_name("Y");
      *output->mutable_type() = input->type();
      auto *node = graph->add_node();
      node->set_op_type("MeanVarianceNormalization");
      node->add_input("X");
      node->add_output("Y");
      auto *axes = node->add_attribute();
      axes->set_name("axes");
      axes->set_type(AttributeProto::INTS);
      axes->add_ints(0);
      inliner::InlineSelectedFunctions(model, *selected, OpSchemaRegistry::Instance());
      ASSERT_GT(graph->node_size(), 1);
      ASSERT_NO_THROW(checker::check_model(model, true));

      const auto outputs =
          core::runtime::RunModel(model, {core::runtime::Tensor::FromFloat("X", {2}, values)});
      ASSERT_EQ(outputs.size(), 1u);
      ASSERT_EQ(outputs[0].data_type, TensorProto::FLOAT);
      const float mean = (values[0] + values[1]) / 2;
      const float first_centered = values[0] - mean;
      const float second_centered = values[1] - mean;
      const float variance =
          version < 29 ? (values[0] * values[0] + values[1] * values[1]) / 2 - mean * mean
                       : (first_centered * first_centered + second_centered * second_centered) / 2;
      for (size_t i = 0; i < values.size(); ++i) {
        const float expected = (values[i] - mean) / (std::sqrt(variance) + 1e-9f);
        const float actual = outputs[0].AsFloat()[i];
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected, std::abs(expected) * 1e-6f + 1e-6f);
      }
    }
  }
}

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
