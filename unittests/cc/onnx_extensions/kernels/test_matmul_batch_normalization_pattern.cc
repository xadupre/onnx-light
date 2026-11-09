// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(MatMulBatchNormalizationFusionPattern, NumericallyPreservesAlignedAndRefusedChains) {
  onnx_kernels::RegisterKernelFunctions();
  const auto schemas = [](const std::string &op) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
  };
  for (int layout = 0; layout < 3; ++layout) {
    SCOPED_TRACE(layout);
    core::builder::GraphBuilder builder("matmul_bn", schemas, {{"", 28}});
    core::symbolic::SymShape input_shape;
    input_shape.PushBack(core::symbolic::SymDim(2));
    input_shape.PushBack(core::symbolic::SymDim(3));
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, input_shape);
    const auto constant = [&](const char *name, const std::vector<int64_t> &dims,
                              const std::vector<float> &values) {
      builder.MakeInitializer(MakeInitializer<float>(name, dims, values));
    };
    constant("w", {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    constant("scale", {4}, {-1, 2, 0, 0.5F});
    constant("bias", {4}, {2, -1, 3, 0.25F});
    constant("mean", {4}, {0.5F, -2, 1, 0});
    constant("variance", {4}, {1, 4, 1, 0.25F});
    builder.MakeNode("MatMul", {"x", "w"}, {"mm"});
    builder.MakeInitializer(
        MakeInitializerShape("target", layout == 0   ? std::vector<int64_t>{2, 4, 1}
                                       : layout == 1 ? std::vector<int64_t>{2, 1, 4}
                                                     : std::vector<int64_t>{1, 4, 2, 1}));
    builder.MakeNode("Reshape", {"mm", "target"}, {"reshaped"});
    std::string input = "reshaped";
    if (layout == 1) {
      NodeProto transpose = MakeNode("Transpose", {"reshaped"}, {"transposed"});
      AddAttribute<std::vector<int64_t>>(transpose, "perm", {0, 2, 1});
      builder.MakeNode("Transpose", {"reshaped"}, {"transposed"}, "", "", transpose.attribute());
      input = "transposed";
    }
    NodeProto batch =
        MakeNode("BatchNormalization", {input, "scale", "bias", "mean", "variance"}, {"out"});
    AddAttribute<float>(batch, "epsilon", 0);
    builder.MakeNode("BatchNormalization", {input, "scale", "bias", "mean", "variance"}, {"out"},
                     "", "", batch.attribute());
    builder.MakeOutput("out");
    const auto tensor = core::runtime::Tensor::FromFloat("x", {2, 3}, {-2, 0, 1, 3, -4, 0.5F});
    const auto expected = core::runtime::RunModel(builder.ToModel(), {tensor});
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(onnx_patterns::CreatePattern("MatMulBatchNormalizationFusion"));
    core::builder::GraphGraph graph(builder, std::move(patterns));
    core::builder::ConstantFoldingOptions folding;
    folding.enabled = false;
    const auto plan = graph.Optimize(4, nullptr, folding);
    EXPECT_EQ(plan.rewrites().empty(), layout == 2);
    EXPECT_EQ(builder.Nodes()[0].op_type(), layout == 2 ? "MatMul" : "Gemm");
    EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 1].output()[0], "out");
    const auto actual = core::runtime::RunModel(builder.ToModel(), {tensor});
    ASSERT_EQ(expected.size(), 1u);
    ASSERT_EQ(actual.size(), 1u);
    EXPECT_EQ(actual[0].shape, expected[0].shape);
    EXPECT_EQ(actual[0].data_type, expected[0].data_type);
    ASSERT_EQ(actual[0].element_count(), 8);
    for (std::size_t index = 0; index < 8; ++index) {
      EXPECT_NEAR(actual[0].AsFloat()[index], expected[0].AsFloat()[index], 1e-5F);
    }
  }
}
