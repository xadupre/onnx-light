// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(GemmOutputTransposePattern, NumericallyPreservesNonSquareMatricesAndAllFlags) {
  onnx_kernels::RegisterKernelFunctions();
  const auto schemas = [](const std::string &op) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
  };
  const std::vector<float> left{-2, 0.5F, 3, 1, -1, 2};
  const std::vector<float> right{-1, 2, -3, 4, 0.5F, 1.5F, -2.5F, 3.5F, 2, -1, 0, 1};
  for (int flags = -1; flags < 4; ++flags) {
    for (bool constant : {false, true}) {
      for (bool input_transposes : {false, true}) {
        SCOPED_TRACE(std::to_string(flags) + ":" + std::to_string(constant) + ":" +
                     std::to_string(input_transposes));
        const int effective_flags = flags < 0 ? 0 : flags;
        std::vector<int64_t> left_shape =
            effective_flags & 1 ? std::vector<int64_t>{3, 2} : std::vector<int64_t>{2, 3};
        std::vector<int64_t> right_shape =
            effective_flags & 2 ? std::vector<int64_t>{4, 3} : std::vector<int64_t>{3, 4};
        if (input_transposes) {
          std::reverse(left_shape.begin(), left_shape.end());
          std::reverse(right_shape.begin(), right_shape.end());
        }
        core::builder::GraphBuilder builder("g", schemas, {{"", 13}});
        std::vector<core::runtime::Tensor> feeds;
        for (bool is_left : {true, false}) {
          const std::string name = is_left ? "x" : "w";
          const auto &dims = is_left ? left_shape : right_shape;
          const auto &values = is_left ? left : right;
          if (constant) {
            builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dims, values));
          } else {
            core::symbolic::SymShape shape;
            for (int64_t dim : dims)
              shape.PushBack(core::symbolic::SymDim(dim));
            builder.MakeInput(name, core::symbolic::TensorType::kFloat, shape);
            feeds.push_back(core::runtime::Tensor::FromFloat(name, dims, values));
          }
          if (input_transposes)
            builder.MakeNode("Transpose", {name}, {name + "t"});
        }
        NodeProto gemm = MakeNode(
            "Gemm", {input_transposes ? "xt" : "x", input_transposes ? "wt" : "w"}, {"gemm"});
        if (flags >= 0) {
          AddAttribute<int64_t>(gemm, "transA", flags & 1);
          AddAttribute<int64_t>(gemm, "transB", (flags >> 1) & 1);
          AddAttribute<float>(gemm, "alpha", 1.75F);
          AddAttribute<float>(gemm, "beta", 0.25F);
        }
        builder.MakeNode("Gemm", {gemm.input(0).value(), gemm.input(1).value()}, {"gemm"}, "", "",
                         gemm.attribute());
        NodeProto transpose = MakeNode("Transpose", {"gemm"}, {"out"});
        if (flags >= 0)
          AddAttribute<std::vector<int64_t>>(transpose, "perm", {1, 0});
        builder.MakeNode("Transpose", {"gemm"}, {"out"}, "", "", transpose.attribute());
        builder.MakeOutput("out");
        const auto expected = core::runtime::RunModel(builder.ToModel(), feeds);
        ASSERT_EQ(expected.size(), 1u);
        std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
        for (const char *name : {"GemmOutputTranspose", "GemmTranspose", "TransposeMatMul"})
          patterns.push_back(onnx_patterns::CreatePattern(name));
        core::builder::GraphGraph graph(builder, std::move(patterns));
        core::builder::ConstantFoldingOptions folding;
        folding.enabled = false;
        ASSERT_FALSE(graph.Optimize(6, nullptr, folding).rewrites().empty());
        ASSERT_EQ(builder.Nodes().size(), 1u);
        const auto actual = core::runtime::RunModel(builder.ToModel(), feeds);
        ASSERT_EQ(actual.size(), 1u);
        EXPECT_EQ(actual[0].shape, expected[0].shape);
        ASSERT_EQ(actual[0].element_count(), 8);
        for (int64_t index = 0; index < 8; ++index)
          EXPECT_NEAR(actual[0].AsFloat()[index], expected[0].AsFloat()[index], 1e-5F);
        EXPECT_TRUE(graph.Optimize(6, nullptr, folding).rewrites().empty());
      }
    }
  }
}
