// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/patterns/matmul/matmul_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

core::builder::GraphBuilder::SchemaLookupFn MatMulAddSchemas() {
  return [](const std::string &op) { return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false); };
}

core::symbolic::SymShape MatMulAddShape(const std::vector<int64_t> &dimensions) {
  core::symbolic::SymShape shape;
  for (int64_t dimension : dimensions) {
    shape.PushBack(core::symbolic::SymDim(dimension));
  }
  return shape;
}

core::runtime::Tensor MatMulAddInput(const std::string &name,
                                     const std::vector<int64_t> &dimensions) {
  std::size_t count = 1;
  for (int64_t dimension : dimensions) {
    count *= static_cast<std::size_t>(dimension);
  }
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 8;
  }
  return core::runtime::Tensor::FromFloat(name, dimensions, values);
}

void ExpectMatMulAddOutputs(const core::runtime::Tensors &actual,
                            const core::runtime::Tensors &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t output = 0; output < actual.size(); ++output) {
    EXPECT_EQ(actual[output].name, expected[output].name);
    EXPECT_EQ(actual[output].shape, expected[output].shape);
    EXPECT_EQ(actual[output].data_type, expected[output].data_type);
    ASSERT_EQ(actual[output].element_count(), expected[output].element_count());
    for (int64_t index = 0; index < actual[output].element_count(); ++index) {
      EXPECT_NEAR(actual[output].AsFloat()[index], expected[output].AsFloat()[index], 1e-5F);
    }
  }
}

bool OptimizeMatMulAdd(core::builder::GraphBuilder &builder, bool opt_in) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  if (opt_in) {
    patterns.push_back(std::make_unique<onnx_patterns::MatMulAddPattern>(2, true));
  } else {
    patterns.push_back(onnx_patterns::CreatePattern("MatMulAdd"));
  }
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::ConstantFoldingOptions folding;
  folding.enabled = false;
  return !graph.Optimize(4, nullptr, folding).rewrites().empty();
}

} // namespace

TEST(MatMulAddPattern, NumericallyPreservesOptInHigherRanksAndBiasOrders) {
  onnx_kernels::RegisterKernelFunctions();
  for (const auto &dimensions : {std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{2, 2, 3, 4}}) {
    for (bool full_bias : {false, true}) {
      for (bool swapped : {false, true}) {
        SCOPED_TRACE(dimensions.size());
        SCOPED_TRACE(full_bias);
        SCOPED_TRACE(swapped);
        core::builder::GraphBuilder builder("higher_rank", MatMulAddSchemas(), {{"", 28}});
        auto output_shape = dimensions;
        output_shape.at(output_shape.size() - 1) = 5;
        const auto bias_shape = full_bias ? output_shape : std::vector<int64_t>{5};
        builder.MakeInput("x", core::symbolic::TensorType::kFloat, MatMulAddShape(dimensions));
        builder.MakeInput("w", core::symbolic::TensorType::kFloat, MatMulAddShape({4, 5}));
        builder.MakeInput("bias", core::symbolic::TensorType::kFloat, MatMulAddShape(bias_shape));
        builder.MakeNode("MatMul", {"x", "w"}, {"mm"});
        builder.MakeNode("Add",
                         swapped ? std::vector<std::string>{"bias", "mm"}
                                 : std::vector<std::string>{"mm", "bias"},
                         {"out"});
        builder.MakeOutput("out");
        const core::runtime::Tensors inputs = {MatMulAddInput("x", dimensions),
                                               MatMulAddInput("w", {4, 5}),
                                               MatMulAddInput("bias", bias_shape)};
        const auto expected = core::runtime::RunModel(builder.ToModel(), inputs);
        ASSERT_TRUE(OptimizeMatMulAdd(builder, true));
        ASSERT_EQ(builder.Nodes().size(), full_bias ? 4u : 3u);
        EXPECT_EQ(builder.Nodes()[0].op_type(), "Reshape");
        EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 2].op_type(), "Gemm");
        EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 1].op_type(), "Reshape");
        EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 1].output()[0], "out");
        ExpectMatMulAddOutputs(core::runtime::RunModel(builder.ToModel(), inputs), expected);
      }
    }
  }
}

TEST(MatMulAddPattern, NumericallyPreservesDynamicBatchesAndWidthsFromOpset15) {
  onnx_kernels::RegisterKernelFunctions();
  for (int opset : {15, 28}) {
    for (bool full_bias : {false, true}) {
      SCOPED_TRACE(opset);
      SCOPED_TRACE(full_bias);
      core::builder::GraphBuilder builder("dynamic", MatMulAddSchemas(), {{"", opset}});
      const core::symbolic::SymDim batch("batch"), width("width");
      builder.MakeInput(
          "x", core::symbolic::TensorType::kFloat,
          core::symbolic::SymShape({batch, core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));
      builder.MakeInput("w", core::symbolic::TensorType::kFloat,
                        core::symbolic::SymShape({core::symbolic::SymDim(4), width}));
      builder.MakeInput("bias", core::symbolic::TensorType::kFloat,
                        full_bias
                            ? core::symbolic::SymShape({batch, core::symbolic::SymDim(3), width})
                            : core::symbolic::SymShape({width}));
      builder.MakeNode("MatMul", {"x", "w"}, {"mm"});
      builder.MakeNode("Add", {"mm", "bias"}, {"out"});
      builder.MakeOutput("out");
      const auto original = builder.ToModel();
      ASSERT_TRUE(OptimizeMatMulAdd(builder, true));
      int gemms = 0, shapes = 0;
      for (const auto &node : builder.Nodes()) {
        gemms += node.op_type() == "Gemm";
        shapes += node.op_type() == "Shape";
      }
      EXPECT_EQ(gemms, 1);
      EXPECT_EQ(shapes, full_bias ? 2 : 1);
      EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 1].output()[0], "out");
      for (int64_t b : {1, 2}) {
        for (int64_t n : {2, 5}) {
          SCOPED_TRACE(b);
          SCOPED_TRACE(n);
          const core::runtime::Tensors inputs = {
              MatMulAddInput("x", {b, 3, 4}), MatMulAddInput("w", {4, n}),
              MatMulAddInput("bias",
                             full_bias ? std::vector<int64_t>{b, 3, n} : std::vector<int64_t>{n})};
          ExpectMatMulAddOutputs(core::runtime::RunModel(builder.ToModel(), inputs),
                                 core::runtime::RunModel(original, inputs));
        }
      }
    }
  }
}

TEST(MatMulAddPattern, NumericallyPreservesAttentionAndLeavesBatchedScoresUnfused) {
  onnx_kernels::RegisterKernelFunctions();
  for (bool opt_in : {false, true}) {
    SCOPED_TRACE(opt_in);
    core::builder::GraphBuilder builder("attention", MatMulAddSchemas(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MatMulAddShape({2, 3, 4}));
    builder.MakeInput("w", core::symbolic::TensorType::kFloat, MatMulAddShape({4, 4}));
    builder.MakeInput("bias", core::symbolic::TensorType::kFloat, MatMulAddShape({4}));
    builder.MakeInput("keys", core::symbolic::TensorType::kFloat, MatMulAddShape({2, 2, 2, 3}));
    builder.MakeInput("mask", core::symbolic::TensorType::kFloat, MatMulAddShape({1, 1, 3, 3}));
    builder.MakeNode("MatMul", {"x", "w"}, {"projection"});
    builder.MakeNode("Add", {"projection", "bias"}, {"biased"});
    builder.MakeInitializer(MakeInitializerShape("heads", {2, 3, 2, 2}));
    builder.MakeNode("Reshape", {"biased", "heads"}, {"split"});
    auto transpose = MakeNode("Transpose", {"split"}, {"query"});
    AddAttribute<std::vector<int64_t>>(transpose, "perm", {0, 2, 1, 3});
    builder.MakeNode("Transpose", {"split"}, {"query"}, "", "", transpose.attribute());
    builder.MakeNode("MatMul", {"query", "keys"}, {"scores"});
    builder.MakeNode("Add", {"scores", "mask"}, {"masked"});
    builder.MakeNode("Softmax", {"masked"}, {"out"});
    builder.MakeOutput("out");
    const auto original = builder.ToModel();
    const core::runtime::Tensors inputs = {
        MatMulAddInput("x", {2, 3, 4}), MatMulAddInput("w", {4, 4}), MatMulAddInput("bias", {4}),
        MatMulAddInput("keys", {2, 2, 2, 3}), MatMulAddInput("mask", {1, 1, 3, 3})};
    EXPECT_EQ(OptimizeMatMulAdd(builder, opt_in), opt_in);
    if (!opt_in) {
      EXPECT_EQ(builder.ToModel().SerializeAsString(), original.SerializeAsString());
    }
    const std::vector<std::string> expected_ops =
        opt_in ? std::vector<std::string>{"Reshape",   "Gemm",   "Reshape", "Reshape",
                                          "Transpose", "MatMul", "Add",     "Softmax"}
               : std::vector<std::string>{"MatMul", "Add", "Reshape", "Transpose",
                                          "MatMul", "Add", "Softmax"};
    ASSERT_EQ(builder.Nodes().size(), expected_ops.size());
    for (std::size_t index = 0; index < expected_ops.size(); ++index) {
      EXPECT_EQ(builder.Nodes()[index].op_type(), expected_ops[index]);
    }
    EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 3].output()[0], "scores");
    EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 2].input()[1], "mask");
    EXPECT_EQ(builder.Nodes()[builder.Nodes().size() - 1].output()[0], "out");
    ExpectMatMulAddOutputs(core::runtime::RunModel(builder.ToModel(), inputs),
                           core::runtime::RunModel(original, inputs));
  }
}
