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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

core::builder::GraphBuilder::SchemaLookupFn PoolSchemas() {
  return [](const std::string &op) { return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false); };
}

void ExpectPoolResultsEqual(const std::vector<core::runtime::Tensor> &actual,
                            const std::vector<core::runtime::Tensor> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t output = 0; output < actual.size(); ++output) {
    EXPECT_EQ(actual[output].shape, expected[output].shape);
    EXPECT_EQ(actual[output].data_type, expected[output].data_type);
    ASSERT_EQ(actual[output].size_bytes(), expected[output].size_bytes());
    EXPECT_EQ(
        std::memcmp(actual[output].bytes(), expected[output].bytes(), actual[output].size_bytes()),
        0);
  }
}

} // namespace

TEST(PadPoolPattern, NumericallyPreservesAsymmetricPaddingDilationAndCeilMode) {
  onnx_kernels::RegisterKernelFunctions();
  for (const auto *op : {"AveragePool", "MaxPool"}) {
    for (bool include : {false, true}) {
      for (bool dilated : {false, true}) {
        for (bool ceil : {false, true}) {
          if (ceil && std::string(op) == "AveragePool" && !include)
            continue;
          SCOPED_TRACE(std::string(op) + ":" + std::to_string(include) + ":" +
                       std::to_string(dilated) + ":" + std::to_string(ceil));
          const bool average = std::string(op) == "AveragePool";
          core::builder::GraphBuilder builder("g", PoolSchemas(), {{"", 19}});
          core::symbolic::SymShape shape;
          for (int64_t dim : {1, 1, 4, 5})
            shape.PushBack(core::symbolic::SymDim(dim));
          builder.MakeInput("x",
                            average ? core::symbolic::TensorType::kFloat
                                    : core::symbolic::TensorType::kUint8,
                            shape);
          builder.MakeInitializer(
              MakeInitializer<int64_t>("pads", {8}, {0, 0, 1, 0, 0, 0, 0, ceil ? 0 : 1}));
          if (average)
            builder.MakeInitializer(MakeInitializer<float>("value", {}, {0}));
          else {
            TensorProto value;
            value.set_name("value");
            value.set_data_type(TensorProto::UINT8);
            value.add_int32_data(0);
            builder.MakeInitializer(value);
          }
          builder.MakeNode("Pad", {"x", "pads", "value"}, {"padded"});
          NodeProto pool;
          AddAttribute<std::vector<int64_t>>(pool, "kernel_shape", {3, 3});
          AddAttribute<std::vector<int64_t>>(pool, "strides", {2, 2});
          if (include) {
            AddAttribute<std::vector<int64_t>>(pool, "pads", {0, 1, 0, 0});
            if (average)
              AddAttribute<int64_t>(pool, "count_include_pad", 1);
          }
          if (dilated)
            AddAttribute<std::vector<int64_t>>(pool, "dilations", {1, 2});
          if (ceil)
            AddAttribute<int64_t>(pool, "ceil_mode", 1);
          builder.MakeNode(op, {"padded"}, {"out"}, "", "", pool.attribute());
          builder.MakeOutput("out");
          const std::vector<core::runtime::Tensor> feeds{
              average ? core::runtime::Tensor::FromFloat(
                            "x", {1, 1, 4, 5},
                            {-2, 3, -4, 1, 5, -6, 7, -8, 2, 0, 3, -1, 4, -5, 6, -7, 8, -2, 1, 9})
                      : core::runtime::Tensor::FromUint8(
                            "x", {1, 1, 4, 5},
                            {0, 3, 4, 1, 5, 6, 7, 8, 2, 0, 3, 1, 4, 5, 6, 7, 8, 2, 1, 9})};
          const auto expected = core::runtime::RunModel(builder.ToModel(), feeds);
          std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
          patterns.push_back(onnx_patterns::CreatePattern("PadPool"));
          core::builder::GraphGraph graph(builder, std::move(patterns));
          core::builder::ConstantFoldingOptions folding;
          folding.enabled = false;
          ASSERT_FALSE(graph.Optimize(2, nullptr, folding).rewrites().empty());
          ASSERT_EQ(builder.Nodes().size(), 1u);
          EXPECT_EQ(builder.Nodes()[0].op_type(), op);
          EXPECT_EQ(builder.Nodes()[0].input(0), "x");
          EXPECT_EQ(builder.Nodes()[0].output(0), "out");
          ExpectPoolResultsEqual(core::runtime::RunModel(builder.ToModel(), feeds), expected);
        }
      }
    }
  }
}

TEST(PadPoolPattern, NumericallyPreservesRefusedNegativeMaximaDivisorsAndIndices) {
  onnx_kernels::RegisterKernelFunctions();
  for (const auto *variant : {"negative_max", "indices", "existing_pad", "ceil_divisor",
                              "ceil_tail", "nonzero_value", "negative_pad"}) {
    SCOPED_TRACE(variant);
    const std::string test = variant;
    const bool maximum = test == "negative_max" || test == "indices" || test == "ceil_tail";
    core::builder::GraphBuilder builder("g", PoolSchemas(), {{"", 19}});
    core::symbolic::SymShape shape;
    for (int64_t dim : {1, 1, 4})
      shape.PushBack(core::symbolic::SymDim(dim));
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
    builder.MakeInitializer(MakeInitializer<int64_t>(
        "pads", {6}, {0, 0, test == "negative_pad" ? -1 : 1, 0, 0, test == "ceil_tail" ? 1 : 0}));
    builder.MakeInitializer(
        MakeInitializer<float>("value", {}, {test == "nonzero_value" ? 2.0F : 0.0F}));
    builder.MakeNode("Pad", {"x", "pads", "value"}, {"padded"});
    NodeProto pool;
    AddAttribute<std::vector<int64_t>>(pool, "kernel_shape", {2});
    AddAttribute<std::vector<int64_t>>(pool, "strides", {2});
    if (test == "existing_pad")
      AddAttribute<std::vector<int64_t>>(pool, "pads", {0, 1});
    if (test == "ceil_divisor" || test == "ceil_tail")
      AddAttribute<int64_t>(pool, "ceil_mode", 1);
    builder.MakeNode(maximum ? "MaxPool" : "AveragePool", {"padded"},
                     test == "indices" ? std::vector<std::string>{"out", "indices"}
                                       : std::vector<std::string>{"out"},
                     "", "", pool.attribute());
    builder.MakeOutput("out");
    if (test == "indices")
      builder.MakeOutput("indices");
    const std::vector<core::runtime::Tensor> feeds{
        core::runtime::Tensor::FromFloat("x", {1, 1, 4}, {-2, -4, -6, -8})};
    const auto before = builder.ToModel();
    const auto expected = core::runtime::RunModel(before, feeds);
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(onnx_patterns::CreatePattern("PadPool"));
    core::builder::GraphGraph graph(builder, std::move(patterns));
    core::builder::ConstantFoldingOptions folding;
    folding.enabled = false;
    EXPECT_TRUE(graph.Optimize(2, nullptr, folding).rewrites().empty());
    EXPECT_TRUE(builder.ToModel().Equals(before));
    ExpectPoolResultsEqual(core::runtime::RunModel(builder.ToModel(), feeds), expected);
  }
}
