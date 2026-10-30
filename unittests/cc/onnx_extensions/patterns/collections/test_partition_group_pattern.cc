// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/collections/partition_group_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

using core::builder::BuilderError;
using core::builder::GraphBuilder;
using core::builder::GraphGraph;
using onnx_patterns::PartitionGroupPattern;

GraphBuilder::SchemaLookupFn SchemaLookup() {
  onnx_shapes::RegisterShapeFunctions();
  return
      [](const std::string &name) { return onnx_op::GetAllOnnxOpSchemasWithHistory(name, false); };
}

core::symbolic::SymShape Shape(std::initializer_list<int64_t> dimensions = {}) {
  core::symbolic::SymShape shape;
  for (int64_t dimension : dimensions)
    shape.PushBack(core::symbolic::SymDim(dimension));
  return shape;
}

GraphBuilder Chain(size_t count = 2, int64_t width = 1) {
  GraphBuilder builder("chain", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({width}));
  std::string input = "x";
  for (size_t i = 0; i < count; ++i) {
    const std::string output = i + 1 == count ? "y" : "v" + std::to_string(i);
    builder.MakeNode(i % 2 == 0 ? "Neg" : "Abs", {input}, {output});
    input = output;
  }
  builder.MakeOutput("y");
  return builder;
}

RewritingProto Optimize(GraphBuilder &builder, uint64_t bytes = 1024, uint64_t maximum = 128) {
  std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns{
      std::make_shared<PartitionGroupPattern>(0, bytes, 2, maximum)};
  GraphGraph graph(builder, patterns);
  return graph.Optimize();
}

TEST(PartitionGroupPattern, CallsCoreSelectionAndPreservesShapeAndNames) {
  auto builder = Chain(4);
  EXPECT_FALSE(Optimize(builder, 8, 2).rewrites().empty());
  ASSERT_EQ(builder.Nodes().size(), 2u);
  ASSERT_EQ(builder.LocalFunctions().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].domain(), "ai.rt");
  EXPECT_EQ(builder.Nodes()[0].op_type(), "PartitionGroup_Neg_0");
  EXPECT_EQ(builder.Nodes()[1].op_type(), "PartitionGroup_Neg_1");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[0].output(0), "v1");
  EXPECT_EQ(builder.Nodes()[1].input(0), "v1");
  EXPECT_EQ(builder.Nodes()[1].output(0), "y");
  EXPECT_TRUE(builder.HasShape("y"));
  VerifyModel(builder.ToModel());
  EXPECT_TRUE(Optimize(builder, 8, 2).rewrites().empty());
  EXPECT_EQ(builder.LocalFunctions().size(), 2u);
}

TEST(PartitionGroupPattern, RunsOnlyAtFirstNodeAndRevalidatesApply) {
  auto builder = Chain();
  GraphGraph graph(builder);
  PartitionGroupPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_THROW(pattern.Apply(graph, {}), BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {nullptr}), BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {match.nodes[1], match.nodes[0]}), BuilderError);
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type(), "PartitionGroup_Neg_0");
}

TEST(PartitionGroupPattern, LimitsAndDefaultSelectionArePreserved) {
  onnx_patterns::RegisterPatterns();
  const auto names = core::builder::RegisteredPatternNames(false);
  EXPECT_EQ(std::find(names.begin(), names.end(), "PartitionGroup"), names.end());
  EXPECT_EQ(onnx_patterns::CreatePattern("PartitionGroup")->Name(), "PartitionGroup");
  EXPECT_THROW(PartitionGroupPattern(0, 1024, 1, 128), std::invalid_argument);
  EXPECT_THROW(PartitionGroupPattern(0, 1024, 3, 2), std::invalid_argument);
  auto builder = Chain();
  EXPECT_TRUE(Optimize(builder, 7).rewrites().empty());
  EXPECT_TRUE(builder.LocalFunctions().empty());
  EXPECT_FALSE(Optimize(builder, 8).rewrites().empty());
  auto large = Chain(4);
  EXPECT_FALSE(Optimize(large, 8, 4).rewrites().empty());
  ASSERT_EQ(large.Nodes().size(), 1u);
  EXPECT_EQ(large.LocalFunctions()[0]->Nodes().size(), 4u);
}

TEST(PartitionGroupPattern, KeepsFanoutAndIntermediateGraphOutputs) {
  auto builder = Chain(3);
  builder.MakeOutput("v0");
  EXPECT_FALSE(Optimize(builder, 12).rewrites().empty());
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].output_size(), 2);
  EXPECT_EQ(builder.Nodes()[0].output(0), "v0");
  EXPECT_EQ(builder.Nodes()[0].output(1), "y");
  EXPECT_EQ(builder.LocalFunctions()[0]->Outputs().size(), 2u);
  VerifyModel(builder.ToModel());
}

TEST(PartitionGroupPattern, KeepsInterleavedConstantsBeforeCallsAndMetadataInBodies) {
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1}));
  builder.MakeNode("Neg", {"x"}, {"middle"}, "", "negative");
  NodeProto constant = MakeNode("Constant", {}, {"weight"});
  auto &value = *constant.add_attribute();
  value.set_name("value");
  value.set_type(AttributeProto::TENSOR);
  value.mutable_t()->set_data_type(TensorProto::FLOAT);
  value.mutable_t()->add_dims(1);
  value.mutable_t()->add_float_data(2);
  builder.ImportNode(constant);
  builder.MakeNode("Add", {"middle", "weight"}, {"y"});
  builder.MakeOutput("y");
  GraphGraph graph(builder);
  PartitionGroupPattern pattern(0, 8);
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 2u);
  EXPECT_EQ(replacement[0].op_type(), "Constant");
  EXPECT_EQ(replacement[1].input_size(), 1);
  EXPECT_EQ(replacement[1].input(0), "x");
  const auto function = builder.LocalFunctions()[0]->ToFunction("ai.rt");
  ASSERT_EQ(function.node_size(), 3);
  EXPECT_EQ(function.node(0).op_type(), "Constant");
  EXPECT_EQ(function.node(1).name(), "negative");
}

TEST(PartitionGroupPattern, DoesNotRewriteFunctionBodiesOrConflictingDomainVersions) {
  GraphBuilder model("model", SchemaLookup(), {{"", 18}});
  auto &body = model.MakeLocalFunction("Body", "local");
  body.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1}));
  body.MakeNode("Neg", {"x"}, {"middle"});
  body.MakeNode("Abs", {"middle"}, {"y"});
  body.MakeOutput("y");
  GraphGraph graph(body);
  EXPECT_EQ(PartitionGroupPattern().Match(graph, body.Nodes()[0]).pattern, nullptr);
  auto conflict = Chain();
  conflict.SetOpsetVersion("ai.rt", 2);
  GraphGraph conflicting_graph(conflict);
  EXPECT_EQ(PartitionGroupPattern().Match(conflicting_graph, conflict.Nodes()[0]).pattern, nullptr);
}

} // namespace
