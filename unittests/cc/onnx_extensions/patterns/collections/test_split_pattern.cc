// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/split_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape2D(int64_t a, int64_t b) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  shape.PushBack(core::symbolic::SymDim(b));
  return shape;
}

utils::RepeatedProtoField<AttributeProto> AxisAttrs(int64_t axis) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attr = attributes.add();
  attr.set_name("axis");
  attr.set_type(AttributeProto::AttributeType::INT);
  attr.set_i(axis);
  return attributes;
}

TEST(SplitConcatPattern, FoldsSplitConcatIntoIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 4));
  builder.MakeNode("Split", {"x"}, {"s0", "s1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Concat", {"s0", "s1"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SplitConcatPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[1], &builder.Nodes()[1]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  ASSERT_EQ(replacements[0].input_size(), 1);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(SplitConcatPattern, OptimizesToIdentityGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 4));
  builder.MakeNode("Split", {"x"}, {"s0", "s1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Concat", {"s0", "s1"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::SplitConcatPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(SplitConcatPattern, RejectsMismatchedAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 4));
  builder.MakeNode("Split", {"x"}, {"s0", "s1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Concat", {"s0", "s1"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SplitConcatPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Split and Concat axes differ");
}

TEST(SplitConcatPattern, RejectsReorderedConcat) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 4));
  builder.MakeNode("Split", {"x"}, {"s0", "s1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Concat", {"s1", "s0"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SplitConcatPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Concat inputs are not the Split outputs in order");
}

TEST(SplitConcatPattern, RejectsExtraConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 4));
  builder.MakeNode("Split", {"x"}, {"s0", "s1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Concat", {"s0", "s1"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeNode("Identity", {"s0"}, {"extra"});
  builder.MakeOutput("y");
  builder.MakeOutput("extra");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SplitConcatPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a Split output is not consumed by exactly one node");
}

TEST(GathersSplitPattern, MergesScalarGathersIntoSplitAndSqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 2));
  builder.MakeInitializer(MakeInitializer<int64_t>("i0", {}, {0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i1", {}, {1}));
  builder.MakeNode("Gather", {"x", "i0"}, {"x1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Gather", {"x", "i1"}, {"x2"}, "", "", AxisAttrs(1));
  builder.MakeOutput("x1");
  builder.MakeOutput("x2");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GathersSplitPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Split");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", -1), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "num_outputs", -1), 2);
  EXPECT_EQ(replacements[1].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[2].op_type().value(), "Squeeze");
}

TEST(GathersSplitPattern, RejectsWhenIndicesDoNotCoverAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 3));
  builder.MakeInitializer(MakeInitializer<int64_t>("i0", {}, {0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i1", {}, {1}));
  builder.MakeNode("Gather", {"x", "i0"}, {"x1"}, "", "", AxisAttrs(1));
  builder.MakeNode("Gather", {"x", "i1"}, {"x2"}, "", "", AxisAttrs(1));
  builder.MakeOutput("x1");
  builder.MakeOutput("x2");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GathersSplitPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SlicesSplitPattern, MergesContiguousSlicesIntoSplit) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 8));
  builder.MakeInitializer(MakeInitializerShape("zero", {0}));
  builder.MakeInitializer(MakeInitializerShape("four", {4}));
  builder.MakeInitializer(MakeInitializerShape("eight", {8}));
  builder.MakeInitializer(MakeInitializerShape("axis1", {1}));
  builder.MakeNode("Slice", {"x", "zero", "four", "axis1"}, {"a"});
  builder.MakeNode("Slice", {"x", "four", "eight", "axis1"}, {"b"});
  builder.MakeOutput("a");
  builder.MakeOutput("b");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SlicesSplitPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Split");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", -1), 1);
  ASSERT_EQ(replacements[0].output_size(), 2);
  EXPECT_EQ(replacements[0].output()[0].value(), "a");
  EXPECT_EQ(replacements[0].output()[1].value(), "b");
}

TEST(SlicesSplitPattern, RejectsNonContiguousSlices) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 8));
  builder.MakeInitializer(MakeInitializerShape("zero", {0}));
  builder.MakeInitializer(MakeInitializerShape("three", {3}));
  builder.MakeInitializer(MakeInitializerShape("four", {4}));
  builder.MakeInitializer(MakeInitializerShape("eight", {8}));
  builder.MakeInitializer(MakeInitializerShape("axis1", {1}));
  builder.MakeNode("Slice", {"x", "zero", "three", "axis1"}, {"a"});
  builder.MakeNode("Slice", {"x", "four", "eight", "axis1"}, {"b"});
  builder.MakeOutput("a");
  builder.MakeOutput("b");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SlicesSplitPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the sibling Slice ranges are not contiguous");
}

} // namespace
} // namespace Test
