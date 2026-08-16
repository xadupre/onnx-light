// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/slice_pattern.h"
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

TEST(SliceSlicePattern, MergesDisjointAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  builder.MakeInitializer(MakeInitializerShape("s0", {0}));
  builder.MakeInitializer(MakeInitializerShape("e0", {2}));
  builder.MakeInitializer(MakeInitializerShape("a0", {0}));
  builder.MakeInitializer(MakeInitializerShape("s1", {1}));
  builder.MakeInitializer(MakeInitializerShape("e1", {4}));
  builder.MakeInitializer(MakeInitializerShape("a1", {1}));
  builder.MakeNode("Slice", {"x", "s0", "e0", "a0"}, {"t"});
  builder.MakeNode("Slice", {"t", "s1", "e1", "a1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 4u);
  EXPECT_EQ(replacements[0].op_type().value(), "Concat");
  EXPECT_EQ(replacements[1].op_type().value(), "Concat");
  EXPECT_EQ(replacements[2].op_type().value(), "Concat");
  EXPECT_EQ(replacements[3].op_type().value(), "Slice");
  ASSERT_EQ(replacements[3].input_size(), 4);
  EXPECT_EQ(replacements[3].input()[0].value(), "x");
  EXPECT_EQ(replacements[3].output()[0].value(), "y");
}

TEST(SliceSlicePattern, MaterialisesOnesStepWhenOnlyOneHasSteps) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  builder.MakeInitializer(MakeInitializerShape("s0", {0}));
  builder.MakeInitializer(MakeInitializerShape("e0", {2}));
  builder.MakeInitializer(MakeInitializerShape("a0", {0}));
  builder.MakeInitializer(MakeInitializerShape("s1", {1}));
  builder.MakeInitializer(MakeInitializerShape("e1", {4}));
  builder.MakeInitializer(MakeInitializerShape("a1", {1}));
  builder.MakeInitializer(MakeInitializerShape("st1", {2}));
  builder.MakeNode("Slice", {"x", "s0", "e0", "a0"}, {"t"});
  builder.MakeNode("Slice", {"t", "s1", "e1", "a1", "st1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 5u);
  EXPECT_EQ(replacements[3].op_type().value(), "Concat");
  EXPECT_EQ(replacements[4].op_type().value(), "Slice");
  ASSERT_EQ(replacements[4].input_size(), 5);
}

TEST(SliceSlicePattern, RejectsOverlappingAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  builder.MakeInitializer(MakeInitializerShape("s0", {0}));
  builder.MakeInitializer(MakeInitializerShape("e0", {2}));
  builder.MakeInitializer(MakeInitializerShape("a0", {0}));
  builder.MakeInitializer(MakeInitializerShape("s1", {1}));
  builder.MakeInitializer(MakeInitializerShape("e1", {4}));
  builder.MakeInitializer(MakeInitializerShape("a1", {0}));
  builder.MakeNode("Slice", {"x", "s0", "e0", "a0"}, {"t"});
  builder.MakeNode("Slice", {"t", "s1", "e1", "a1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the two slices share an axis");
}

} // namespace
} // namespace Test
