// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/slice_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <limits>
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

TEST(SliceEliminationPattern, ReplacesModernFullRangeSlice) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  builder.MakeInitializer(MakeInitializerShape("starts", {0, 0}));
  builder.MakeInitializer(MakeInitializerShape(
      "ends", {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()}));
  builder.MakeInitializer(MakeInitializerShape("axes", {0, 1}));
  builder.MakeInitializer(MakeInitializerShape("steps", {1, 1}));
  builder.MakeNode("Slice", {"x", "starts", "ends", "axes", "steps"}, {"y"}, "", "slice");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(SliceEliminationPattern, SupportsAttributeForm) {
  core::builder::GraphBuilder builder("g");
  builder.SetOpsetVersion("", 9);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  NodeProto slice = MakeNode("Slice", {"x"}, {"y"});
  AddAttribute<std::vector<int64_t>>(slice, "starts", {0});
  AddAttribute<std::vector<int64_t>>(slice, "ends", {std::numeric_limits<int64_t>::max()});
  AddAttribute<std::vector<int64_t>>(slice, "axes", {1});

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceEliminationPattern pattern;
  EXPECT_EQ(pattern.Match(graph, slice).pattern, &pattern);
}

TEST(SliceEliminationPattern, RejectsNonIdentityParameters) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  builder.MakeInitializer(MakeInitializerShape("starts", {0}));
  builder.MakeInitializer(MakeInitializerShape("ends", {std::numeric_limits<int64_t>::max()}));
  builder.MakeInitializer(MakeInitializerShape("steps", {2}));
  builder.MakeNode("Slice", {"x", "starts", "ends", "", "steps"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceEliminationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Slice steps are not all one");
}

} // namespace
} // namespace Test
