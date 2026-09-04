// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"
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

core::symbolic::SymShape Shape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  return shape;
}

core::symbolic::SymShape ScalarShape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  return shape;
}

TEST(ClipClipPattern, MergesComplementaryBounds) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("mn", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeInput("mx", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeNode("Clip", {"x", "mn"}, {"x1"});
  builder.MakeNode("Clip", {"x1", "", "mx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ClipClipPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[1], &builder.Nodes()[1]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Clip");
  ASSERT_EQ(replacements[0].input_size(), 3);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "mn");
  EXPECT_EQ(replacements[0].input()[2].value(), "mx");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(ClipClipPattern, ProducesExpectedMergedGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("mn", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeInput("mx", core::symbolic::TensorType::kFloat, ScalarShape());
  // The first Clip provides the maximum, the second one the minimum.
  builder.MakeNode("Clip", {"x", "", "mx"}, {"x1"});
  builder.MakeNode("Clip", {"x1", "mn"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::ClipClipPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Clip");
  ASSERT_EQ(builder.Nodes()[0].input_size(), 3);
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].input()[1].value(), "mn");
  EXPECT_EQ(builder.Nodes()[0].input()[2].value(), "mx");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(ClipClipPattern, RejectsSameBound) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("mn1", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeInput("mn2", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeNode("Clip", {"x", "mn1"}, {"x1"});
  builder.MakeNode("Clip", {"x1", "mn2"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ClipClipPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "exactly one Clip must define the minimum bound");
}

TEST(ClipClipPattern, RejectsSharedIntermediate) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("mn", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeInput("mx", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeNode("Clip", {"x", "mn"}, {"x1"});
  builder.MakeNode("Clip", {"x1", "", "mx"}, {"y"});
  builder.MakeNode("Identity", {"x1"}, {"shared"});
  builder.MakeOutput("y");
  builder.MakeOutput("shared");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ClipClipPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the first Clip output has another use");
}

TEST(ClipClipPattern, RejectsNonClipPredecessor) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("mx", core::symbolic::TensorType::kFloat, ScalarShape());
  builder.MakeNode("Relu", {"x"}, {"x1"});
  builder.MakeNode("Clip", {"x1", "", "mx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ClipClipPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the preceding node is not a default-domain Clip");
}

TEST(ReluClipFusionPattern, RemovesReluForNonNegativeConstantMinimum) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("min", {}, {0.5F}));
  builder.MakeInitializer(MakeInitializer<float>("max", {}, {6.0F}));
  builder.MakeNode("Relu", {"x"}, {"r"}, "", "relu");
  builder.MakeNode("Clip", {"r", "min", "max"}, {"y"}, "", "clip");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReluClipFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Clip");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "min");
}

TEST(ReluClipFusionPattern, SupportsAttributeForm) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 10);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Relu", {"x"}, {"r"});
  NodeProto clip = MakeNode("Clip", {"r"}, {"y"});
  AddAttribute<float>(clip, "min", 0.0F);
  AddAttribute<float>(clip, "max", 6.0F);
  builder.MakeNode("Clip", {"r"}, {"y"}, "", "clip", clip.attribute());
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReluClipFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, &pattern);
}

TEST(ReluClipFusionPattern, RejectsMissingNegativeAndSharedMinimumCases) {
  core::builder::GraphBuilder missing("missing", SchemaLookup());
  missing.SetOpsetVersion("", 18);
  missing.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  missing.MakeNode("Relu", {"x"}, {"r"});
  missing.MakeNode("Clip", {"r"}, {"y"});
  missing.MakeOutput("y");
  core::builder::GraphGraph missing_graph(missing);

  core::builder::GraphBuilder negative("negative", SchemaLookup());
  negative.SetOpsetVersion("", 18);
  negative.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  negative.MakeInitializer(MakeInitializer<float>("min", {}, {-1.0F}));
  negative.MakeNode("Relu", {"x"}, {"r"});
  negative.MakeNode("Clip", {"r", "min"}, {"y"});
  negative.MakeOutput("y");
  core::builder::GraphGraph negative_graph(negative);

  core::builder::GraphBuilder shared("shared", SchemaLookup());
  shared.SetOpsetVersion("", 18);
  shared.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  shared.MakeInitializer(MakeInitializer<float>("min", {}, {0.0F}));
  shared.MakeNode("Relu", {"x"}, {"r"});
  shared.MakeNode("Clip", {"r", "min"}, {"y"});
  shared.MakeNode("Identity", {"r"}, {"other"});
  shared.MakeOutput("y");
  shared.MakeOutput("other");
  core::builder::GraphGraph shared_graph(shared);

  onnx_patterns::ReluClipFusionPattern pattern;
  EXPECT_EQ(pattern.Match(missing_graph, missing.Nodes()[1]).pattern, nullptr);
  EXPECT_EQ(pattern.Match(negative_graph, negative.Nodes()[1]).pattern, nullptr);
  EXPECT_EQ(pattern.Match(shared_graph, shared.Nodes()[1]).pattern, nullptr);
}

} // namespace
} // namespace Test
