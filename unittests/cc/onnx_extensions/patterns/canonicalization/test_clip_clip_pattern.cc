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

} // namespace
} // namespace Test
