// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/dropout_pattern.h"
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

TEST(DropoutPattern, ReplacesInferenceDropout) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Dropout", {"x"}, {"y", "mask"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::DropoutPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 1u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(DropoutPattern, ProducesIdentityGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Dropout", {"x"}, {"y", "mask"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::DropoutPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(DropoutPattern, RejectsUsedMaskOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Dropout", {"x"}, {"y", "mask"});
  builder.MakeNode("Identity", {"mask"}, {"mask_used"});
  builder.MakeOutput("y");
  builder.MakeOutput("mask_used");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::DropoutPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a Dropout mask output is used");
}

TEST(DropoutPattern, RejectsEnabledTrainingMode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("ratio", {}, {0.5f}));
  TensorProto training_mode;
  training_mode.set_name("training_mode");
  training_mode.set_data_type(TensorProto::DataType::BOOL);
  training_mode.ref_int32_data().push_back(1);
  builder.MakeInitializer(training_mode);
  builder.MakeNode("Dropout", {"x", "ratio", "training_mode"}, {"y", "mask"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::DropoutPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Dropout training mode is enabled");
}

} // namespace
} // namespace Test
