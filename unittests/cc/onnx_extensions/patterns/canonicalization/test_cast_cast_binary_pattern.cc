// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/cast_cast_binary_pattern.h"
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

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType to) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("to");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(static_cast<int64_t>(to));
  builder.MakeNode("Cast", {input}, {output}, "", "", attributes);
}

TEST(CastCastBinaryPattern, MovesCastsAfterBinaryOperation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT16);
  AddCast(builder, "y", "cast_y", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Add", {"cast_x", "cast_y"}, {"z"}, "", "sum");
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastBinaryPattern pattern;
  EXPECT_EQ(pattern.priority, 1);
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Add", "Div", "Mul", "Sub"}));

  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[2]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Add");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[1].op_type().value(), "Cast");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "z");
  ASSERT_NE(FindAttribute(replacements[1], "to"), nullptr);
  EXPECT_EQ(FindAttribute(replacements[1], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT16));
}

TEST(CastCastBinaryPattern, ProducesExpectedOptimizedGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT16);
  AddCast(builder, "y", "cast_y", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Mul", {"cast_x", "cast_y"}, {"z"});
  builder.MakeOutput("z");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastBinaryPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Mul");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].input()[1].value(), "y");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Cast");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), builder.Nodes()[0].output()[0].value());
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "z");
}

TEST(CastCastBinaryPattern, RejectsSharedCastOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT16);
  AddCast(builder, "y", "cast_y", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Add", {"cast_x", "cast_y"}, {"z"});
  builder.MakeNode("Identity", {"cast_x"}, {"shared"});
  builder.MakeOutput("z");
  builder.MakeOutput("shared");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastBinaryPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a Cast output has another use");
}

TEST(CastCastBinaryPattern, RejectsOneCastFeedingBothInputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Add", {"cast_x", "cast_x"}, {"z"});
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastBinaryPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the binary inputs must be produced by distinct Cast nodes");
}

TEST(CastCastBinaryPattern, RejectsLowerPrecisionComputation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  AddCast(builder, "y", "cast_y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Sub", {"cast_x", "cast_y"}, {"z"});
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastBinaryPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "moving the binary operation would lower its precision");
}

} // namespace
} // namespace Test
