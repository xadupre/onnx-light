// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/cast_op_cast_pattern.h"
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

TEST(CastOpCastPattern, RecastsUnconvertedBinaryInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "y", "cast_y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Add", {"x", "cast_y"}, {"sum"});
  AddCast(builder, "sum", "z", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastOpCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[1], &builder.Nodes()[1]);
  EXPECT_EQ(match.nodes[2], &builder.Nodes()[2]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  ASSERT_NE(FindAttribute(replacements[0], "to"), nullptr);
  EXPECT_EQ(FindAttribute(replacements[0], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT16));
  EXPECT_EQ(replacements[1].op_type().value(), "Add");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "y");
  EXPECT_EQ(replacements[1].output()[0].value(), "z");
}

TEST(CastOpCastPattern, ProducesExpectedUnaryGraphAndPreservesAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &axis = attributes.add();
  axis.set_name("axis");
  axis.set_type(AttributeProto::AttributeType::INT);
  axis.set_i(0);
  builder.MakeNode("Softmax", {"cast_x"}, {"softmax"}, "", "probabilities", attributes);
  AddCast(builder, "softmax", "z", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("z");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastOpCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Softmax");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "z");
  ASSERT_NE(FindAttribute(builder.Nodes()[0], "axis"), nullptr);
  EXPECT_EQ(FindAttribute(builder.Nodes()[0], "axis")->i(), 0);
}

TEST(CastOpCastPattern, PreservesSharedComputationOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"cast_x"}, {"negative"});
  AddCast(builder, "negative", "z", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("negative");
  builder.MakeOutput("z");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastOpCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "z");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Cast");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "z");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "negative");
  ASSERT_NE(FindAttribute(builder.Nodes()[1], "to"), nullptr);
  EXPECT_EQ(FindAttribute(builder.Nodes()[1], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
}

TEST(CastOpCastPattern, RejectsOperationWithoutInputCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Mul", {"x", "y"}, {"product"});
  AddCast(builder, "product", "z", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastOpCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "at least one operation input must be produced by a Cast");
}

TEST(CastOpCastPattern, RejectsMismatchedRoundTripTypes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kBfloat16, Shape());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("Sub", {"cast_x", "y"}, {"difference"});
  AddCast(builder, "difference", "z", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastOpCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason,
            "input Cast source and target types do not match the output and computation");
}

TEST(CastOpCastPattern, RejectsSharedInputCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"cast_x"}, {"negative"});
  AddCast(builder, "negative", "z", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Identity", {"cast_x"}, {"shared"});
  builder.MakeOutput("z");
  builder.MakeOutput("shared");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastOpCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an operation input has another use");
}

} // namespace
} // namespace Test
