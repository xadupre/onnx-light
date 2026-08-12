// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder_pattern_optimization.h"
#include "onnx_core/builder/patterns/cast_cast_pattern.h"

#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

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

std::vector<const NodeProto *> Candidates(const core::builder::GraphBuilder &builder) {
  std::vector<const NodeProto *> candidates;
  for (const NodeProto &node : builder.Nodes()) {
    candidates.push_back(&node);
  }
  return candidates;
}

} // namespace

TEST(PatternOptimization, CastCastCollapsesRedundantOuterCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphBuilderPatternOptimization optimizer(builder);
  core::builder::CastCastPattern pattern;
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Cast"}));

  const std::vector<core::builder::MatchResult> matches =
      pattern.Match(optimizer, Candidates(builder));
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].pattern, &pattern);
  ASSERT_EQ(matches[0].nodes.size(), 2u);
  EXPECT_EQ(matches[0].insert_at, matches[0].nodes[1]);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(optimizer, matches[0].nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  ASSERT_NE(FindAttribute(replacements[0], "to"), nullptr);
  EXPECT_EQ(FindAttribute(replacements[0], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
}

TEST(PatternOptimization, CastCastUsesIdentityForSafeRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphBuilderPatternOptimization optimizer(builder);
  core::builder::CastCastPattern pattern;
  const std::vector<core::builder::MatchResult> matches =
      pattern.Match(optimizer, Candidates(builder));
  ASSERT_EQ(matches.size(), 1u);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(optimizer, matches[0].nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_TRUE(replacements[0].attribute().empty());
}

TEST(PatternOptimization, CastCastKeepsSharedInnerCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"middle"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphBuilderPatternOptimization optimizer(builder);
  core::builder::CastCastPattern pattern;
  const std::vector<core::builder::MatchResult> matches =
      pattern.Match(optimizer, Candidates(builder));
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].insert_at, nullptr);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(optimizer, matches[0].nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].output()[0].value(), "middle");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
}

TEST(PatternOptimization, CastCastRejectsLossyRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::INT32);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphBuilderPatternOptimization optimizer(builder);
  core::builder::CastCastPattern pattern;
  EXPECT_TRUE(pattern.Match(optimizer, Candidates(builder)).empty());
}

} // namespace Test
