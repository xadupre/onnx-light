// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
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
  shape.PushBack(core::symbolic::SymDim(3));
  return shape;
}

core::symbolic::SymShape ScalarShape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  return shape;
}

TEST(IdentityPattern, MatchesAddZeroScalar) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Add", {"x", "zero"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(IdentityPattern, MatchesMulOneScalarOnLeft) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("one", {1}, {1.0f}));
  builder.MakeNode("Mul", {"one", "x"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
}

TEST(IdentityPattern, MatchesIdentityTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(3));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &perm = attributes.add();
  perm.set_name("perm");
  perm.set_type(AttributeProto::AttributeType::INTS);
  perm.ref_ints().push_back(0);
  perm.ref_ints().push_back(1);
  builder.MakeNode("Transpose", {"x"}, {"y"}, "", "", attributes);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);
}

TEST(IdentityPattern, OptimizeReplacesNoOpAdd) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Add", {"x", "zero"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::IdentityPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(IdentityPattern, RejectsNonNeutralScalar) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("five", {1}, {5.0f}));
  builder.MakeNode("Add", {"x", "five"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the scalar constant operand is not a neutral element");
}

} // namespace
} // namespace Test
