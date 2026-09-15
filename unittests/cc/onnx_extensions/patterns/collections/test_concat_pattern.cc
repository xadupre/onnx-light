// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/concat_pattern.h"
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

core::symbolic::SymShape Shape1D(int64_t a) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  return shape;
}

core::symbolic::SymShape Shape2D(int64_t a, int64_t b) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  shape.PushBack(core::symbolic::SymDim(b));
  return shape;
}

core::symbolic::SymShape Shape4D(int64_t a, int64_t b, int64_t c, int64_t d) {
  core::symbolic::SymShape shape;
  for (int64_t value : {a, b, c, d}) {
    shape.PushBack(core::symbolic::SymDim(value));
  }
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

TEST(ConcatEmptyPattern, DropsEmptyInputToIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape2D(2, 0));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeNode("Concat", {"a", "b"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatEmptyPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "b");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(ConcatEmptyPattern, DropsEmptyInputKeepsConcat) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape2D(2, 0));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInput("c", core::symbolic::TensorType::kFloat, Shape2D(2, 4));
  builder.MakeNode("Concat", {"a", "b", "c"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatEmptyPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Concat");
  ASSERT_EQ(replacements[0].input_size(), 2);
  EXPECT_EQ(replacements[0].input()[0].value(), "b");
  EXPECT_EQ(replacements[0].input()[1].value(), "c");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", 0), 1);
}

TEST(ConcatEmptyPattern, RejectsWhenNoEmptyInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape2D(2, 2));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeNode("Concat", {"a", "b"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatEmptyPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ConcatSliceEliminationPattern, RecoversConcatInputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape2D(2, 2));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInitializer(MakeInitializerShape("s0", {0}));
  builder.MakeInitializer(MakeInitializerShape("s2", {2}));
  builder.MakeInitializer(MakeInitializerShape("s5", {5}));
  builder.MakeInitializer(MakeInitializerShape("axis", {1}));
  builder.MakeNode("Concat", {"a", "b"}, {"c"}, "", "", AxisAttrs(1));
  builder.MakeNode("Slice", {"c", "s2", "s5", "axis"}, {"yb"});
  builder.MakeNode("Slice", {"c", "s0", "s2", "axis"}, {"ya"});
  builder.MakeOutput("ya");
  builder.MakeOutput("yb");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatSliceEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].input()[0].value(), "a");
  EXPECT_EQ(replacements[0].output()[0].value(), "ya");
  EXPECT_EQ(replacements[1].input()[0].value(), "b");
  EXPECT_EQ(replacements[1].output()[0].value(), "yb");
}

TEST(ConcatSliceEliminationPattern, RejectsInexactPartition) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape2D(2, 2));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInitializer(MakeInitializerShape("s0", {0}));
  builder.MakeInitializer(MakeInitializerShape("s2", {2}));
  builder.MakeInitializer(MakeInitializerShape("s4", {4}));
  builder.MakeInitializer(MakeInitializerShape("axis", {1}));
  builder.MakeNode("Concat", {"a", "b"}, {"c"}, "", "", AxisAttrs(1));
  builder.MakeNode("Slice", {"c", "s0", "s2", "axis"}, {"ya"});
  builder.MakeNode("Slice", {"c", "s2", "s4", "axis"}, {"yb"});
  builder.MakeOutput("ya");
  builder.MakeOutput("yb");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatSliceEliminationPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConcatSliceEliminationPattern, RejectsEscapingConcatOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape2D(2, 2));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInitializer(MakeInitializerShape("s0", {0}));
  builder.MakeInitializer(MakeInitializerShape("s2", {2}));
  builder.MakeInitializer(MakeInitializerShape("s5", {5}));
  builder.MakeInitializer(MakeInitializerShape("axis", {1}));
  builder.MakeNode("Concat", {"a", "b"}, {"c"}, "", "", AxisAttrs(1));
  builder.MakeNode("Slice", {"c", "s0", "s2", "axis"}, {"ya"});
  builder.MakeNode("Slice", {"c", "s2", "s5", "axis"}, {"yb"});
  builder.MakeOutput("c");
  builder.MakeOutput("ya");
  builder.MakeOutput("yb");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatSliceEliminationPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(SliceConcatToSpaceToDepthPattern, FusesCanonicalFocus) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 3, 8, 10));
  builder.MakeInitializer(MakeInitializer<int64_t>("ends", {2}, {8, 10}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {2}, {2, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("steps", {2}, {2, 2}));
  for (int i = 0; i < 4; ++i) {
    builder.MakeInitializer(
        MakeInitializer<int64_t>(("starts" + std::to_string(i)).c_str(), {2}, {i / 2, i % 2}));
    builder.MakeNode("Slice", {"x", "starts" + std::to_string(i), "ends", "axes", "steps"},
                     {"s" + std::to_string(i)});
  }
  builder.MakeNode("Concat", {"s0", "s1", "s2", "s3"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceConcatToSpaceToDepthPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[4]);
  ASSERT_EQ(match.pattern, &pattern);
  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "SpaceToDepth");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "blocksize", 0), 2);
}

TEST(SliceConcatToSpaceToDepthPattern, RejectsPermutedPhases) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 3, 8, 10));
  builder.MakeInitializer(MakeInitializer<int64_t>("ends", {2}, {8, 10}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {2}, {2, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("steps", {2}, {2, 2}));
  for (int i = 0; i < 4; ++i) {
    builder.MakeInitializer(
        MakeInitializer<int64_t>(("starts" + std::to_string(i)).c_str(), {2}, {i / 2, i % 2}));
    builder.MakeNode("Slice", {"x", "starts" + std::to_string(i), "ends", "axes", "steps"},
                     {"s" + std::to_string(i)});
  }
  builder.MakeNode("Concat", {"s1", "s0", "s2", "s3"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceConcatToSpaceToDepthPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[4]).pattern, nullptr);
}

TEST(SliceConcatToSpaceToDepthPattern, RejectsSpatialCrop) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 3, 8, 10));
  builder.MakeInitializer(MakeInitializer<int64_t>("ends", {2}, {6, 10}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {2}, {2, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("steps", {2}, {2, 2}));
  for (int i = 0; i < 4; ++i) {
    builder.MakeInitializer(
        MakeInitializer<int64_t>(("starts" + std::to_string(i)).c_str(), {2}, {i / 2, i % 2}));
    builder.MakeNode("Slice", {"x", "starts" + std::to_string(i), "ends", "axes", "steps"},
                     {"s" + std::to_string(i)});
  }
  builder.MakeNode("Concat", {"s0", "s1", "s2", "s3"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SliceConcatToSpaceToDepthPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[4]).pattern, nullptr);
}

TEST(ConcatGatherPattern, RewritesToNarrowerGather) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kInt64, Shape1D(2));
  builder.MakeInput("b", core::symbolic::TensorType::kInt64, Shape1D(3));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {3}));
  builder.MakeNode("Concat", {"a", "b"}, {"c"}, "", "", AxisAttrs(0));
  builder.MakeNode("Gather", {"c", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatGatherPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "b");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(ConcatGatherPattern, RewritesToIdentityForSingletonInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kInt64, Shape1D(1));
  builder.MakeInput("b", core::symbolic::TensorType::kInt64, Shape1D(3));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeNode("Concat", {"a", "b"}, {"c"}, "", "", AxisAttrs(0));
  builder.MakeNode("Gather", {"c", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatGatherPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "a");
}

TEST(ConcatTwiceUnaryPattern, PushesUnaryThroughConcat) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeNode("Concat", {"x", "x"}, {"c"}, "", "", AxisAttrs(0));
  builder.MakeNode("Sin", {"c"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatTwiceUnaryPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Sin");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Concat");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(ConcatTwiceUnaryPattern, RejectsOldOpset) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 17);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeNode("Concat", {"x", "x"}, {"c"}, "", "", AxisAttrs(0));
  builder.MakeNode("Sin", {"c"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatTwiceUnaryPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

} // namespace
} // namespace Test
