// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/reshape/reshape_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape(const std::vector<int64_t> &dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

void AddShape(core::builder::GraphBuilder &builder, const std::string &name,
              const std::vector<int64_t> &values) {
  builder.MakeInitializer(MakeInitializerShape(name.c_str(), values));
}

NodeProto ShapeSlice(const std::string &input, const std::string &output, int64_t start,
                     int64_t end) {
  NodeProto node = MakeNode("Shape", {input}, {output});
  AddAttribute<int64_t>(node, "start", start);
  AddAttribute<int64_t>(node, "end", end);
  return node;
}

void MakeAxisZeroConcat(core::builder::GraphBuilder &builder,
                        const std::vector<std::string> &inputs, const std::string &output) {
  NodeProto node = MakeNode("Concat", inputs, {output});
  AddAttribute<int64_t>(node, "axis", 0);
  builder.MakeNode("Concat", inputs, {output}, "", "", node.attribute());
}

TEST(ConcatReshapePattern, ReplacesOneDynamicShapeElementAndRejectsForeignInput) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "prefix", {2});
  const NodeProto slice = ShapeSlice("x", "tail", 1, 2);
  builder.MakeNode("Shape", {"x"}, {"tail"}, "", "", slice.attribute());
  MakeAxisZeroConcat(builder, {"prefix", "tail"}, "target");
  builder.MakeNode("Reshape", {"x", "target"}, {"out"});
  builder.MakeOutput("out");
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConcatReshapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Concat");
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("unknown", core::symbolic::TensorType::kInt64, Shape({1}));
  AddShape(rejected, "prefix", {2});
  MakeAxisZeroConcat(rejected, {"prefix", "unknown"}, "target");
  rejected.MakeNode("Reshape", {"x", "target"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(ReshapePattern, ReplacesIdentityReshapeAndRejectsDifferentTarget) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "shape", {2, 3});
  builder.MakeNode("Reshape", {"x", "shape"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(rejected, "shape", {3, 2});
  rejected.MakeNode("Reshape", {"x", "shape"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(ShapedBasedReshapePattern, RemovesZeroPrefixReshapeAndRejectsNonzeroPrefix) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "shape", {0, -1});
  builder.MakeNode("Reshape", {"x", "shape"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapedBasedReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(rejected, "shape", {1, -1});
  rejected.MakeNode("Reshape", {"x", "shape"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(ReduceReshapePattern, RemovesKeepdimsReshapeAndRejectsKeepdimsZero) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "axes", {1});
  AddShape(builder, "shape", {2});
  builder.MakeNode("ReduceSum", {"x", "axes"}, {"reduced"});
  builder.MakeNode("Reshape", {"reduced", "shape"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReduceReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "ReduceSum");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "keepdims", 1), 0);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(rejected, "axes", {1});
  AddShape(rejected, "shape", {2});
  NodeProto reduction = MakeNode("ReduceSum", {"x", "axes"}, {"reduced"});
  AddAttribute<int64_t>(reduction, "keepdims", 0);
  rejected.MakeNode("ReduceSum", {"x", "axes"}, {"reduced"}, "", "", reduction.attribute());
  rejected.MakeNode("Reshape", {"reduced", "shape"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(ReshapeReshapePattern, KeepsSecondTargetAndRejectsSharedIntermediate) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddShape(builder, "first", {6, 4});
  AddShape(builder, "second", {2, 3, 4});
  builder.MakeNode("Reshape", {"x", "first"}, {"middle"});
  builder.MakeNode("Reshape", {"middle", "second"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReshapeReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "second");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddShape(rejected, "first", {6, 4});
  AddShape(rejected, "second", {2, 3, 4});
  rejected.MakeNode("Reshape", {"x", "first"}, {"middle"});
  rejected.MakeNode("Reshape", {"middle", "second"}, {"out"});
  rejected.MakeNode("Identity", {"middle"}, {"other"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(ReshapeReshapeBinaryPattern, MovesEqualReshapesAfterBinaryAndRejectsDifferentTargets) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "shape", {6});
  builder.MakeNode("Reshape", {"x", "shape"}, {"left"});
  builder.MakeNode("Reshape", {"y", "shape"}, {"right"});
  builder.MakeNode("Add", {"left", "right"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReshapeReshapeBinaryPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Add");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(rejected, "left_shape", {6});
  AddShape(rejected, "right_shape", {1, 6});
  rejected.MakeNode("Reshape", {"x", "left_shape"}, {"left"});
  rejected.MakeNode("Reshape", {"y", "right_shape"}, {"right"});
  rejected.MakeNode("Add", {"left", "right"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[2]).pattern, nullptr);
}

TEST(Reshape2Of3Pattern, MovesTwoInputReshapesAndRejectsOnlyOneReshape) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "shape", {6});
  builder.MakeNode("Reshape", {"x", "shape"}, {"left"});
  builder.MakeNode("Reshape", {"y", "shape"}, {"right"});
  builder.MakeNode("Add", {"left", "right"}, {"out"});
  builder.MakeOutput("out");
  core::builder::GraphGraph graph(builder);
  onnx_patterns::Reshape2Of3Pattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Add");
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({6}));
  rejected.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({6}));
  AddShape(rejected, "shape", {6});
  rejected.MakeNode("Reshape", {"x", "shape"}, {"left"});
  rejected.MakeNode("Add", {"left", "y"}, {"out"});
  rejected.MakeOutput("out");
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(StaticConcatReshapePattern, ReplacesOnlyDynamicElementAndRejectsTwoDynamicElements) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("dynamic", core::symbolic::TensorType::kInt64, Shape({1}));
  AddShape(builder, "prefix", {2});
  MakeAxisZeroConcat(builder, {"prefix", "dynamic"}, "target");
  builder.MakeNode("Reshape", {"x", "target"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::StaticConcatReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Concat");
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("first", core::symbolic::TensorType::kInt64, Shape({1}));
  rejected.MakeInput("second", core::symbolic::TensorType::kInt64, Shape({1}));
  MakeAxisZeroConcat(rejected, {"first", "second"}, "target");
  rejected.MakeNode("Reshape", {"x", "target"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(ShapeBasedEditDistanceReshapePattern, AlignsStaticShapeAndRejectsNonConcatTarget) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddShape(builder, "six", {6});
  AddShape(builder, "four", {4});
  MakeAxisZeroConcat(builder, {"six", "four"}, "target");
  builder.MakeNode("Reshape", {"x", "target"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedEditDistanceReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddShape(rejected, "target", {6, 4});
  rejected.MakeNode("Identity", {"target"}, {"not_concat"});
  rejected.MakeNode("Reshape", {"x", "not_concat"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(ShapeBasedReshapeIsSqueezePattern, ReplacesWithUnsqueezeAndRejectsUnchangedShape) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "shape", {1, 2, 3});
  builder.MakeNode("Reshape", {"x", "shape"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedReshapeIsSqueezePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(rejected, "shape", {2, 3});
  rejected.MakeNode("Reshape", {"x", "shape"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(UnsqueezeReshapePattern, MovesSpecialUnsqueezeAxisAndRejectsOtherAxes) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddShape(builder, "axes", {2});
  AddShape(builder, "shape", {0, 1, -1, 0});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"unsqueezed"});
  builder.MakeNode("Reshape", {"unsqueezed", "shape"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddShape(rejected, "axes", {1});
  AddShape(rejected, "shape", {0, 1, -1, 0});
  rejected.MakeNode("Unsqueeze", {"x", "axes"}, {"unsqueezed"});
  rejected.MakeNode("Reshape", {"unsqueezed", "shape"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(UnsqueezeOrSqueezeReshapePattern, RemovesUnaryShapeEditAndRejectsUnsafeZeroCopy) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(builder, "axes", {0});
  AddShape(builder, "shape", {6});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"unsqueezed"});
  builder.MakeNode("Reshape", {"unsqueezed", "shape"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeOrSqueezeReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddShape(rejected, "axes", {0});
  AddShape(rejected, "shape", {0, 6});
  rejected.MakeNode("Unsqueeze", {"x", "axes"}, {"unsqueezed"});
  rejected.MakeNode("Reshape", {"unsqueezed", "shape"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(ReshapeSqueezePattern, RemovesInsertedUnitDimensionAndRejectsNonUnitAxis) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4, 8}));
  AddShape(builder, "shape", {0, 0, 0, 1, 8});
  AddShape(builder, "axes", {3});
  builder.MakeNode("Reshape", {"x", "shape"}, {"reshaped"});
  builder.MakeNode("Squeeze", {"reshaped", "axes"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReshapeSqueezePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4, 8}));
  AddShape(rejected, "shape", {0, 1, 0, 1, 8});
  AddShape(rejected, "axes", {1});
  rejected.MakeNode("Reshape", {"x", "shape"}, {"reshaped"});
  rejected.MakeNode("Squeeze", {"reshaped", "axes"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

} // namespace
} // namespace Test
