// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/gather_pattern.h"
#include "onnx_extensions/patterns/collections/slice_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>
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

core::symbolic::SymShape ShapeND(const std::vector<int64_t> &dims) {
  core::symbolic::SymShape shape;
  for (int64_t d : dims) {
    shape.PushBack(core::symbolic::SymDim(d));
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

utils::RepeatedProtoField<AttributeProto> PermAttrs(const std::vector<int64_t> &perm) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attr = attributes.add();
  attr.set_name("perm");
  attr.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : perm) {
    attr.ints().push_back(v);
  }
  return attributes;
}

std::vector<int64_t> AttributeInts(const NodeProto &node, const char *name) {
  std::vector<int64_t> values;
  GetAttributeInts(node, name, values);
  return values;
}

int64_t AxisAttribute(const NodeProto &node) { return GetAttributeOr<int64_t>(node, "axis", -999); }

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const auto &tensor : builder.Initializers()) {
    if (tensor.name().value() == name) {
      return &tensor;
    }
  }
  return nullptr;
}

TEST(GatherConcatPattern, ShiftsIndexOntoNonConstantInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape1D(3));
  builder.MakeInitializer(MakeInitializer<int64_t>("pre", {2}, {10, 11}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {2, 3}));
  builder.MakeNode("Concat", {"pre", "x"}, {"c"}, "", "", AxisAttrs(0));
  builder.MakeNode("Gather", {"c", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherConcatPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  const TensorProto *shifted = FindInitializer(builder, replacements[0].input()[1].value());
  ASSERT_NE(shifted, nullptr);
  std::vector<int64_t> values;
  ASSERT_TRUE(ReadIntegerValues(*shifted, values));
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], 0);
  EXPECT_EQ(values[1], 1);
}

TEST(GatherConcatPattern, RejectsIndexBeforeNonConstant) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape1D(3));
  builder.MakeInitializer(MakeInitializer<int64_t>("pre", {2}, {10, 11}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeNode("Concat", {"pre", "x"}, {"c"}, "", "", AxisAttrs(0));
  builder.MakeNode("Gather", {"c", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherConcatPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherGatherPattern, ComposesConstantIndices) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("i1", {3}, {4, 2, 0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i2", {2}, {2, 1}));
  builder.MakeNode("Gather", {"x", "i1"}, {"g"});
  builder.MakeNode("Gather", {"g", "i2"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherGatherPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  const TensorProto *fused = FindInitializer(builder, replacements[0].input()[1].value());
  ASSERT_NE(fused, nullptr);
  std::vector<int64_t> values;
  ASSERT_TRUE(ReadIntegerValues(*fused, values));
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], 0);
  EXPECT_EQ(values[1], 2);
}

TEST(GatherToSlicePattern, ScalarIndexEmitsSliceAndSqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {2}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Slice");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  ASSERT_EQ(replacements[0].input_size(), 4);
  const TensorProto *starts = FindInitializer(builder, replacements[0].input()[1].value());
  const TensorProto *ends = FindInitializer(builder, replacements[0].input()[2].value());
  const TensorProto *axes = FindInitializer(builder, replacements[0].input()[3].value());
  ASSERT_NE(starts, nullptr);
  ASSERT_NE(ends, nullptr);
  ASSERT_NE(axes, nullptr);
  std::vector<int64_t> values;
  ASSERT_TRUE(ReadIntegerValues(*starts, values));
  EXPECT_EQ(values, (std::vector<int64_t>{2}));
  ASSERT_TRUE(ReadIntegerValues(*ends, values));
  EXPECT_EQ(values, (std::vector<int64_t>{3}));
  ASSERT_TRUE(ReadIntegerValues(*axes, values));
  EXPECT_EQ(values, (std::vector<int64_t>{0}));
  // The Slice output must be a fresh internal name, not the original "y".
  EXPECT_NE(replacements[0].output()[0].value(), "y");

  EXPECT_EQ(replacements[1].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
  ASSERT_EQ(replacements[1].input_size(), 2);
  const TensorProto *squeeze_axes = FindInitializer(builder, replacements[1].input()[1].value());
  ASSERT_NE(squeeze_axes, nullptr);
  ASSERT_TRUE(ReadIntegerValues(*squeeze_axes, values));
  EXPECT_EQ(values, (std::vector<int64_t>{0}));
}

TEST(GatherToSlicePattern, VectorSingletonMapsDirectlyToSlice) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {3}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Slice");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  std::vector<int64_t> values;
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[1].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{3}));
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[2].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{4}));
}

TEST(GatherToSlicePattern, ArithmeticRangeWithStepMapsToStridedSlice) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(10));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {3}, {2, 4, 6}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  ASSERT_EQ(replacements[0].input_size(), 5);
  std::vector<int64_t> values;
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[1].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{2}));
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[2].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{8}));
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[4].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{2}));
}

TEST(GatherToSlicePattern, NormalizesNegativeAxisAndIndex) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(4, 5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {-1}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"}, "", "", AxisAttrs(-1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  std::vector<int64_t> values;
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[1].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{4}));
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[2].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{5}));
  ASSERT_TRUE(
      ReadIntegerValues(*FindInitializer(builder, replacements[0].input()[3].value()), values));
  EXPECT_EQ(values, (std::vector<int64_t>{1}));
}

TEST(GatherToSlicePattern, AttributeFormBeforeOpset10) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 9);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {1}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Slice");
  ASSERT_EQ(replacements[0].input_size(), 1);
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> axes;
  ASSERT_TRUE(GetAttributeInts(replacements[0], "starts", starts));
  ASSERT_TRUE(GetAttributeInts(replacements[0], "ends", ends));
  ASSERT_TRUE(GetAttributeInts(replacements[0], "axes", axes));
  EXPECT_EQ(starts, (std::vector<int64_t>{1}));
  EXPECT_EQ(ends, (std::vector<int64_t>{2}));
  EXPECT_EQ(axes, (std::vector<int64_t>{0}));

  EXPECT_EQ(replacements[1].op_type().value(), "Squeeze");
  ASSERT_EQ(replacements[1].input_size(), 1);
  std::vector<int64_t> squeeze_axes;
  ASSERT_TRUE(GetAttributeInts(replacements[1], "axes", squeeze_axes));
  EXPECT_EQ(squeeze_axes, (std::vector<int64_t>{0}));
}

TEST(GatherToSlicePattern, RejectsNonConstantIndex) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInput("idx", core::symbolic::TensorType::kInt64, core::symbolic::SymShape());
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Gather index is not constant");
}

TEST(GatherToSlicePattern, RejectsNegativeIndexWithoutKnownDimension) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim("N"));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {-1}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a negative index requires a statically known axis dimension");
}

TEST(GatherToSlicePattern, RejectsNonArithmeticIndices) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {3}, {0, 2, 5}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason,
            "the Gather indices are not a constant-step arithmetic progression");
}

TEST(GatherToSlicePattern, RejectsDescendingIndices) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {3, 1}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Gather indices are not a strictly ascending progression");
}

TEST(GatherToSlicePattern, RejectsNonUnitStepBeforeOpset10) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 9);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(10));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {3}, {2, 4, 6}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a non-unit step requires the Slice input form (opset >= 10)");
}

TEST(GatherToSlicePattern, RejectsMultiDimensionalIndex) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1, 1}, {2}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherToSlicePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Gather index is not a constant scalar or vector");
}

TEST(GatherToSlicePattern, OptimizerConvergesFullRangeGatherToIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape1D(5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {5}, {0, 1, 2, 3, 4}));
  builder.MakeNode("Gather", {"x", "idx"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::GatherToSlicePattern>());
  patterns.push_back(std::make_unique<onnx_patterns::SliceEliminationPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  EXPECT_FALSE(graph.Optimize().empty());
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");

  // A second optimization pass finds nothing left to rewrite (converged).
  EXPECT_TRUE(graph.Optimize().empty());
}

// --- GatherUpstreamPropagationPattern -------------------------------------

TEST(GatherUpstreamPropagationPattern, PropagatesThroughAddSkippingLowerRankBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, ShapeND({4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 2}));
  builder.MakeNode("Add", {"x", "b"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(AxisAttribute(replacements[0]), 1);
  EXPECT_EQ(replacements[1].op_type().value(), "Add");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "b");
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughAddSkippingBroadcastDim) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, ShapeND({1, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 1}));
  builder.MakeNode("Add", {"x", "b"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "b");
}

TEST(GatherUpstreamPropagationPattern, RejectsScalarIndexWithBroadcastDim) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, ShapeND({1, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {0}));
  builder.MakeNode("Add", {"x", "b"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, RejectsSharedProducerOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, ShapeND({4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 2}));
  builder.MakeNode("Add", {"x", "b"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeNode("Identity", {"t"}, {"z"});
  builder.MakeOutput("y");
  builder.MakeOutput("z");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughTransposeVectorIndex) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 3}));
  builder.MakeNode("Transpose", {"x"}, {"t"}, "", "", PermAttrs({2, 0, 1}));
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(AxisAttribute(replacements[0]), 2);
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(AttributeInts(replacements[1], "perm"), (std::vector<int64_t>{2, 0, 1}));
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(GatherUpstreamPropagationPattern, RejectsScalarIndexAfterTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {0}));
  builder.MakeNode("Transpose", {"x"}, {"t"}, "", "", PermAttrs({2, 0, 1}));
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughReshapeKeepingZeroMarker) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 2}));
  builder.MakeInitializer(MakeInitializerShape("shape", {0, 0, 4}));
  builder.MakeNode("Reshape", {"x", "shape"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(AxisAttribute(replacements[0]), 1);
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "shape");
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughReshapeUpdatingConcreteEntry) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 1}));
  builder.MakeInitializer(MakeInitializerShape("shape", {2, 3, 4}));
  builder.MakeNode("Reshape", {"x", "shape"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  const std::string &new_shape_name = replacements[1].input()[1].value();
  EXPECT_NE(new_shape_name, "shape");
  const TensorProto *new_shape = FindInitializer(builder, new_shape_name);
  ASSERT_NE(new_shape, nullptr);
  std::vector<int64_t> new_shape_values;
  ASSERT_TRUE(ReadIntegerValues(*new_shape, new_shape_values));
  EXPECT_EQ(new_shape_values, (std::vector<int64_t>{2, 2, 4}));
}

TEST(GatherUpstreamPropagationPattern, RejectsScalarIndexOnConcreteReshapeEntry) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {0}));
  builder.MakeInitializer(MakeInitializerShape("shape", {2, 3, 4}));
  builder.MakeNode("Reshape", {"x", "shape"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, RejectsCoincidentalMiddleReshapeDimension) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({4, 3, 5}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 1}));
  builder.MakeInitializer(MakeInitializerShape("shape", {2, 3, 10}));
  builder.MakeNode("Reshape", {"x", "shape"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, RejectsReshapeAxisMovedByRankChange) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 3, 7}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeInitializer(MakeInitializerShape("shape", {6, 0, 7}));
  builder.MakeNode("Reshape", {"x", "shape"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, RejectsReshapeWithZeroSizedBlock) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({0, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeInitializer(MakeInitializerShape("shape", {0, 0, 3}));
  builder.MakeNode("Reshape", {"x", "shape"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(2));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughMatMulBatchDimension) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, ShapeND({8, 2, 3}));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, ShapeND({8, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 2}));
  builder.MakeNode("MatMul", {"a", "b"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "a");
  EXPECT_EQ(replacements[1].op_type().value(), "Gather");
  EXPECT_EQ(replacements[1].input()[0].value(), "b");
  EXPECT_EQ(replacements[2].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[2].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[2].input()[1].value(), replacements[1].output()[0].value());
  EXPECT_EQ(replacements[2].output()[0].value(), "y");
}

TEST(GatherUpstreamPropagationPattern, RejectsMatMulContractedAdjacentAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, ShapeND({8, 2, 3}));
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, ShapeND({8, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeNode("MatMul", {"a", "b"}, {"t"});
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughSoftmaxBeforeReductionAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeNode("Softmax", {"x"}, {"t"}, "", "", AxisAttrs(2));
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(AxisAttribute(replacements[0]), 0);
  EXPECT_EQ(replacements[1].op_type().value(), "Softmax");
  EXPECT_EQ(AxisAttribute(replacements[1]), 2);
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(GatherUpstreamPropagationPattern, RejectsSoftmaxOnReductionAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeNode("Softmax", {"x"}, {"t"}, "", "", AxisAttrs(2));
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(2));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(GatherUpstreamPropagationPattern, PropagatesThroughLayerNormalizationDataInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInput("scale", core::symbolic::TensorType::kFloat, ShapeND({4}));
  builder.MakeInput("bias", core::symbolic::TensorType::kFloat, ShapeND({4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {1}, {0}));
  builder.MakeNode("LayerNormalization", {"x", "scale", "bias"}, {"t", "mean", "std"}, "", "",
                   AxisAttrs(-1));
  builder.MakeNode("Gather", {"t", "idx"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherUpstreamPropagationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "LayerNormalization");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "scale");
  EXPECT_EQ(replacements[1].input()[2].value(), "bias");
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(GatherUpstreamPropagationPattern, OptimizesChainOfAddsToConvergence) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ShapeND({2, 3, 4}));
  builder.MakeInput("b1", core::symbolic::TensorType::kFloat, ShapeND({4}));
  builder.MakeInput("b2", core::symbolic::TensorType::kFloat, ShapeND({4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 2}));
  builder.MakeNode("Add", {"x", "b1"}, {"t1"});
  builder.MakeNode("Add", {"t1", "b2"}, {"t2"});
  builder.MakeNode("Gather", {"t2", "idx"}, {"y"}, "", "", AxisAttrs(1));
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::GatherUpstreamPropagationPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize();
  EXPECT_FALSE(rewrites.empty());

  // The final graph output is still "y", produced by the outer Add.
  ASSERT_GE(builder.Nodes().size(), 1u);
  const NodeProto *final_node = nullptr;
  for (const auto &node : builder.Nodes()) {
    if (!node.output().empty() && node.output()[0].value() == "y") {
      final_node = &node;
    }
  }
  ASSERT_NE(final_node, nullptr);
  EXPECT_EQ(final_node->op_type().value(), "Add");

  // No Add node still consumes the original, full-size "x" or "t1"/"t2"
  // directly with a Gather still downstream: the pattern must have moved all
  // the way to the graph inputs, and re-optimizing must be a no-op.
  for (const auto &node : builder.Nodes()) {
    if (node.op_type().value() == "Gather") {
      const std::string &data = node.input()[0].value();
      EXPECT_TRUE(data == "x" || data == "b1" || data == "b2");
    }
  }
  EXPECT_TRUE(graph.Optimize().empty());
}

} // namespace
} // namespace Test
