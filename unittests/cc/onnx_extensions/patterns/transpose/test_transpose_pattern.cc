// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_extensions/patterns/transpose/transpose_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

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
  for (int64_t d : dims) {
    shape.PushBack(core::symbolic::SymDim(d));
  }
  return shape;
}

utils::RepeatedProtoField<AttributeProto> PermAttr(const std::vector<int64_t> &perm) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("perm");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : perm) {
    attribute.ints().push_back(v);
  }
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> AxisAttr(int64_t axis) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("axis");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(axis);
  return attributes;
}

std::vector<int64_t> AttributeInts(const NodeProto &node, const char *name) {
  std::vector<int64_t> values;
  GetAttributeInts(node, name, values);
  return values;
}

TEST(TransposeToInitializerPattern, FoldsFloatInitializer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(
      MakeInitializer<float>("weight", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  builder.MakeNode("Transpose", {"weight"}, {"weight_t"}, "", "transpose_weight", PermAttr({1, 0}));
  builder.MakeOutput("weight_t");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::TransposeToInitializerPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  EXPECT_TRUE(builder.Nodes().empty());
  ASSERT_EQ(builder.Initializers().size(), 2u);
  const TensorProto &folded = builder.Initializers()[1];
  EXPECT_EQ(folded.name().value(), "weight_t");
  EXPECT_EQ(folded.data_type(), TensorProto::DataType::FLOAT);
  ASSERT_EQ(folded.dims().size(), 2u);
  EXPECT_EQ(folded.dims()[0], 3);
  EXPECT_EQ(folded.dims()[1], 2);
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(folded);
  ASSERT_EQ(tensor.element_count(), 6);
  const float *values = tensor.AsFloat();
  EXPECT_EQ(std::vector<float>(values, values + 6),
            (std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
}

TEST(TransposeToInitializerPattern, UsesDefaultReversePermutation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<int64_t>("weight", {2, 1, 3}, {1, 2, 3, 4, 5, 6}));
  builder.MakeNode("Transpose", {"weight"}, {"weight_t"});
  builder.MakeOutput("weight_t");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::TransposeToInitializerPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  EXPECT_TRUE(builder.Nodes().empty());
  ASSERT_EQ(builder.Initializers().size(), 2u);
  const TensorProto &folded = builder.Initializers()[1];
  EXPECT_EQ(folded.data_type(), TensorProto::DataType::INT64);
  ASSERT_EQ(folded.dims().size(), 3u);
  EXPECT_EQ(folded.dims()[0], 3);
  EXPECT_EQ(folded.dims()[1], 1);
  EXPECT_EQ(folded.dims()[2], 2);
}

TEST(TransposeToInitializerPattern, RejectsRuntimeInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeNode("Transpose", {"x"}, {"y"}, "", "", PermAttr({1, 0}));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeToInitializerPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(TransposeToInitializerPattern, RejectsConstantNodeOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &value = attributes.add();
  value.set_name("value");
  value.set_type(AttributeProto::AttributeType::TENSOR);
  *value.mutable_t() = MakeInitializer<float>("", {2, 3}, {1, 2, 3, 4, 5, 6});
  builder.MakeNode("Constant", {}, {"weight"}, "", "", attributes);
  builder.MakeNode("Transpose", {"weight"}, {"weight_t"}, "", "", PermAttr({1, 0}));
  builder.MakeOutput("weight_t");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeToInitializerPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(TransposeTransposePattern, CancelsOppositeTransposes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeNode("Transpose", {"x"}, {"t0"}, "", "", PermAttr({1, 0}));
  builder.MakeNode("Transpose", {"t0"}, {"out"}, "", "", PermAttr({1, 0}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
}

TEST(TransposeTransposePattern, MergesIntoSingleTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  builder.MakeNode("Transpose", {"x"}, {"t0"}, "", "", PermAttr({1, 0, 2}));
  builder.MakeNode("Transpose", {"t0"}, {"out"}, "", "", PermAttr({0, 2, 1}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
  EXPECT_EQ(AttributeInts(replacements[0], "perm"), (std::vector<int64_t>{1, 2, 0}));
}

TEST(TransposeTransposePattern, KeepsSharedTransposeWhenCancelling) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeNode("Transpose", {"x"}, {"t0"}, "", "", PermAttr({1, 0}));
  builder.MakeNode("Transpose", {"t0"}, {"out"}, "", "", PermAttr({1, 0}));
  builder.MakeNode("Identity", {"t0"}, {"other"});
  builder.MakeOutput("out");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[0].output()[0].value(), "t0");
  EXPECT_EQ(replacements[1].op_type().value(), "Identity");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(TransposeTransposePattern, RejectsSharedTransposeWhenMerging) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  builder.MakeNode("Transpose", {"x"}, {"t0"}, "", "", PermAttr({1, 0, 2}));
  builder.MakeNode("Transpose", {"t0"}, {"out"}, "", "", PermAttr({0, 2, 1}));
  builder.MakeNode("Identity", {"t0"}, {"other"});
  builder.MakeOutput("out");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(TransposeGatherPattern, DropsTransposeWhenOrderKept) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 4, 5, 6}));
  builder.MakeInput("ind", core::symbolic::TensorType::kInt64, Shape({}));
  builder.MakeNode("Transpose", {"x"}, {"t"}, "", "", PermAttr({1, 0, 2, 3}));
  builder.MakeNode("Gather", {"t", "ind"}, {"out"}, "", "", AxisAttr(0));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeGatherPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "ind");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", 0), 1);
}

TEST(TransposeGatherPattern, SwapsTransposePastGather) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 4, 5}));
  builder.MakeInput("ind", core::symbolic::TensorType::kInt64, Shape({}));
  builder.MakeNode("Transpose", {"x"}, {"t"}, "", "", PermAttr({2, 1, 0}));
  builder.MakeNode("Gather", {"t", "ind"}, {"out"}, "", "", AxisAttr(1));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeGatherPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gather");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", 0), 1);
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
  EXPECT_EQ(AttributeInts(replacements[1], "perm"), (std::vector<int64_t>{1, 0}));
}

TEST(TransposeGatherPattern, RejectsNonScalarIndex) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 4, 5, 6}));
  builder.MakeInput("ind", core::symbolic::TensorType::kInt64, Shape({2}));
  builder.MakeNode("Transpose", {"x"}, {"t"}, "", "", PermAttr({1, 0, 2, 3}));
  builder.MakeNode("Gather", {"t", "ind"}, {"out"}, "", "", AxisAttr(0));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeGatherPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

} // namespace
} // namespace Test
