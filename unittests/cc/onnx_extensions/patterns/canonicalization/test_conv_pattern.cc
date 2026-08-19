// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>
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

core::symbolic::SymShape Shape4D(int64_t n, int64_t c, int64_t h, int64_t w) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(n));
  shape.PushBack(core::symbolic::SymDim(c));
  shape.PushBack(core::symbolic::SymDim(h));
  shape.PushBack(core::symbolic::SymDim(w));
  return shape;
}

const AttributeProto *FindAttr(const NodeProto &node, const std::string &name) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name().value() == name) {
      return &attribute;
    }
  }
  return nullptr;
}

TEST(ConvBiasNullPattern, RemovesZeroBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {0.0f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBiasNullPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Conv");
  EXPECT_EQ(replacements[0].input_size(), 2);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "w");
}

TEST(ConvBiasNullPattern, RejectsNonZeroBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {0.5f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBiasNullPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Conv bias is not a known all-zero constant");
}

TEST(ConvBiasNullPattern, RemovesShapeExpandedZeroBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape4D(2, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {0.0f, 0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  utils::RepeatedProtoField<AttributeProto> shape_attributes;
  AttributeProto &start = shape_attributes.add();
  start.set_name("start");
  start.set_type(AttributeProto::AttributeType::INT);
  start.set_i(0);
  AttributeProto &end = shape_attributes.add();
  end.set_name("end");
  end.set_type(AttributeProto::AttributeType::INT);
  end.set_i(1);
  builder.MakeNode("Shape", {"b"}, {"b_shape"}, "", "", shape_attributes);
  builder.MakeNode("Expand", {"zero", "b_shape"}, {"expanded_bias"});
  builder.MakeNode("Conv", {"x", "w", "expanded_bias"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::ConvBiasNullPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Conv");
  ASSERT_EQ(builder.Nodes()[0].input_size(), 2);
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].input()[1].value(), "w");
}

TEST(ConvBiasNullPattern, RejectsExpandedZeroWithDynamicShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape4D(2, 1, 2, 2));
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  builder.MakeInput("bias_shape", core::symbolic::TensorType::kInt64, shape);
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Expand", {"zero", "bias_shape"}, {"expanded_bias"});
  builder.MakeNode("Conv", {"x", "w", "expanded_bias"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBiasNullPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Conv bias is not a constant");
}

TEST(PadConvPattern, FoldsSpatialPadding) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<int64_t>("pads", {8}, {0, 0, 1, 1, 0, 0, 1, 1}));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeNode("Pad", {"x", "pads"}, {"padded"});
  builder.MakeNode("Conv", {"padded", "w"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadConvPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Conv");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  const AttributeProto *pads = FindAttr(replacements[0], "pads");
  ASSERT_NE(pads, nullptr);
  ASSERT_EQ(pads->ints().size(), 4);
  EXPECT_EQ(pads->ints()[0], 1);
  EXPECT_EQ(pads->ints()[1], 1);
  EXPECT_EQ(pads->ints()[2], 1);
  EXPECT_EQ(pads->ints()[3], 1);
}

TEST(PadConvPattern, RejectsChannelPadding) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<int64_t>("pads", {8}, {0, 1, 1, 1, 0, 1, 1, 1}));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeNode("Pad", {"x", "pads"}, {"padded"});
  builder.MakeNode("Conv", {"padded", "w"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadConvPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Pad node pads the batch or channel dimension");
}

} // namespace
} // namespace Test
