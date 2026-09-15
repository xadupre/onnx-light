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

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const TensorProto &initializer : builder.Initializers()) {
    if (initializer.name().value() == name) {
      return &initializer;
    }
  }
  return nullptr;
}

template <typename Pattern> void OptimizeWith(core::builder::GraphBuilder &builder) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<Pattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();
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

TEST(ConvAddFusionPattern, FoldsBiasAndLeavesActivation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {0.5f, -0.5f}));
  builder.MakeInitializer(MakeInitializer<float>("add", {1, 2, 1, 1}, {1.0f, 2.0f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"conv"});
  builder.MakeNode("Add", {"add", "conv"}, {"added"});
  builder.MakeNode("Relu", {"added"}, {"y"});
  builder.MakeOutput("y");

  OptimizeWith<onnx_patterns::ConvAddFusionPattern>(builder);

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Conv");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "added");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Relu");
  const TensorProto *folded_bias = FindInitializer(builder, builder.Nodes()[0].input()[2].value());
  ASSERT_NE(folded_bias, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*folded_bias, values));
  EXPECT_EQ(values, (std::vector<double>{1.5, 1.5}));
}

TEST(ConvAddFusionPattern, RejectsSharedConvOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("add", {2, 1, 1}, {1.0f, 2.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("Add", {"conv", "add"}, {"y"});
  builder.MakeNode("Identity", {"conv"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvAddFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvAddFusionPattern, RejectsNonChannelBroadcast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("add", {2}, {1.0f, 2.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("Add", {"conv", "add"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvAddFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvMulFusionPattern, FoldsScalarIntoDoubleWeightsAndBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<double>("w", {2, 1, 1, 1}, {2.0, 3.0}));
  builder.MakeInitializer(MakeInitializer<double>("b", {2}, {0.5, -0.5}));
  builder.MakeInitializer(MakeInitializer<double>("scale", {}, {4.0}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"conv"});
  builder.MakeNode("Mul", {"conv", "scale"}, {"y"});
  builder.MakeOutput("y");

  OptimizeWith<onnx_patterns::ConvMulFusionPattern>(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &conv = builder.Nodes()[0];
  const TensorProto *folded_weights = FindInitializer(builder, conv.input()[1].value());
  const TensorProto *folded_bias = FindInitializer(builder, conv.input()[2].value());
  ASSERT_NE(folded_weights, nullptr);
  ASSERT_NE(folded_bias, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*folded_weights, values));
  EXPECT_EQ(values, (std::vector<double>{8.0, 12.0}));
  ASSERT_TRUE(ReadFloatingValues(*folded_bias, values));
  EXPECT_EQ(values, (std::vector<double>{2.0, -2.0}));
}

TEST(ConvMulFusionPattern, RejectsMismatchedDtype) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<double>("scale", {}, {4.0}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("Mul", {"conv", "scale"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvMulFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvBatchNormalizationFusionPattern, FoldsParameters) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 14);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {0.5f, -0.5f}));
  builder.MakeInitializer(MakeInitializer<float>("scale", {2}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("bn_b", {2}, {1.0f, -1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("mean", {2}, {0.5f, 1.5f}));
  builder.MakeInitializer(MakeInitializer<float>("var", {2}, {4.0f, 9.0f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"conv"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &epsilon = attributes.add();
  epsilon.set_name("epsilon");
  epsilon.set_type(AttributeProto::AttributeType::FLOAT);
  epsilon.set_f(0.0f);
  builder.MakeNode("BatchNormalization", {"conv", "scale", "bn_b", "mean", "var"}, {"y", "", ""},
                   "", "", attributes);
  builder.MakeOutput("y");

  {
    core::builder::GraphGraph graph(builder);
    onnx_patterns::ConvBatchNormalizationFusionPattern pattern;
    const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
    ASSERT_NE(match.pattern, nullptr)
        << (match.no_match.has_value() ? match.no_match->reason : "no diagnostic");
  }
  OptimizeWith<onnx_patterns::ConvBatchNormalizationFusionPattern>(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &conv = builder.Nodes()[0];
  const TensorProto *folded_weights = FindInitializer(builder, conv.input()[1].value());
  const TensorProto *folded_bias = FindInitializer(builder, conv.input()[2].value());
  ASSERT_NE(folded_weights, nullptr);
  ASSERT_NE(folded_bias, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*folded_weights, values));
  EXPECT_EQ(values, (std::vector<double>{2.0, 3.0}));
  ASSERT_TRUE(ReadFloatingValues(*folded_bias, values));
  EXPECT_EQ(values, (std::vector<double>{1.0, -3.0}));
}

TEST(ConvBatchNormalizationFusionPattern, RejectsOptionalOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 14);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 1, 1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("scale", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("bn_b", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("mean", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("var", {1}, {1.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("BatchNormalization", {"conv", "scale", "bn_b", "mean", "var"},
                   {"y", "saved_mean", ""});
  builder.MakeOutput("y");
  builder.MakeOutput("saved_mean");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBatchNormalizationFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvBatchNormalizationFusionPattern, RejectsTrainingOutputsBeforeOpset14) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 1, 1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("scale", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("bn_b", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("mean", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("var", {1}, {1.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("BatchNormalization", {"conv", "scale", "bn_b", "mean", "var"},
                   {"y", "running_mean", "running_var", "saved_mean", "saved_var"});

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBatchNormalizationFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

} // namespace
} // namespace Test
