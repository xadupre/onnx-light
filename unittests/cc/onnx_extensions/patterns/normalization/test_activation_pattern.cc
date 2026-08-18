// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/patterns/normalization/activation_pattern.h"
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

core::symbolic::SymShape Shape(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

utils::RepeatedProtoField<AttributeProto>
IntAttributes(std::initializer_list<std::pair<const char *, int64_t>> values) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &[name, value] : values) {
    AttributeProto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::INT);
    attribute.set_i(value);
  }
  return attributes;
}

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType type) {
  builder.MakeNode("Cast", {input}, {output}, "", "",
                   IntAttributes({{"to", static_cast<int64_t>(type)}}));
}

void AddFloatInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                         const std::vector<int64_t> &dims, const std::vector<float> &values) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dims, values));
}

void AddDoubleInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                          double value) {
  builder.MakeInitializer(MakeInitializer<double>(name.c_str(), {1}, {value}));
}

void AddFloat16Initializer(core::builder::GraphBuilder &builder, const std::string &name,
                           float value) {
  builder.MakeInitializer(
      MakeInitializer<uint16_t>(name.c_str(), {1}, {core::runtime::FloatToFloat16Bits(value)}));
}

void AddSubgraphReference(core::builder::GraphBuilder &builder, const std::string &subgraph_name,
                          const std::string &output) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("body_ref");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s(subgraph_name);
  builder.MakeNode("SubgraphCarrier", {}, {output}, "", "", attributes);
}

void SetNodeDomain(core::builder::GraphBuilder &builder, std::size_t index,
                   const std::string &domain) {
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[index].set_domain(domain);
}

void SetValueType(core::builder::GraphBuilder &builder, const std::string &name,
                  core::symbolic::TensorType type, const core::symbolic::SymShape &shape) {
  auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  shapes.Set(name, core::symbolic::SymTensor(nullptr, type, shape));
}

TEST(GeluPattern, FusesExactTanhDecomposition) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.7978515625);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"}, "", "gelu_decomposition");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern(0, 20, "com.microsoft");
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 8u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[7], &builder.Nodes()[7]);
  EXPECT_EQ(match.insert_at, nullptr);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gelu");
  EXPECT_EQ(replacements[0].domain().value(), "com.microsoft");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<std::string>(replacements[0], "approximate", ""), "tanh");
}

TEST(GeluPattern, RejectsWrongTanhScale) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.8);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "one Gelu scalar constant has an unexpected value");
}

TEST(GeluPattern, RejectsIntermediateGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.7978515625);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"});
  builder.MakeOutput("x3");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an intermediate Gelu result is externally used");
}

TEST(GeluPattern, RejectsNonOnnxIntermediateDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.SetOpsetVersion("custom", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.7978515625);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"});
  builder.MakeOutput("y");
  SetNodeDomain(builder, 4, "custom");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the tanh-based Gelu topology is incomplete");
}

TEST(LeakyReluPattern, FusesWhereDecomposition) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 3}));
  AddFloatInitializer(builder, "zero", {1}, {0.0F});
  AddFloatInitializer(builder, "slope", {1}, {-0.33F});
  builder.MakeNode("Greater", {"x", "zero"}, {"positive"});
  builder.MakeNode("Mul", {"x", "slope"}, {"negative"});
  builder.MakeNode("Where", {"positive", "x", "negative"}, {"y"}, "", "where");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LeakyReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.insert_at, nullptr);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "LeakyRelu");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[0], "alpha", 0.0F), -0.33F);
}

TEST(LeakyReluPattern, RejectsNonzeroThreshold) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 3}));
  AddFloatInitializer(builder, "zero", {1}, {1.0F});
  AddFloatInitializer(builder, "slope", {1}, {-0.33F});
  builder.MakeNode("Greater", {"x", "zero"}, {"positive"});
  builder.MakeNode("Mul", {"x", "slope"}, {"negative"});
  builder.MakeNode("Where", {"positive", "x", "negative"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LeakyReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the threshold is not zero or the slope is not scalar");
}

TEST(LeakyReluPattern, RejectsSubgraphCapture) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 3}));
  AddFloatInitializer(builder, "zero", {1}, {0.0F});
  AddFloatInitializer(builder, "slope", {1}, {-0.33F});
  builder.MakeNode("Greater", {"x", "zero"}, {"positive"});
  builder.MakeNode("Mul", {"x", "slope"}, {"negative"});
  builder.MakeNode("Where", {"positive", "x", "negative"}, {"y"});
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"positive"}, {"captured"});
  body.MakeOutput("captured");
  AddSubgraphReference(builder, "body", "body_result");
  builder.MakeOutput("y");
  builder.MakeOutput("body_result");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LeakyReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a LeakyRelu intermediate result is externally used");
}

void BuildSoftmaxCrossEntropyGraph(
    core::builder::GraphBuilder &builder, int64_t squeeze_axis,
    core::symbolic::TensorType score_type = core::symbolic::TensorType::kFloat16,
    int64_t ignored_index_branch = 0, float ignored_loss_branch = 0.0F) {
  builder.SetOpsetVersion("", 14);
  builder.MakeInput("x", score_type, Shape({3, 5}));
  builder.MakeInput("indices", core::symbolic::TensorType::kInt64, Shape({3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("ignore", {1}, {-100}));
  builder.MakeInitializer(MakeInitializer<int64_t>("zeroi", {1}, {ignored_index_branch}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axis", {1}, {squeeze_axis}));
  if (score_type == core::symbolic::TensorType::kFloat16) {
    AddFloat16Initializer(builder, "zerof", ignored_loss_branch);
  } else {
    AddFloatInitializer(builder, "zerof", {1}, {ignored_loss_branch});
  }
  builder.MakeNode("Equal", {"indices", "ignore"}, {"ignored"});
  builder.MakeNode("Not", {"ignored"}, {"kept"});
  builder.MakeNode("Where", {"kept", "indices", "zeroi"}, {"safe_indices"});
  builder.MakeNode("Unsqueeze", {"safe_indices", "axis"}, {"indices_2d"});
  builder.MakeNode("LogSoftmax", {"x"}, {"log_probabilities"}, "", "",
                   IntAttributes({{"axis", 1}}));
  builder.MakeNode("GatherElements", {"log_probabilities", "indices_2d"}, {"selected"}, "", "",
                   IntAttributes({{"axis", 1}}));
  builder.MakeNode("Squeeze", {"selected", "axis"}, {"selected_1d"});
  builder.MakeNode("Neg", {"selected_1d"}, {"losses"});
  builder.MakeNode("Where", {"kept", "losses", "zerof"}, {"masked_losses"});
  AddCast(builder, "kept", "kept_float", TensorProto::DataType::FLOAT);
  builder.MakeNode("ReduceSum", {"kept_float"}, {"denominator_float"}, "", "",
                   IntAttributes({{"keepdims", 0}}));
  AddCast(builder, "denominator_float", "denominator", TensorProto::DataType::FLOAT16);
  AddCast(builder, "masked_losses", "losses_float", TensorProto::DataType::FLOAT);
  builder.MakeNode("ReduceSum", {"losses_float"}, {"numerator_float"}, "", "",
                   IntAttributes({{"keepdims", 0}}));
  AddCast(builder, "numerator_float", "numerator", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Div", {"numerator", "denominator"}, {"y"}, "", "loss");
  builder.MakeOutput("y");
}

TEST(SoftmaxCrossEntropyLossCastPattern, FusesExactMaskedMeanLoss) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 16u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[15], &builder.Nodes()[15]);
  EXPECT_EQ(match.insert_at, nullptr);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "SoftmaxCrossEntropyLoss");
  EXPECT_EQ(replacements[0].domain().value(), "");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "indices");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "ignore_index", 0), -100);
  EXPECT_EQ(GetAttributeOr<std::string>(replacements[0], "reduction", ""), "mean");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsWrongSqueezeAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 0);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the ignore index or squeeze axes constants differ");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonzeroIgnoredIndexBranch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1, core::symbolic::TensorType::kFloat16, 1);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an ignored-label Where branch is not exactly zero");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonzeroIgnoredLossBranch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1, core::symbolic::TensorType::kFloat16, 0, 0.5F);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an ignored-label Where branch is not exactly zero");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonFloat16Scores) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1, core::symbolic::TensorType::kFloat);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the score, label, zero-branch, loss, or output types differ");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonFloat16Output) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);
  SetValueType(builder, "y", core::symbolic::TensorType::kFloat, Shape({}));

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the score, label, zero-branch, loss, or output types differ");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonOnnxInternalDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);
  SetNodeDomain(builder, 4, "custom");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the masked GatherElements loss topology is incomplete");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsIntermediateGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);
  builder.MakeOutput("selected");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an intermediate loss result is externally used");
}

TEST(MaxReluPattern, ReplacesSymmetricZeroMaximum) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kInt32, Shape({2, 3}));
  builder.MakeInitializer(MakeInitializer<int32_t>("zero", {1}, {0}));
  builder.MakeNode("Max", {"zero", "x"}, {"y"}, "", "maximum");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MaxReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Relu");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(replacements[0].name().value(), "MaxReluPattern--maximum");
}

TEST(MaxReluPattern, RejectsTwoZeroConstants) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInitializer(MakeInitializer<int32_t>("zero0", {1}, {0}));
  builder.MakeInitializer(MakeInitializer<int32_t>("zero1", {1}, {0}));
  builder.MakeNode("Max", {"zero0", "zero1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MaxReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "Max does not have exactly one scalar zero input");
}

TEST(MaxReluPattern, RejectsIntegerReluBeforeOpsetFourteen) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.MakeInput("x", core::symbolic::TensorType::kInt32, Shape({2, 3}));
  builder.MakeInitializer(MakeInitializer<int32_t>("zero", {1}, {0}));
  builder.MakeNode("Max", {"zero", "x"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MaxReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "integer Relu requires default-domain opset 14 or newer");
}

} // namespace
} // namespace Test
