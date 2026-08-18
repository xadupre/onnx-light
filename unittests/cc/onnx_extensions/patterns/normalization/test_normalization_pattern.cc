// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/patterns/normalization/normalization_pattern.h"
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

core::builder::GraphBuilder::SchemaLookupFn SingleOutputNormalizationSchemaLookup() {
  return [](const std::string &op_type) {
    if (op_type == "LayerNormalization") {
      return std::vector<core::schema::LightOpSchema>{};
    }
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
Attributes(std::initializer_list<std::pair<const char *, int64_t>> integers = {},
           std::initializer_list<std::pair<const char *, float>> floats = {}) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &[name, value] : integers) {
    AttributeProto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::INT);
    attribute.set_i(value);
  }
  for (const auto &[name, value] : floats) {
    AttributeProto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::FLOAT);
    attribute.set_f(value);
  }
  return attributes;
}

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType type) {
  builder.MakeNode("Cast", {input}, {output}, "", "",
                   Attributes({{"to", static_cast<int64_t>(type)}}));
}

void AddFloatInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                         const std::vector<int64_t> &dims, const std::vector<float> &values) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dims, values));
}

void AddAxes(core::builder::GraphBuilder &builder, const std::string &name,
             const std::vector<int64_t> &axes) {
  builder.MakeInitializer(MakeInitializerShape(name.c_str(), axes));
}

void SetNodeDomain(core::builder::GraphBuilder &builder, std::size_t index,
                   const std::string &domain) {
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[index].set_domain(domain);
}

void SetNodeIntegerAttribute(core::builder::GraphBuilder &builder, std::size_t index,
                             const std::string &name, int64_t value) {
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  AddAttribute<int64_t>(nodes[index], name.c_str(), value);
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

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const TensorProto &initializer : builder.Initializers()) {
    if (initializer.name().value() == name) {
      return &initializer;
    }
  }
  return nullptr;
}

TEST(LayerNormalizationPattern, FusesExactVarianceGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "mean",
                   Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"}, "", "variance");
  builder.MakeNode("Add", {"variance", "epsilon"}, {"variance_epsilon"});
  builder.MakeNode("Sqrt", {"variance_epsilon"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[3]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 7u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[3], &builder.Nodes()[3]);
  EXPECT_EQ(match.nodes[4], &builder.Nodes()[4]);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[3]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "LayerNormalization");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", 0), -1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "stash_type", 0),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[0], "epsilon", 0.0F), 1e-5F);
  ASSERT_EQ(builder.Initializers().size(), 5u);
  EXPECT_EQ(builder.Initializers()[3].dims()[0], 3);
  EXPECT_EQ(builder.Initializers()[4].dims()[0], 3);
}

TEST(LayerNormalizationPattern, RejectsImplicitMeanKeepdims) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"});
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Sqrt", {"variance"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the first ReduceMean does not explicitly keep dimensions");
}

TEST(LayerNormalizationPattern, PreservesZeroEpsilonWhenAddIsAbsent) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "", Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Sqrt", {"variance"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[0], "epsilon", -1.0F), 0.0F);
}

TEST(LayerNormalizationPattern, FusesReciprocalMulForm) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "", Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Add", {"variance", "epsilon"}, {"variance_epsilon"});
  builder.MakeNode("Sqrt", {"variance_epsilon"}, {"standard_deviation"});
  builder.MakeNode("Reciprocal", {"standard_deviation"}, {"inverse_standard_deviation"});
  builder.MakeNode("Mul", {"centered", "inverse_standard_deviation"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 8u);
  EXPECT_EQ(match.nodes[6], &builder.Nodes()[6]);
  EXPECT_EQ(match.nodes[7], &builder.Nodes()[7]);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "LayerNormalization");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(LayerNormalizationPattern, RejectsEmptyAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "", Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Sqrt", {"variance"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the variance ReduceMean axes are not constant integers");
}

TEST(LayerNormalizationPattern, RejectsVarianceWithoutKeepdims) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "", Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Sqrt", {"variance"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("y");
  SetNodeIntegerAttribute(builder, 3, "keepdims", 0);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the variance ReduceMean does not keep dimensions");
}

TEST(LayerNormalizationPattern, RejectsIntermediateGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "", Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Sqrt", {"variance"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("mean");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a LayerNormalization intermediate is externally used");
}

TEST(LayerNormalizationPattern, RejectsNonOnnxPowerDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.SetOpsetVersion("custom", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("ReduceMean", {"x", "axes"}, {"mean"}, "", "", Attributes({{"keepdims", 1}}));
  builder.MakeNode("Sub", {"x", "mean"}, {"centered"});
  builder.MakeNode("Pow", {"centered", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"variance"});
  builder.MakeNode("Sqrt", {"variance"}, {"standard_deviation"});
  builder.MakeNode("Div", {"centered", "standard_deviation"}, {"y"});
  builder.MakeOutput("y");
  SetNodeDomain(builder, 2, "custom");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[3]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the variance input is not produced by an unshared Pow");
}

TEST(LayerNormalizationScalePattern, FoldsFollowingScaleAndBias) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale0", {3}, {1.0F, 2.0F, 3.0F});
  AddFloatInitializer(builder, "scale1", {3}, {4.0F, 5.0F, 6.0F});
  AddFloatInitializer(builder, "bias1", {3}, {0.1F, 0.2F, 0.3F});
  builder.MakeNode("LayerNormalization", {"x", "scale0"}, {"normalized"}, "", "layer",
                   Attributes({{"axis", -1}, {"stash_type", 1}}, {{"epsilon", 1e-5F}}));
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"scaled"});
  builder.MakeNode("Add", {"scaled", "bias1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationScalePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[2]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Mul");
  EXPECT_EQ(replacements[0].input()[0].value(), "scale0");
  EXPECT_EQ(replacements[0].input()[1].value(), "scale1");
  EXPECT_EQ(replacements[1].op_type().value(), "LayerNormalization");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].input()[1].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[2].value(), "bias1");
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[1], "axis", 0), -1);
}

TEST(LayerNormalizationScalePattern, RejectsMismatchedScaleShape) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale0", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "scale1", {1}, {2.0F});
  builder.MakeNode("LayerNormalization", {"x", "scale0"}, {"normalized"});
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"scaled"});
  builder.MakeNode("Identity", {"scaled"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationScalePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason,
            "the following Mul scale shape differs from the existing scale");
}

TEST(LayerNormalizationScalePattern, ScalesExistingBiasForTerminalMul) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale0", {3}, {1.0F, 2.0F, 3.0F});
  AddFloatInitializer(builder, "bias0", {3}, {0.1F, 0.2F, 0.3F});
  AddFloatInitializer(builder, "scale1", {3}, {4.0F, 5.0F, 6.0F});
  builder.MakeNode("LayerNormalization", {"x", "scale0", "bias0"}, {"normalized"});
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationScalePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Mul");
  EXPECT_EQ(replacements[0].input()[0].value(), "scale0");
  EXPECT_EQ(replacements[0].input()[1].value(), "scale1");
  EXPECT_EQ(replacements[1].op_type().value(), "Mul");
  EXPECT_EQ(replacements[1].input()[0].value(), "scale1");
  EXPECT_EQ(replacements[1].input()[1].value(), "bias0");
  EXPECT_EQ(replacements[2].op_type().value(), "LayerNormalization");
  EXPECT_EQ(replacements[2].input()[1].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[2].input()[2].value(), replacements[1].output()[0].value());
  EXPECT_EQ(replacements[2].output()[0].value(), "y");
}

TEST(LayerNormalizationScalePattern, RejectsTerminalMismatchedScaleShape) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale0", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "scale1", {1}, {2.0F});
  builder.MakeNode("LayerNormalization", {"x", "scale0"}, {"normalized"});
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationScalePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason,
            "the following Mul scale shape differs from the existing scale");
}

TEST(LayerNormalizationScalePattern, RejectsNormalizedGraphOutput) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale0", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "scale1", {3}, {2.0F, 2.0F, 2.0F});
  builder.MakeNode("LayerNormalization", {"x", "scale0"}, {"normalized"});
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"y"});
  builder.MakeOutput("normalized");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LayerNormalizationScalePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the normalized value is externally used");
}

TEST(CastLayerNormalizationCastPattern, MovesCastsToParameters) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("LayerNormalization", {"cast_x", "scale", "bias"}, {"normalized"}, "", "layer",
                   Attributes({{"axis", -1}, {"stash_type", 1}}));
  AddCast(builder, "normalized", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastLayerNormalizationCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[1]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "scale");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "to", 0),
            static_cast<int64_t>(TensorProto::DataType::FLOAT16));
  EXPECT_EQ(replacements[1].input()[0].value(), "bias");
  EXPECT_EQ(replacements[2].op_type().value(), "LayerNormalization");
  EXPECT_EQ(replacements[2].input()[0].value(), "x");
  EXPECT_EQ(replacements[2].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[2], "stash_type", 0), 1);
}

TEST(CastLayerNormalizationCastPattern, PreservesUnusedOptionalOutputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("LayerNormalization", {"cast_x", "scale"},
                   {"normalized", "mean", "inverse_standard_deviation"});
  AddCast(builder, "normalized", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastLayerNormalizationCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[1].output_size(), 3u);
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
  EXPECT_EQ(replacements[1].output()[1].value(), "mean");
  EXPECT_EQ(replacements[1].output()[2].value(), "inverse_standard_deviation");
}

TEST(CastLayerNormalizationCastPattern, RejectsUsedOptionalOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("LayerNormalization", {"cast_x", "scale"},
                   {"normalized", "mean", "inverse_standard_deviation"});
  AddCast(builder, "normalized", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");
  builder.MakeOutput("inverse_standard_deviation");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastLayerNormalizationCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an optional normalization output is used");
}

TEST(CastLayerNormalizationCastPattern, RejectsSharedInputCast) {
  core::builder::GraphBuilder builder("g", SingleOutputNormalizationSchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("LayerNormalization", {"cast_x", "scale"}, {"normalized"});
  AddCast(builder, "normalized", "y", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Identity", {"cast_x"}, {"shared"});
  builder.MakeOutput("y");
  builder.MakeOutput("shared");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastLayerNormalizationCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the input Cast output is externally used");
}

TEST(BatchNormalizationPattern, ReplacesNeutralBatchNormalization) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "batch",
                   Attributes({}, {{"epsilon", 0.0F}}));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(replacements[0].name().value(), "BatchNormalizationPattern--batch");
}

TEST(BatchNormalizationPattern, RejectsNonzeroEpsilon) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "epsilon is not zero");
}

TEST(BatchNormalizationPattern, RejectsTrainingMode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "",
                   Attributes({{"training_mode", 1}}, {{"epsilon", 0.0F}}));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "BatchNormalization is not in inference mode");
}

TEST(BatchNormalizationPattern, RejectsRunningMeanGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "",
                   Attributes({}, {{"epsilon", 0.0F}}));
  builder.MakeOutput("y");
  builder.MakeOutput("running_mean");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the running mean output is used");
}

TEST(BatchNormalizationTrainingPattern, ExpandsTrainingNormalization) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 2.0F, 3.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.1F, 0.2F, 0.3F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "batch",
                   Attributes({{"training_mode", 1}}, {{"epsilon", 1e-3F}}));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationTrainingPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 11u);
  EXPECT_EQ(replacements[0].op_type().value(), "ReduceMean");
  EXPECT_EQ(replacements[1].op_type().value(), "Sub");
  EXPECT_EQ(replacements[2].op_type().value(), "Mul");
  EXPECT_EQ(replacements[3].op_type().value(), "ReduceMean");
  EXPECT_EQ(replacements[4].op_type().value(), "Add");
  EXPECT_EQ(replacements[5].op_type().value(), "Sqrt");
  EXPECT_EQ(replacements[6].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[7].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[8].op_type().value(), "Div");
  EXPECT_EQ(replacements[9].op_type().value(), "Mul");
  EXPECT_EQ(replacements[10].op_type().value(), "Add");
  EXPECT_EQ(replacements[10].output()[0].value(), "y");
  const TensorProto *axes = FindInitializer(builder, "BatchNormalizationTrainingPattern.axes");
  ASSERT_NE(axes, nullptr);
  ASSERT_EQ(axes->int64_data().size(), 2u);
  EXPECT_EQ(axes->int64_data()[0], 0);
  EXPECT_EQ(axes->int64_data()[1], 2);
}

TEST(BatchNormalizationTrainingPattern, RejectsOpsetBelowEighteen) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 17);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "",
                   Attributes({{"training_mode", 1}}));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationTrainingPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the default-domain opset is below 18");
}

TEST(BatchNormalizationTrainingPattern, RejectsInferenceMode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "",
                   Attributes({{"training_mode", 0}}));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationTrainingPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "BatchNormalization is not in training mode");
}

TEST(BatchNormalizationTrainingPattern, RejectsCapturedRunningVariance) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddFloatInitializer(builder, "scale", {3}, {1.0F, 1.0F, 1.0F});
  AddFloatInitializer(builder, "bias", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "mean", {3}, {0.0F, 0.0F, 0.0F});
  AddFloatInitializer(builder, "variance", {3}, {1.0F, 1.0F, 1.0F});
  builder.MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "variance"},
                   {"y", "running_mean", "running_variance"}, "", "",
                   Attributes({{"training_mode", 1}}));
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"running_variance"}, {"captured"});
  body.MakeOutput("captured");
  AddSubgraphReference(builder, "body", "body_result");
  builder.MakeOutput("y");
  builder.MakeOutput("body_result");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::BatchNormalizationTrainingPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the running variance output is used");
}

TEST(RMSNormalizationPattern, FusesCastRmsGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape({2, 3}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  AddCast(builder, "x", "cast_x", TensorProto::DataType::FLOAT);
  builder.MakeNode("Pow", {"cast_x", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"mean"}, "", "mean");
  builder.MakeNode("Add", {"mean", "epsilon"}, {"mean_epsilon"});
  builder.MakeNode("Sqrt", {"mean_epsilon"}, {"root"});
  builder.MakeNode("Div", {"one", "root"}, {"reciprocal"});
  builder.MakeNode("Mul", {"cast_x", "reciprocal"}, {"normalized"});
  AddCast(builder, "normalized", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 8u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[7], &builder.Nodes()[7]);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[2]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "RMSNormalization");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", 0), -1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "stash_type", 0),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
  const TensorProto *scale = FindInitializer(builder, replacements[0].input()[1].value());
  ASSERT_NE(scale, nullptr);
  EXPECT_EQ(scale->data_type(), TensorProto::DataType::FLOAT16);
  EXPECT_EQ(scale->dims()[0], 3);
}

TEST(RMSNormalizationPattern, RejectsSharedEpsilonResult) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  builder.MakeNode("Pow", {"x", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"mean"});
  builder.MakeNode("Add", {"mean", "epsilon"}, {"mean_epsilon"});
  builder.MakeNode("Sqrt", {"mean_epsilon"}, {"root"});
  builder.MakeNode("Div", {"one", "root"}, {"reciprocal"});
  builder.MakeNode("Mul", {"x", "reciprocal"}, {"y"});
  builder.MakeNode("Identity", {"mean_epsilon"}, {"shared"});
  builder.MakeOutput("y");
  builder.MakeOutput("shared");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Add is not followed by one default-domain Sqrt");
}

TEST(RMSNormalizationPattern, AcceptsEpsilonAsFirstAddInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  builder.MakeNode("Pow", {"x", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"mean"});
  builder.MakeNode("Add", {"epsilon", "mean"}, {"mean_epsilon"});
  builder.MakeNode("Sqrt", {"mean_epsilon"}, {"root"});
  builder.MakeNode("Div", {"one", "root"}, {"reciprocal"});
  builder.MakeNode("Mul", {"x", "reciprocal"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[0], "epsilon", 0.0F), 1e-5F);
}

TEST(RMSNormalizationPattern, RejectsNonSuffixAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddAxes(builder, "axes", {-2});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  builder.MakeNode("Pow", {"x", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"mean"});
  builder.MakeNode("Add", {"mean", "epsilon"}, {"mean_epsilon"});
  builder.MakeNode("Sqrt", {"mean_epsilon"}, {"root"});
  builder.MakeNode("Div", {"one", "root"}, {"reciprocal"});
  builder.MakeNode("Mul", {"x", "reciprocal"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the ReduceMean axes are not a suffix of the normalized input");
}

TEST(RMSNormalizationPattern, RejectsReduceWithoutKeepdims) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  builder.MakeNode("Pow", {"x", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"mean"});
  builder.MakeNode("Add", {"mean", "epsilon"}, {"mean_epsilon"});
  builder.MakeNode("Sqrt", {"mean_epsilon"}, {"root"});
  builder.MakeNode("Div", {"one", "root"}, {"reciprocal"});
  builder.MakeNode("Mul", {"x", "reciprocal"}, {"y"});
  builder.MakeOutput("y");
  SetNodeIntegerAttribute(builder, 1, "keepdims", 0);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the ReduceMean does not keep dimensions");
}

TEST(RMSNormalizationPattern, RejectsIntermediateGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  AddAxes(builder, "axes", {-1});
  AddFloatInitializer(builder, "epsilon", {1}, {1e-5F});
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  builder.MakeNode("Pow", {"x", "two"}, {"squared"});
  builder.MakeNode("ReduceMean", {"squared", "axes"}, {"mean"});
  builder.MakeNode("Add", {"mean", "epsilon"}, {"mean_epsilon"});
  builder.MakeNode("Sqrt", {"mean_epsilon"}, {"root"});
  builder.MakeNode("Div", {"one", "root"}, {"reciprocal"});
  builder.MakeNode("Mul", {"x", "reciprocal"}, {"y"});
  builder.MakeOutput("mean_epsilon");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an intermediate RMS result has another use");
}

TEST(RMSNormalizationMulPattern, MultipliesConstantScales) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 2}));
  AddFloatInitializer(builder, "scale0", {2}, {3.0F, 4.0F});
  AddFloatInitializer(builder, "scale1", {2}, {5.0F, 6.0F});
  builder.MakeNode("RMSNormalization", {"x", "scale0"}, {"normalized"}, "", "rms",
                   Attributes({{"axis", -1}, {"stash_type", 1}}, {{"epsilon", 1e-5F}}));
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[1]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "RMSNormalization");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  const TensorProto *scale = FindInitializer(builder, replacements[0].input()[1].value());
  ASSERT_NE(scale, nullptr);
  ASSERT_EQ(scale->float_data().size(), 2u);
  EXPECT_FLOAT_EQ(scale->float_data()[0], 15.0F);
  EXPECT_FLOAT_EQ(scale->float_data()[1], 24.0F);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "axis", 0), -1);
}

TEST(RMSNormalizationMulPattern, RejectsDifferentScaleShapes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 2}));
  AddFloatInitializer(builder, "scale0", {2}, {3.0F, 4.0F});
  AddFloatInitializer(builder, "scale1", {1}, {5.0F});
  builder.MakeNode("RMSNormalization", {"x", "scale0"}, {"normalized"});
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the two scale shapes differ");
}

TEST(RMSNormalizationMulPattern, RejectsNormalizedGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 2}));
  AddFloatInitializer(builder, "scale0", {2}, {3.0F, 4.0F});
  AddFloatInitializer(builder, "scale1", {2}, {5.0F, 6.0F});
  builder.MakeNode("RMSNormalization", {"x", "scale0"}, {"normalized"});
  builder.MakeNode("Mul", {"normalized", "scale1"}, {"y"});
  builder.MakeOutput("normalized");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::RMSNormalizationMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the RMSNormalization output is externally used");
}

} // namespace
} // namespace Test
