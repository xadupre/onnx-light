// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/layout/layout_pattern.h"
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

void SetModernOpset(core::builder::GraphBuilder &builder) { builder.SetOpsetVersion("", 18); }

void AddAxesInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                        const std::vector<int64_t> &axes) {
  builder.MakeInitializer(MakeInitializerShape(name.c_str(), axes));
}

void AddScalarInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                          int64_t value) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(static_cast<int>(TensorProto::DataType::INT64));
  tensor.int64_data().push_back(value);
  builder.MakeInitializer(tensor);
}

utils::RepeatedProtoField<AttributeProto> PermAttr(const std::vector<int64_t> &perm) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("perm");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : perm) {
    attribute.ints().push_back(value);
  }
  return attributes;
}

std::vector<int64_t> AttributeInts(const NodeProto &node, const char *name) {
  std::vector<int64_t> values;
  GetAttributeInts(node, name, values);
  return values;
}

std::vector<int64_t> InitializerInts(const core::builder::GraphBuilder &builder,
                                     const std::string &name) {
  std::vector<int64_t> values;
  for (const TensorProto &tensor : builder.Initializers()) {
    if (tensor.name().value() == name) {
      ReadIntegerValues(tensor, values);
      break;
    }
  }
  return values;
}

TEST(SqueezeAddPattern, MovesEqualSqueezesAfterAdd) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("y", core::symbolic::TensorType::kInt64, Shape({1}));
  AddAxesInitializer(builder, "axes", {0});
  builder.MakeNode("Squeeze", {"x", "axes"}, {"sx"});
  builder.MakeNode("Squeeze", {"y", "axes"}, {"sy"});
  builder.MakeNode("Add", {"sx", "sy"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeAddPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Add");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[1].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[1].input()[1].value(), "axes");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(SqueezeAddPattern, RejectsDifferentAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("y", core::symbolic::TensorType::kInt64, Shape({1, 1}));
  AddAxesInitializer(builder, "first_axes", {0});
  AddAxesInitializer(builder, "second_axes", {1});
  builder.MakeNode("Squeeze", {"x", "first_axes"}, {"sx"});
  builder.MakeNode("Squeeze", {"y", "second_axes"}, {"sy"});
  builder.MakeNode("Add", {"sx", "sy"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeAddPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(MulUnsqueezeUnsqueezePattern, MovesEqualUnsqueezesAfterMul) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  AddAxesInitializer(builder, "axes", {2});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"ux"});
  builder.MakeNode("Unsqueeze", {"y", "axes"}, {"uy"});
  builder.MakeNode("Mul", {"ux", "uy"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulUnsqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Mul");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[1].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[1].input()[1].value(), "axes");
}

TEST(MulUnsqueezeUnsqueezePattern, RejectsDifferentAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3}));
  AddAxesInitializer(builder, "first_axes", {0});
  AddAxesInitializer(builder, "second_axes", {1});
  builder.MakeNode("Unsqueeze", {"x", "first_axes"}, {"ux"});
  builder.MakeNode("Unsqueeze", {"y", "second_axes"}, {"uy"});
  builder.MakeNode("Mul", {"ux", "uy"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulUnsqueezeUnsqueezePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(SqueezeBinaryUnsqueezePattern, MovesUnsqueezeBeforeBinary) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape({1}));
  AddScalarInitializer(builder, "two", 2);
  AddAxesInitializer(builder, "zero", {0});
  builder.MakeNode("Squeeze", {"x"}, {"sx"});
  builder.MakeNode("Div", {"sx", "two"}, {"div"});
  builder.MakeNode("Unsqueeze", {"div", "zero"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeBinaryUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "two");
  EXPECT_EQ(replacements[0].input()[1].value(), "zero");
  EXPECT_EQ(replacements[1].op_type().value(), "Div");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(SqueezeBinaryUnsqueezePattern, RejectsNonScalarBinaryRightInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape({1}));
  AddAxesInitializer(builder, "two", {2});
  AddAxesInitializer(builder, "zero", {0});
  builder.MakeNode("Squeeze", {"x"}, {"sx"});
  builder.MakeNode("Div", {"sx", "two"}, {"div"});
  builder.MakeNode("Unsqueeze", {"div", "zero"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeBinaryUnsqueezePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(SwapUnsqueezeTransposePattern, SwapsUnsqueezeAndTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddAxesInitializer(builder, "axes", {1});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Transpose", {"xu"}, {"out"}, "", "", PermAttr({0, 2, 1, 3}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapUnsqueezeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Transpose");
  EXPECT_EQ(AttributeInts(replacements[0], "perm"), (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(replacements[1].op_type().value(), "Unsqueeze");
  EXPECT_EQ(InitializerInts(builder, replacements[1].input()[1].value()),
            (std::vector<int64_t>{2}));
}

TEST(SwapUnsqueezeTransposePattern, RejectsSharedUnsqueezeOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  AddAxesInitializer(builder, "axes", {1});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Transpose", {"xu"}, {"out"}, "", "", PermAttr({0, 2, 1, 3}));
  builder.MakeNode("Identity", {"xu"}, {"other"});
  builder.MakeOutput("out");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapUnsqueezeTransposePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(TransposeEqualReshapePattern, ReplacesSingletonTransposeWithReshape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 2, 1, 5}));
  builder.MakeNode("Transpose", {"x"}, {"out"}, "", "", PermAttr({0, 2, 1, 3}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeEqualReshapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(InitializerInts(builder, replacements[0].input()[1].value()),
            (std::vector<int64_t>{0, 1, 2, 0}));
}

TEST(TransposeEqualReshapePattern, RejectsMultipleMovedNonSingletonDimensions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 2, 4, 5}));
  builder.MakeNode("Transpose", {"x"}, {"out"}, "", "", PermAttr({0, 2, 1, 3}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeEqualReshapePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(TransposeReshapeTransposePattern, MovesReshapeAfterSecondTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4, 5}));
  AddAxesInitializer(builder, "shape", {2, 20, 3});
  builder.MakeNode("Transpose", {"x"}, {"first"}, "", "", PermAttr({0, 2, 3, 1}));
  builder.MakeNode("Reshape", {"first", "shape"}, {"reshaped"});
  builder.MakeNode("Transpose", {"reshaped"}, {"out"}, "", "", PermAttr({0, 2, 1}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeReshapeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[0].output()[0].value(), "first");
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(AttributeInts(replacements[1], "perm"), (std::vector<int64_t>{0, 3, 1, 2}));
  EXPECT_EQ(replacements[2].op_type().value(), "Reshape");
  EXPECT_EQ(InitializerInts(builder, replacements[2].input()[1].value()),
            (std::vector<int64_t>{2, 3, 20}));
}

TEST(TransposeReshapeTransposePattern, MovesReshapeBeforeFirstTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({32, 256, 28, 26}));
  AddAxesInitializer(builder, "shape", {32, 2, 14, 2, 13, 256});
  builder.MakeNode("Transpose", {"x"}, {"first"}, "", "", PermAttr({0, 2, 3, 1}));
  builder.MakeNode("Reshape", {"first", "shape"}, {"reshaped"});
  builder.MakeNode("Transpose", {"reshaped"}, {"out"}, "", "", PermAttr({0, 1, 3, 2, 4, 5}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeReshapeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(InitializerInts(builder, replacements[0].input()[1].value()),
            (std::vector<int64_t>{32, 256, 2, 14, 2, 13}));
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(AttributeInts(replacements[1], "perm"), (std::vector<int64_t>{0, 2, 3, 4, 5, 1}));
  EXPECT_EQ(replacements[1].output()[0].value(), "reshaped");
  EXPECT_EQ(replacements[2].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[2].output()[0].value(), "out");
}

TEST(TransposeReshapeTransposePattern, RejectsDynamicReshapeShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  SetModernOpset(builder);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 4, 5}));
  builder.MakeInput("shape", core::symbolic::TensorType::kInt64, Shape({3}));
  builder.MakeNode("Transpose", {"x"}, {"first"}, "", "", PermAttr({0, 2, 3, 1}));
  builder.MakeNode("Reshape", {"first", "shape"}, {"reshaped"});
  builder.MakeNode("Transpose", {"reshaped"}, {"out"}, "", "", PermAttr({0, 2, 1}));
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeReshapeTransposePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

} // namespace
} // namespace Test
