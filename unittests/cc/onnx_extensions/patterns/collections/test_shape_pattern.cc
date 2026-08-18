// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/shape_pattern.h"
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

core::symbolic::SymShape Shape4D(int64_t a, int64_t b, int64_t c, int64_t d) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  shape.PushBack(core::symbolic::SymDim(b));
  shape.PushBack(core::symbolic::SymDim(c));
  shape.PushBack(core::symbolic::SymDim(d));
  return shape;
}

core::symbolic::SymShape Shape3D(int64_t a, int64_t b, int64_t c) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  shape.PushBack(core::symbolic::SymDim(b));
  shape.PushBack(core::symbolic::SymDim(c));
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

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const TensorProto &tensor : builder.Initializers()) {
    if (tensor.name().value() == name) {
      return &tensor;
    }
  }
  return nullptr;
}

TEST(GatherShapePattern, FusesContiguousRangeIntoBoundedShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(2, 3, 4, 5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {1, 2}));
  builder.MakeNode("Shape", {"x"}, {"s"});
  builder.MakeNode("Gather", {"s", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherShapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Shape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "start", -1), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "end", -1), 3);
}

TEST(GatherShapePattern, ScalarIndexInsertsSqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(2, 3, 4, 5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {}, {2}));
  builder.MakeNode("Shape", {"x"}, {"s"});
  builder.MakeNode("Gather", {"s", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherShapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Shape");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "start", -1), 2);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "end", -1), 3);
  EXPECT_EQ(replacements[1].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
}

TEST(GatherShapePattern, RejectsNonContiguousRange) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(2, 3, 4, 5));
  builder.MakeInitializer(MakeInitializer<int64_t>("idx", {2}, {0, 2}));
  builder.MakeNode("Shape", {"x"}, {"s"});
  builder.MakeNode("Gather", {"s", "idx"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GatherShapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ShapeTransposePattern, RewritesShapeOfTransposeIntoGather) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Transpose", {"x"}, {"xt"}, "", "", PermAttr({2, 0, 1}));
  builder.MakeNode("Shape", {"xt"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Shape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Gather");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[1], "axis", -1), 0);

  const TensorProto *perm = FindInitializer(builder, replacements[1].input()[1].value());
  ASSERT_NE(perm, nullptr);
  std::vector<int64_t> perm_values;
  ASSERT_TRUE(ReadIntegerValues(*perm, perm_values));
  EXPECT_EQ(perm_values, (std::vector<int64_t>{2, 0, 1}));
}

TEST(ShapeTransposePattern, HonoursShapeStartEndRange) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Transpose", {"x"}, {"xt"}, "", "", PermAttr({2, 0, 1}));
  utils::RepeatedProtoField<AttributeProto> shape_attrs;
  AttributeProto &start = shape_attrs.add();
  start.set_name("start");
  start.set_type(AttributeProto::AttributeType::INT);
  start.set_i(1);
  builder.MakeNode("Shape", {"xt"}, {"y"}, "", "", shape_attrs);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  const TensorProto *perm = FindInitializer(builder, replacements[1].input()[1].value());
  ASSERT_NE(perm, nullptr);
  std::vector<int64_t> perm_values;
  ASSERT_TRUE(ReadIntegerValues(*perm, perm_values));
  EXPECT_EQ(perm_values, (std::vector<int64_t>{0, 1}));
}

TEST(ShapeTransposePattern, RejectsShapeWithoutTransposeParent) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Shape", {"x"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeTransposePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(UnsqueezeShapePattern, RewritesShapeOfUnsqueezeIntoConcat) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {1}));
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Shape", {"xu"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeShapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Shape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "start", -1), 0);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "end", -1), 1);
  EXPECT_EQ(replacements[1].op_type().value(), "Shape");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[1], "start", -1), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[1], "end", -1), 3);
  EXPECT_EQ(replacements[2].op_type().value(), "Concat");
  EXPECT_EQ(replacements[2].output()[0].value(), "y");
  ASSERT_EQ(replacements[2].input_size(), 3);
  EXPECT_EQ(replacements[2].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[2].input()[2].value(), replacements[1].output()[0].value());

  const TensorProto *one = FindInitializer(builder, replacements[2].input()[1].value());
  ASSERT_NE(one, nullptr);
  std::vector<int64_t> one_values;
  ASSERT_TRUE(ReadIntegerValues(*one, one_values));
  EXPECT_EQ(one_values, (std::vector<int64_t>{1}));
}

TEST(UnsqueezeShapePattern, RejectsWhenNoShapeConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {1}));
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Identity", {"xu"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeShapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

} // namespace
} // namespace Test
