// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/shape_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <map>
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

utils::RepeatedProtoField<AttributeProto> CastToAttr(TensorProto::DataType to) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("to");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(static_cast<int64_t>(to));
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> ShapeStartEndAttrs(int64_t start, int64_t end) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &start_attr = attributes.add();
  start_attr.set_name("start");
  start_attr.set_type(AttributeProto::AttributeType::INT);
  start_attr.set_i(start);
  AttributeProto &end_attr = attributes.add();
  end_attr.set_name("end");
  end_attr.set_type(AttributeProto::AttributeType::INT);
  end_attr.set_i(end);
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> AxisAttrs(int64_t axis) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("axis");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(axis);
  return attributes;
}

void SetNodeDomain(core::builder::GraphBuilder &builder, std::size_t index,
                   const std::string &domain) {
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[index].set_domain(domain);
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

TEST(UnsqueezeShapePattern, RejectsNonConstantAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::symbolic::SymShape axes_shape;
  axes_shape.PushBack(core::symbolic::SymDim(1));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeInput("axes", core::symbolic::TensorType::kInt64, axes_shape);
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Shape", {"xu"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeShapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(PreShapeNodeEliminationPattern, EliminatesCastBeforeSingleShapeConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Shape", {"c"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Shape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(PreShapeNodeEliminationPattern, PreservesShapeStartEndAttributesAndName) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::FLOAT16));
  builder.MakeNode("Shape", {"c"}, {"y"}, "", "shape_node", ShapeStartEndAttrs(1, 3));
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Shape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(replacements[0].name().value(), "shape_node");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "start", -1), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "end", -1), 3);
}

TEST(PreShapeNodeEliminationPattern, KeepsCastAliveForAdditionalNonShapeConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Shape", {"c"}, {"y1"});
  builder.MakeNode("Identity", {"c"}, {"y2"});
  builder.MakeOutput("y1");
  builder.MakeOutput("y2");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].output()[0].value(), "c");
  EXPECT_EQ(replacements[1].op_type().value(), "Shape");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].output()[0].value(), "y1");
}

TEST(PreShapeNodeEliminationPattern, KeepsCastAliveWhenOutputIsGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Shape", {"c"}, {"y"});
  builder.MakeOutput("c");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[1].op_type().value(), "Shape");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
}

TEST(PreShapeNodeEliminationPattern, KeepsCastAliveWhenCapturedBySubgraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Shape", {"c"}, {"y"});
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"c"}, {"captured"});
  body.MakeOutput("captured");
  AddSubgraphReference(builder, "body", "body_result");
  builder.MakeOutput("y");
  builder.MakeOutput("body_result");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[1].op_type().value(), "Shape");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
}

TEST(PreShapeNodeEliminationPattern, RejectsWhenConsumerIsNotShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Identity", {"c"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Cast output is not consumed by a Shape");
}

TEST(PreShapeNodeEliminationPattern, RejectsNonDefaultDomainCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Shape", {"c"}, {"y"});
  builder.MakeOutput("y");
  SetNodeDomain(builder, 0, "custom.domain");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "candidate is not a default-domain Cast");
}

TEST(PreShapeNodeEliminationPattern, RejectsCustomDomainShapeConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c"}, "", "", CastToAttr(TensorProto::DataType::INT64));
  builder.MakeNode("Shape", {"c"}, {"y"});
  builder.MakeOutput("y");
  SetNodeDomain(builder, 1, "custom.domain");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PreShapeNodeEliminationPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Cast output is not consumed by a Shape");
}

TEST(PreShapeNodeEliminationPattern, OptimizesMultipleShapeConsumersToFixedPoint) {
  // Mirrors the graph shape used by ONNX Runtime's PreShapeNodeElimination
  // test: a Cast feeding both a non-Shape op and several Shape nodes. "shape2"
  // and "shape3" carry different start/end ranges so the unrelated
  // node-deduplication cleanup pass cannot merge them, keeping the assertions
  // focused on PreShapeNodeElimination's own rewrite.
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape3D(2, 3, 4));
  builder.MakeNode("Cast", {"x"}, {"c1"}, "", "", CastToAttr(TensorProto::DataType::FLOAT16));
  builder.MakeNode("Shape", {"c1"}, {"shape1"});
  builder.MakeNode("Add", {"c1", "c1"}, {"add"});
  builder.MakeNode("Cast", {"add"}, {"c2"}, "", "", CastToAttr(TensorProto::DataType::FLOAT));
  builder.MakeNode("Shape", {"c2"}, {"shape2"}, "", "", ShapeStartEndAttrs(0, 2));
  builder.MakeNode("Shape", {"c2"}, {"shape3"}, "", "", ShapeStartEndAttrs(1, 3));
  builder.MakeNode("Concat", {"shape1", "shape2", "shape3"}, {"y"}, "", "", AxisAttrs(0));
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::PreShapeNodeEliminationPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  // "c1" still feeds the Add, so the first Cast is retained even though its
  // Shape consumer ("shape1") is rewritten to read "x" directly. "c2" only
  // ever fed Shape nodes, so the second Cast converges away entirely and its
  // Shape consumers ("shape2", "shape3") are redirected to "add" (the Cast's
  // own input), each keeping its own start/end range.
  int cast_count = 0;
  for (const NodeProto &node : builder.Nodes()) {
    if (node.op_type().value() == "Cast") {
      ++cast_count;
    }
  }
  EXPECT_EQ(cast_count, 1);
  std::map<std::string, std::string> shape_input_by_output;
  std::map<std::string, const NodeProto *> shape_node_by_output;
  for (const NodeProto &node : builder.Nodes()) {
    if (node.op_type().value() == "Shape") {
      ASSERT_EQ(node.input_size(), 1);
      shape_input_by_output[node.output()[0].value()] = node.input()[0].value();
      shape_node_by_output[node.output()[0].value()] = &node;
    }
  }
  ASSERT_EQ(shape_input_by_output.size(), 3u);
  EXPECT_EQ(shape_input_by_output["shape1"], "x");
  EXPECT_EQ(shape_input_by_output["shape2"], "add");
  EXPECT_EQ(shape_input_by_output["shape3"], "add");
  EXPECT_EQ(GetAttributeOr<int64_t>(*shape_node_by_output["shape2"], "start", -1), 0);
  EXPECT_EQ(GetAttributeOr<int64_t>(*shape_node_by_output["shape2"], "end", -1), 2);
  EXPECT_EQ(GetAttributeOr<int64_t>(*shape_node_by_output["shape3"], "start", -1), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(*shape_node_by_output["shape3"], "end", -1), 3);
}

} // namespace
} // namespace Test
