// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/sequence_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape2D(int64_t a, int64_t b) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  shape.PushBack(core::symbolic::SymDim(b));
  return shape;
}

utils::RepeatedProtoField<AttributeProto> IntAttrs(const std::string &name, int64_t value) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attr = attributes.add();
  attr.set_name(name);
  attr.set_type(AttributeProto::AttributeType::INT);
  attr.set_i(value);
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> AxisKeepdims(int64_t axis, int64_t keepdims) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &a = attributes.add();
  a.set_name("axis");
  a.set_type(AttributeProto::AttributeType::INT);
  a.set_i(axis);
  AttributeProto &k = attributes.add();
  k.set_name("keepdims");
  k.set_type(AttributeProto::AttributeType::INT);
  k.set_i(keepdims);
  return attributes;
}

TEST(SequenceConstructAtPattern, ReplacesWithIdentities) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x0", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInput("x1", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInitializer(MakeInitializer<int64_t>("i0", {}, {0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i1", {}, {1}));
  builder.MakeNode("SequenceConstruct", {"x0", "x1"}, {"seq"});
  builder.MakeNode("SequenceAt", {"seq", "i0"}, {"y0"});
  builder.MakeNode("SequenceAt", {"seq", "i1"}, {"y1"});
  builder.MakeOutput("y0");
  builder.MakeOutput("y1");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SequenceConstructAtPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x0");
  EXPECT_EQ(replacements[0].output()[0].value(), "y0");
  EXPECT_EQ(replacements[1].input()[0].value(), "x1");
  EXPECT_EQ(replacements[1].output()[0].value(), "y1");
}

TEST(SequenceConstructAtPattern, RejectsNonSequenceAtConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x0", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInput("x1", core::symbolic::TensorType::kFloat, Shape2D(2, 3));
  builder.MakeInitializer(MakeInitializer<int64_t>("i0", {}, {0}));
  builder.MakeNode("SequenceConstruct", {"x0", "x1"}, {"seq"});
  builder.MakeNode("SequenceAt", {"seq", "i0"}, {"y0"});
  builder.MakeNode("SequenceLength", {"seq"}, {"len"});
  builder.MakeOutput("y0");
  builder.MakeOutput("len");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SequenceConstructAtPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
}

TEST(SplitToSequenceSequenceAtPattern, ReplacesWithSplitWhenSplitProvided) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 6));
  builder.MakeInitializer(MakeInitializer<int64_t>("split", {3}, {2, 2, 2}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i0", {}, {0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i1", {}, {1}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i2", {}, {2}));
  builder.MakeNode("SplitToSequence", {"x", "split"}, {"seq"}, "", "", IntAttrs("axis", 1));
  builder.MakeNode("SequenceAt", {"seq", "i0"}, {"y0"});
  builder.MakeNode("SequenceAt", {"seq", "i1"}, {"y1"});
  builder.MakeNode("SequenceAt", {"seq", "i2"}, {"y2"});
  builder.MakeOutput("y0");
  builder.MakeOutput("y1");
  builder.MakeOutput("y2");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SplitToSequenceSequenceAtPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Split");
  ASSERT_EQ(replacements[0].input_size(), 2);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "split");
  ASSERT_EQ(replacements[0].output_size(), 3);
  EXPECT_EQ(replacements[0].output()[0].value(), "y0");
  EXPECT_EQ(replacements[0].output()[1].value(), "y1");
  EXPECT_EQ(replacements[0].output()[2].value(), "y2");
}

TEST(SplitToSequenceSequenceAtPattern, InsertsSqueezeWhenKeepdimsZero) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D(2, 2));
  builder.MakeInitializer(MakeInitializer<int64_t>("i0", {}, {0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("i1", {}, {1}));
  builder.MakeNode("SplitToSequence", {"x"}, {"seq"}, "", "", AxisKeepdims(1, 0));
  builder.MakeNode("SequenceAt", {"seq", "i0"}, {"y0"});
  builder.MakeNode("SequenceAt", {"seq", "i1"}, {"y1"});
  builder.MakeOutput("y0");
  builder.MakeOutput("y1");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SplitToSequenceSequenceAtPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Split");
  ASSERT_EQ(replacements[0].input_size(), 1);
  EXPECT_EQ(replacements[1].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[2].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[1].output()[0].value(), "y0");
  EXPECT_EQ(replacements[2].output()[0].value(), "y1");
}

} // namespace
} // namespace Test
