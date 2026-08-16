// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/gather_pattern.h"
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

core::symbolic::SymShape Shape1D(int64_t a) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
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

} // namespace
} // namespace Test
