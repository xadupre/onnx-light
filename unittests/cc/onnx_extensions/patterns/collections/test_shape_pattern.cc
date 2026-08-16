// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/shape_pattern.h"
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

core::symbolic::SymShape Shape4D(int64_t a, int64_t b, int64_t c, int64_t d) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(a));
  shape.PushBack(core::symbolic::SymDim(b));
  shape.PushBack(core::symbolic::SymDim(c));
  shape.PushBack(core::symbolic::SymDim(d));
  return shape;
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

} // namespace
} // namespace Test
