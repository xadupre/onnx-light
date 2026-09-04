// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/pad_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape2D() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(3));
  return shape;
}

TEST(PadPadFusionPattern, SumsModernFullRankPads) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D());
  builder.MakeInitializer(MakeInitializerShape("p0", {1, 2, 3, 4}));
  builder.MakeInitializer(MakeInitializerShape("p1", {5, 6, 7, 8}));
  builder.MakeInitializer(MakeInitializer<float>("value", {}, {2.0F}));
  builder.MakeNode("Pad", {"x", "p0", "value"}, {"t"}, "", "first");
  builder.MakeNode("Pad", {"t", "p1", "value"}, {"y"}, "", "second");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadPadFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  ASSERT_EQ(replacements[0].input_size(), 3);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[2].value(), "value");
  const TensorProto *pads = graph.GetComputedConstant(replacements[0].input()[1].value());
  ASSERT_NE(pads, nullptr);
  std::vector<int64_t> values;
  ASSERT_TRUE(ReadIntegerValues(*pads, values));
  EXPECT_EQ(values, (std::vector<int64_t>{6, 8, 10, 12}));
}

TEST(PadPadFusionPattern, ExpandsAxesAndSupportsDefaultValue) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D());
  builder.MakeInitializer(MakeInitializerShape("p0", {1, 2}));
  builder.MakeInitializer(MakeInitializerShape("a0", {0}));
  builder.MakeInitializer(MakeInitializerShape("p1", {3, 4}));
  builder.MakeInitializer(MakeInitializerShape("a1", {-1}));
  builder.MakeNode("Pad", {"x", "p0", "", "a0"}, {"t"});
  builder.MakeNode("Pad", {"t", "p1", "", "a1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadPadFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements[0].input_size(), 2);
  const TensorProto *pads = graph.GetComputedConstant(replacements[0].input()[1].value());
  ASSERT_NE(pads, nullptr);
  std::vector<int64_t> values;
  ASSERT_TRUE(ReadIntegerValues(*pads, values));
  EXPECT_EQ(values, (std::vector<int64_t>{1, 3, 2, 4}));
}

TEST(PadPadFusionPattern, SupportsAttributeForm) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 10);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D());
  NodeProto first = MakeNode("Pad", {"x"}, {"t"});
  AddAttribute<std::vector<int64_t>>(first, "pads", {1, 0, 0, 1});
  AddAttribute<float>(first, "value", 2.0F);
  NodeProto second = MakeNode("Pad", {"t"}, {"y"});
  AddAttribute<std::vector<int64_t>>(second, "pads", {0, 2, 3, 0});
  AddAttribute<float>(second, "value", 2.0F);
  builder.MakeNode("Pad", {"x"}, {"t"}, "", "first", first.attribute());
  builder.MakeNode("Pad", {"t"}, {"y"}, "", "second", second.attribute());
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadPadFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  std::vector<int64_t> pads;
  ASSERT_TRUE(GetAttributeInts(replacements[0], "pads", pads));
  EXPECT_EQ(pads, (std::vector<int64_t>{1, 2, 3, 1}));
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[0], "value", 0.0F), 2.0F);
}

TEST(PadPadFusionPattern, RejectsDifferentValuesNegativePadsAndSharedOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2D());
  builder.MakeInitializer(MakeInitializerShape("p0", {1, 0, 0, 1}));
  builder.MakeInitializer(MakeInitializerShape("p1", {0, -1, 0, 1}));
  builder.MakeInitializer(MakeInitializer<float>("v0", {}, {1.0F}));
  builder.MakeInitializer(MakeInitializer<float>("v1", {}, {2.0F}));
  builder.MakeNode("Pad", {"x", "p0", "v0"}, {"t"});
  builder.MakeNode("Pad", {"t", "p1", "v1"}, {"y"});
  builder.MakeNode("Identity", {"t"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadPadFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

} // namespace
} // namespace Test
