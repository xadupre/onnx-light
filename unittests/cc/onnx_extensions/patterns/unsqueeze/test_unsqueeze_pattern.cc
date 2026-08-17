// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

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
  for (int64_t d : dims) {
    shape.PushBack(core::symbolic::SymDim(d));
  }
  return shape;
}

void AddAxesInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                        const std::vector<int64_t> &axes) {
  builder.MakeInitializer(MakeInitializerShape(name.c_str(), axes));
}

// Reads the values of the initializer named ``name`` from the builder.
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

TEST(UnsqueezeUnsqueezePattern, MergesTwoSingleAxisUnsqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "a0", {2});
  AddAxesInitializer(builder, "a1", {3});
  builder.MakeNode("Unsqueeze", {"x", "a0"}, {"u0"});
  builder.MakeNode("Unsqueeze", {"u0", "a1"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
  EXPECT_EQ(InitializerInts(builder, replacements[0].input()[1].value()),
            (std::vector<int64_t>{2, 3}));
}

TEST(UnsqueezeUnsqueezePattern, MergesMultiAxisWithKnownRank) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "a0", {0, 1});
  AddAxesInitializer(builder, "a1", {4});
  builder.MakeNode("Unsqueeze", {"x", "a0"}, {"u0"});
  builder.MakeNode("Unsqueeze", {"u0", "a1"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(InitializerInts(builder, replacements[0].input()[1].value()),
            (std::vector<int64_t>{0, 1, 4}));
}

TEST(UnsqueezeUnsqueezePattern, KeepsFirstWhenShared) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "a0", {2});
  AddAxesInitializer(builder, "a1", {3});
  builder.MakeNode("Unsqueeze", {"x", "a0"}, {"u0"});
  builder.MakeNode("Unsqueeze", {"u0", "a1"}, {"out"});
  builder.MakeNode("Identity", {"u0"}, {"other"});
  builder.MakeOutput("out");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].output()[0].value(), "u0");
  EXPECT_EQ(replacements[1].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(UnsqueezeUnsqueezePattern, RejectsNonConstantAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("a0", core::symbolic::TensorType::kInt64, Shape({1}));
  AddAxesInitializer(builder, "a1", {3});
  builder.MakeNode("Unsqueeze", {"x", "a0"}, {"u0"});
  builder.MakeNode("Unsqueeze", {"u0", "a1"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(UnsqueezeUnsqueezePattern, RejectsMissingSecondUnsqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "a0", {2});
  builder.MakeNode("Unsqueeze", {"x", "a0"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SqueezeUnsqueezePattern, CollapsesMatchingAxesToIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "axes", {2});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"u0"});
  builder.MakeNode("Squeeze", {"u0", "axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
}

TEST(SqueezeUnsqueezePattern, CollapsesSubsetToSqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 1}));
  AddAxesInitializer(builder, "unsqueeze_axes", {0});
  AddAxesInitializer(builder, "squeeze_axes", {0, 2});
  builder.MakeNode("Unsqueeze", {"x", "unsqueeze_axes"}, {"u0"});
  builder.MakeNode("Squeeze", {"u0", "squeeze_axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
  // The remaining Squeeze axis (2) shifts down by the removed Unsqueeze axis (0).
  EXPECT_EQ(InitializerInts(builder, replacements[0].input()[1].value()),
            (std::vector<int64_t>{1}));
}

TEST(SqueezeUnsqueezePattern, KeepsFirstWhenShared) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "axes", {2});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"u0"});
  builder.MakeNode("Squeeze", {"u0", "axes"}, {"out"});
  builder.MakeNode("Identity", {"u0"}, {"other"});
  builder.MakeOutput("out");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].output()[0].value(), "u0");
  EXPECT_EQ(replacements[1].op_type().value(), "Identity");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(SqueezeUnsqueezePattern, RejectsSameOperatorTwice) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddAxesInitializer(builder, "a0", {2});
  AddAxesInitializer(builder, "a1", {3});
  builder.MakeNode("Unsqueeze", {"x", "a0"}, {"u0"});
  builder.MakeNode("Unsqueeze", {"u0", "a1"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SqueezeUnsqueezePattern, RejectsDisjointAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 1}));
  AddAxesInitializer(builder, "unsqueeze_axes", {2});
  AddAxesInitializer(builder, "squeeze_axes", {1});
  builder.MakeNode("Unsqueeze", {"x", "unsqueeze_axes"}, {"u0"});
  builder.MakeNode("Squeeze", {"u0", "squeeze_axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SqueezeUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

} // namespace
} // namespace Test
