// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/expand/where_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>
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

core::symbolic::SymShape Shape2() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  return shape;
}

core::symbolic::SymShape ScalarShape() { return {}; }

void AddInt64Initializer(core::builder::GraphBuilder &builder, const std::string &name,
                         const std::vector<int64_t> &values) {
  const std::vector<int64_t> dims = values.empty()
                                        ? std::vector<int64_t>{}
                                        : std::vector<int64_t>{static_cast<int64_t>(values.size())};
  builder.MakeInitializer(MakeInitializer<int64_t>(name.c_str(), dims, values));
}

void AddFloatInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                         float value) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), {1}, {value}));
}

void AddFloatScalarInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                               float value) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), {}, {value}));
}

TEST(NotWherePattern, SwapsWhereBranches) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, Shape2());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeNode("Not", {"cond"}, {"ncond"});
  builder.MakeNode("Where", {"ncond", "x", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::NotWherePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Where");
  EXPECT_EQ(replacements[0].input()[0].value(), "cond");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[0].input()[2].value(), "x");
}

TEST(NotWherePattern, KeepsSharedNotNode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, Shape2());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeNode("Not", {"cond"}, {"ncond"});
  builder.MakeNode("Where", {"ncond", "x", "y"}, {"out"});
  builder.MakeNode("Identity", {"ncond"}, {"other"});
  builder.MakeOutput("out");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::NotWherePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Not");
  EXPECT_EQ(replacements[1].op_type().value(), "Where");
}

TEST(UnsqueezeEqualPattern, PreservesRankForMatchingUnsqueezePair) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  AddInt64Initializer(builder, "axes", {0});
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Unsqueeze", {"y", "axes"}, {"yu"});
  builder.MakeNode("Equal", {"xu", "yu"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeEqualPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Equal");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[1].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "axes");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(UnsqueezeEqualPattern, MovesEqualAfterCompatibleUnsqueeze) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  AddInt64Initializer(builder, "axes", {0});
  AddFloatInitializer(builder, "minus_one", -1.0F);
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Equal", {"x", "minus_one"}, {"equal"});
  builder.MakeNode("Unsqueeze", {"equal", "axes"}, {"out"});
  builder.MakeOutput("xu");
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeEqualPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].output()[0].value(), "xu");
  EXPECT_EQ(replacements[1].op_type().value(), "Equal");
  EXPECT_EQ(replacements[1].input()[0].value(), "xu");
  EXPECT_EQ(replacements[1].input()[1].value(), "minus_one");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(UnsqueezeEqualPattern, MovesScalarEqualWithRankZeroConstant) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ScalarShape());
  AddInt64Initializer(builder, "axes", {0});
  AddFloatScalarInitializer(builder, "minus_one", -1.0F);
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Equal", {"x", "minus_one"}, {"equal"});
  builder.MakeNode("Unsqueeze", {"equal", "axes"}, {"out"});
  builder.MakeOutput("xu");
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  ASSERT_TRUE(graph.HasShape("x"));
  ASSERT_TRUE(graph.HasShape("equal"));
  EXPECT_EQ(graph.GetShape("x").Shape().Rank(), 0u);
  EXPECT_EQ(graph.GetShape("equal").Shape().Rank(), 0u);

  onnx_patterns::UnsqueezeEqualPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[1].op_type().value(), "Equal");
  EXPECT_EQ(replacements[1].input()[0].value(), "xu");
  EXPECT_EQ(replacements[1].input()[1].value(), "minus_one");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(UnsqueezeEqualPattern, RejectsScalarEqualWithRankOneConstant) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, ScalarShape());
  AddInt64Initializer(builder, "axes", {0});
  AddFloatInitializer(builder, "minus_one", -1.0F);
  builder.MakeNode("Unsqueeze", {"x", "axes"}, {"xu"});
  builder.MakeNode("Equal", {"x", "minus_one"}, {"equal"});
  builder.MakeNode("Unsqueeze", {"equal", "axes"}, {"out"});
  builder.MakeOutput("xu");
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  ASSERT_TRUE(graph.HasShape("x"));
  ASSERT_TRUE(graph.HasShape("minus_one"));
  ASSERT_TRUE(graph.HasShape("equal"));
  ASSERT_TRUE(graph.HasShape("out"));
  EXPECT_EQ(graph.GetShape("x").Shape().Rank(), 0u);
  EXPECT_EQ(graph.GetShape("minus_one").Shape().Rank(), 1u);
  EXPECT_EQ(graph.GetShape("equal").Shape().Rank(), 1u);
  EXPECT_EQ(graph.GetShape("out").Shape().Rank(), 2u);

  onnx_patterns::UnsqueezeEqualPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason,
            "the constant is not rank zero and Equal does not preserve the input rank");
}

TEST(UnsqueezeEqualPattern, RejectsDifferentAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  AddInt64Initializer(builder, "axes0", {0});
  AddInt64Initializer(builder, "axes1", {1});
  builder.MakeNode("Unsqueeze", {"x", "axes0"}, {"xu"});
  builder.MakeNode("Unsqueeze", {"y", "axes1"}, {"yu"});
  builder.MakeNode("Equal", {"xu", "yu"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeEqualPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Unsqueeze axes differ");
}

TEST(UnsqueezeEqualPattern, RejectsDifferentUpstreamAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  AddInt64Initializer(builder, "axes0", {0});
  AddInt64Initializer(builder, "axes1", {1});
  AddFloatInitializer(builder, "minus_one", -1.0F);
  builder.MakeNode("Unsqueeze", {"x", "axes0"}, {"xu"});
  builder.MakeNode("Equal", {"x", "minus_one"}, {"equal"});
  builder.MakeNode("Unsqueeze", {"equal", "axes1"}, {"out"});
  builder.MakeOutput("xu");
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::UnsqueezeEqualPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the sibling and result Unsqueeze axes differ");
}

TEST(WhereAddPattern, ReplacesUpstreamAdditiveMaskInBothAddOrders) {
  for (bool where_first : {true, false}) {
    core::builder::GraphBuilder builder(where_first ? "left" : "right", SchemaLookup());
    builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape2());
    AddFloatInitializer(builder, "x", 0.25F);
    AddFloatInitializer(builder, "zero", 0.0F);
    AddFloatInitializer(builder, "minus_inf", -std::numeric_limits<float>::infinity());
    builder.MakeNode("Where", {"mask", "zero", "minus_inf"}, {"masked"});
    builder.MakeNode("Add",
                     where_first ? std::vector<std::string>{"masked", "x"}
                                 : std::vector<std::string>{"x", "masked"},
                     {"out"});
    builder.MakeOutput("out");

    core::builder::GraphGraph graph(builder);
    onnx_patterns::WhereAddPattern pattern;
    const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
    ASSERT_EQ(match.pattern, &pattern) << (where_first ? "left" : "right");

    const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
    ASSERT_EQ(replacements.size(), 1u);
    EXPECT_EQ(replacements[0].op_type().value(), "Where");
    EXPECT_EQ(replacements[0].input()[0].value(), "mask");
    EXPECT_EQ(replacements[0].input()[1].value(), "x");
    EXPECT_EQ(replacements[0].input()[2].value(), "minus_inf");
    EXPECT_EQ(replacements[0].output()[0].value(), "out");
  }
}

TEST(WhereAddPattern, RejectsPositiveInfinityAddend) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape2());
  AddFloatInitializer(builder, "x", std::numeric_limits<float>::infinity());
  AddFloatInitializer(builder, "zero", 0.0F);
  AddFloatInitializer(builder, "minus_inf", -std::numeric_limits<float>::infinity());
  builder.MakeNode("Where", {"mask", "zero", "minus_inf"}, {"masked"});
  builder.MakeNode("Add", {"masked", "x"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::WhereAddPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(WhereAddPattern, RejectsWrongUpstreamWhereConstants) {
  for (bool positive_infinity : {false, true}) {
    core::builder::GraphBuilder builder(positive_infinity ? "positive_inf" : "nonzero",
                                        SchemaLookup());
    builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape2());
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
    AddFloatInitializer(builder, "then", positive_infinity ? 0.0F : 1.0F);
    AddFloatInitializer(builder, "else",
                        positive_infinity ? std::numeric_limits<float>::infinity()
                                          : -std::numeric_limits<float>::infinity());
    builder.MakeNode("Where", {"mask", "then", "else"}, {"masked"});
    builder.MakeNode("Add", {"masked", "x"}, {"out"});
    builder.MakeOutput("out");

    core::builder::GraphGraph graph(builder);
    onnx_patterns::WhereAddPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  }
}

TEST(WhereAddPattern, RejectsExternallyUsedUpstreamWhereOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape2());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  AddFloatInitializer(builder, "zero", 0.0F);
  AddFloatInitializer(builder, "minus_inf", -std::numeric_limits<float>::infinity());
  builder.MakeNode("Where", {"mask", "zero", "minus_inf"}, {"masked"});
  builder.MakeNode("Add", {"masked", "x"}, {"out"});
  builder.MakeOutput("masked");
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::WhereAddPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(WhereAddPattern, FactorsSharedAddInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, Shape2());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeNode("Add", {"x", "z"}, {"tx"});
  builder.MakeNode("Add", {"y", "z"}, {"ty"});
  builder.MakeNode("Where", {"cond", "tx", "ty"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::WhereAddPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Where");
  EXPECT_EQ(replacements[0].input()[0].value(), "cond");
  EXPECT_EQ(replacements[0].input()[1].value(), "x");
  EXPECT_EQ(replacements[0].input()[2].value(), "y");
  EXPECT_EQ(replacements[1].op_type().value(), "Add");
  EXPECT_EQ(replacements[1].input()[1].value(), "z");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(WhereAddPattern, RejectsBranchesWithoutCommonAddInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, Shape2());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeNode("Add", {"x", "z"}, {"tx"});
  builder.MakeNode("Add", {"y", "w"}, {"ty"});
  builder.MakeNode("Where", {"cond", "tx", "ty"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::WhereAddPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "Where branches are not two Add nodes sharing one input");
}

TEST(WhereAddPattern, OptimizeProducesTwoNodeGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, Shape2());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, Shape2());
  builder.MakeNode("Add", {"x", "z"}, {"tx"});
  builder.MakeNode("Add", {"y", "z"}, {"ty"});
  builder.MakeNode("Where", {"cond", "tx", "ty"}, {"out"});
  builder.MakeOutput("out");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::WhereAddPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Where");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Add");
}

} // namespace
} // namespace Test
