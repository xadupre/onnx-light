// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/cast_cast_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"

#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

#include <algorithm>
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

core::symbolic::SymShape Shape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  return shape;
}

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType to) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("to");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(static_cast<int64_t>(to));
  builder.MakeNode("Cast", {input}, {output}, "", "", attributes);
}

} // namespace

TEST(PatternOptimization, CastCastCollapsesRedundantOuterCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  EXPECT_EQ(pattern.priority, 1);
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Cast"}));
  EXPECT_EQ(pattern.ToString(),
            "PatternOptimization(name=CastCast, priority=1, fast_op_types=[Cast])");

  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);
  EXPECT_EQ(match.insert_at, match.nodes[1]);
  EXPECT_EQ(match.ToString(), "MatchResult(pattern=CastCast, nodes=[Cast(outputs=[middle]), "
                              "Cast(outputs=[y])], insert_at=Cast(outputs=[y]))");

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  ASSERT_NE(FindAttribute(replacements[0], "to"), nullptr);
  EXPECT_EQ(FindAttribute(replacements[0], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
}

TEST(PatternOptimization, AcceptsCustomPriority) {
  onnx_patterns::CastCastPattern pattern(3);
  EXPECT_EQ(pattern.priority, 3);
}

TEST(PatternOptimization, CastCastUsesIdentityForSafeRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_TRUE(replacements[0].attribute().empty());
}

TEST(PatternOptimization, CastCastKeepsSharedInnerCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"middle"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, nullptr);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].output()[0].value(), "middle");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
}

TEST(PatternOptimization, CastCastRejectsLossyRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::INT32);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  EXPECT_TRUE(match.nodes.empty());
}

TEST(PatternOptimization, OptimizeAppliesPatternAndCleanupPasses) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"x"}, {"unused"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize();
  ASSERT_EQ(rewrites.size(), 1u);
  ASSERT_NE(rewrites[0].pattern, nullptr);
  EXPECT_EQ(rewrites[0].pattern->Name(), "CastCast");
  EXPECT_EQ(rewrites[0].matched_nodes, std::vector<std::size_t>({0, 1}));
  EXPECT_EQ(rewrites[0].added_nodes.size(), 1u);
  EXPECT_TRUE(rewrites[0].added_initializers.empty());
  EXPECT_EQ(rewrites[0].insert_at, 1u);
  EXPECT_EQ(rewrites[0].iteration, 0u);
  EXPECT_EQ(rewrites[0].ToString(), "LocalRewriting(pattern=CastCast, matched_nodes=[0, 1], "
                                    "added_nodes=[Cast(outputs=[y])], added_initializers=[], "
                                    "insert_at=1, iteration=0)");
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Cast");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(PatternOptimization, OptimizeAppliesDisjointMatchesInOneIteration) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x1", core::symbolic::TensorType::kFloat16, Shape());
  builder.MakeInput("x2", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x1", "middle1", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle1", "y1", TensorProto::DataType::FLOAT);
  AddCast(builder, "x2", "middle2", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle2", "y2", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y1");
  builder.MakeOutput("y2");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(1);
  ASSERT_EQ(rewrites.size(), 2u);
  EXPECT_EQ(rewrites[0].iteration, 0u);
  EXPECT_EQ(rewrites[1].iteration, 0u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x1");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "x2");

  const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
  ASSERT_EQ(replayed.node_size(), 2);
  EXPECT_EQ(replayed.node(0).input()[0].value(), "x1");
  EXPECT_EQ(replayed.node(0).output()[0].value(), "y1");
  EXPECT_EQ(replayed.node(1).input()[0].value(), "x2");
  EXPECT_EQ(replayed.node(1).output()[0].value(), "y2");
}

TEST(PatternOptimization, ReplayRestoresAddedInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();

  core::builder::LocalRewriting rewrite;
  rewrite.pattern = std::make_shared<onnx_patterns::CastCastPattern>();
  rewrite.matched_nodes = {0};
  TensorProto &weight = rewrite.added_initializers.add();
  weight.set_name("weight");
  weight.set_data_type(TensorProto::DataType::FLOAT);
  weight.add_dims(2);
  weight.add_float_data(1.0f);
  weight.add_float_data(2.0f);
  rewrite.added_nodes.add() = MakeNode("Add", {"x", "weight"}, {"y"});

  const GraphProto replayed = core::builder::Replay(
      original, std::vector<core::builder::LocalRewriting>{rewrite}, SchemaLookup());
  ASSERT_EQ(replayed.initializer_size(), 1);
  EXPECT_EQ(replayed.initializer(0).name().value(), "weight");
  ASSERT_EQ(replayed.node_size(), 1);
  EXPECT_EQ(replayed.node(0).op_type().value(), "Add");
  EXPECT_EQ(replayed.node(0).input()[1].value(), "weight");
}

TEST(PatternOptimization, LocalRewritingKeepsPatternAlive) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<core::builder::LocalRewriting> rewrites;
  {
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    rewrites = graph.Optimize();
  }

  ASSERT_EQ(rewrites.size(), 1u);
  ASSERT_NE(rewrites[0].pattern, nullptr);
  EXPECT_EQ(rewrites[0].pattern->Name(), "CastCast");
}

TEST(PatternOptimization, OptimizeKeepsSharedInnerCastInTopologicalOrder) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"middle"}, {"other"});
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  EXPECT_EQ(graph.Optimize().size(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "middle");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "y");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Neg");
}

TEST(PatternOptimization, OptimizeHonorsDoNotRemovePredicate) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns), [](const NodeProto &node) {
    return node.output()[0].value() == "middle";
  });

  EXPECT_TRUE(graph.Optimize().empty());
  EXPECT_EQ(builder.Nodes().size(), 2u);
}

TEST(PatternOptimization, RegistersBuiltInPatternsOnce) {
  onnx_patterns::RegisterPatterns();
  const std::vector<std::string> names = core::builder::RegisteredPatternNames();
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastCast"), 1);

  onnx_patterns::RegisterPatterns();
  EXPECT_EQ(core::builder::RegisteredPatternNames(), names);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns =
      core::builder::CreateRegisteredPatterns();
  const bool found = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::CastCastPattern *>(pattern.get()) != nullptr;
  });
  EXPECT_TRUE(found);
}

} // namespace Test
