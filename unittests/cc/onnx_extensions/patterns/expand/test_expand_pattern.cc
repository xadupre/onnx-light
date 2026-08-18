// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/expand/expand_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>
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

void AddInt64Initializer(core::builder::GraphBuilder &builder, const std::string &name,
                         const std::vector<int64_t> &values) {
  const std::vector<int64_t> dims{static_cast<int64_t>(values.size())};
  builder.MakeInitializer(MakeInitializer<int64_t>(name.c_str(), dims, values));
}

std::vector<int64_t> InitializerValues(const core::builder::GraphBuilder &builder,
                                       const std::string &name) {
  for (const TensorProto &tensor : builder.Initializers()) {
    if (tensor.name().value() == name) {
      std::vector<int64_t> values;
      ReadIntegerValues(tensor, values);
      return values;
    }
  }
  return {};
}

TEST(ExpandPattern, RemovesRedundantExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddInt64Initializer(builder, "shape", {2, 3});
  builder.MakeNode("Expand", {"x", "shape"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
}

TEST(ExpandPattern, RejectsShapeChangingExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3}));
  AddInt64Initializer(builder, "shape", {2, 3});
  builder.MakeNode("Expand", {"x", "shape"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ExpandBroadcastPattern, DropsExpandBeforeBinary) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 1024, 1}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 1024, 1024}));
  AddInt64Initializer(builder, "shape", {2, 1024, 1024});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Mul", {"xe", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandBroadcastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Mul");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
}

TEST(ExpandBroadcastPattern, KeepsExpandWhenOperandsDoNotBroadcast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 1}));
  AddInt64Initializer(builder, "shape", {2, 3});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Mul", {"xe", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandBroadcastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ShapeBasedConcatExpandPattern, ReplacesUnchangedTargetDimensionsWithOnes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({core::symbolic::SymDim("a"), 1}));
  AddInt64Initializer(builder, "two", {2});
  NodeProto shape = MakeNode("Shape", {"x"}, {"first"});
  AddAttribute<int64_t>(shape, "start", 0);
  AddAttribute<int64_t>(shape, "end", 1);
  builder.MakeNode("Shape", {"x"}, {"first"}, "", "", shape.attribute());
  NodeProto concat = MakeNode("Concat", {"first", "two"}, {"target"});
  AddAttribute<int64_t>(concat, "axis", 0);
  builder.MakeNode("Concat", {"first", "two"}, {"target"}, "", "", concat.attribute());
  builder.MakeNode("Expand", {"x", "target"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedConcatExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Concat");
  ASSERT_EQ(replacements[0].input_size(), 2);
  EXPECT_EQ(InitializerValues(builder, replacements[0].input()[0].value()),
            (std::vector<int64_t>{1}));
  EXPECT_EQ(replacements[0].input()[1].value(), "two");
  EXPECT_EQ(replacements[1].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].input()[1].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(ShapeBasedConcatExpandPattern, RejectsTwoChangedDimensions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 1}));
  AddInt64Initializer(builder, "two", {2});
  AddInt64Initializer(builder, "three", {3});
  NodeProto concat = MakeNode("Concat", {"two", "three"}, {"target"});
  AddAttribute<int64_t>(concat, "axis", 0);
  builder.MakeNode("Concat", {"two", "three"}, {"target"}, "", "", concat.attribute());
  builder.MakeNode("Expand", {"x", "target"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedConcatExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ShapeBasedConcatExpandPattern, RejectsSharedTargetShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({core::symbolic::SymDim("a"), 1}));
  AddInt64Initializer(builder, "two", {2});
  NodeProto shape = MakeNode("Shape", {"x"}, {"first"});
  AddAttribute<int64_t>(shape, "start", 0);
  AddAttribute<int64_t>(shape, "end", 1);
  builder.MakeNode("Shape", {"x"}, {"first"}, "", "", shape.attribute());
  NodeProto concat = MakeNode("Concat", {"first", "two"}, {"target"});
  AddAttribute<int64_t>(concat, "axis", 0);
  builder.MakeNode("Concat", {"first", "two"}, {"target"}, "", "", concat.attribute());
  builder.MakeNode("Expand", {"x", "target"}, {"out"});
  builder.MakeNode("Identity", {"target"}, {"saved_target"});
  builder.MakeOutput("out");
  builder.MakeOutput("saved_target");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedConcatExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ShapeBasedExpandBroadcastPattern, RemovesDynamicExpandBeforeBinary) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput(
      "x", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape({1, core::symbolic::SymDim("b"), core::symbolic::SymDim("c")}));
  builder.MakeInput(
      "y", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape(
          {core::symbolic::SymDim("a"), core::symbolic::SymDim("b"), core::symbolic::SymDim("c")}));
  builder.MakeNode("Shape", {"y"}, {"target"});
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  builder.MakeNode("Add", {"xe", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandBroadcastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[1]);
  EXPECT_EQ(match.nodes[1], nullptr);
  EXPECT_EQ(match.nodes[2], &builder.Nodes()[2]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Add");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
}

TEST(ShapeBasedExpandBroadcastPattern, PreservesSharedExpandForOtherConsumers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({1, core::symbolic::SymDim("b")}));
  builder.MakeInput(
      "y", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape({core::symbolic::SymDim("a"), core::symbolic::SymDim("b")}));
  builder.MakeNode("Shape", {"y"}, {"target"});
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  builder.MakeNode("Mul", {"xe", "y"}, {"out"});
  builder.MakeOutput("out");
  builder.MakeOutput("xe");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandBroadcastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[1]);
  EXPECT_EQ(match.nodes[1], nullptr);
  EXPECT_EQ(match.nodes[2], &builder.Nodes()[2]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].input()[1].value(), "y");

  auto optimized_pattern = std::make_shared<onnx_patterns::ShapeBasedExpandBroadcastPattern>();
  core::builder::GraphGraph optimizer(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{optimized_pattern});
  const std::vector<core::builder::LocalRewriting> rewrites = optimizer.Optimize(1);
  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(rewrites[0].matched_nodes, (std::vector<std::size_t>{1, 2}));
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Expand");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Mul");
  EXPECT_EQ(builder.Nodes()[2].input()[0].value(), "x");
}

TEST(ShapeBasedExpandBroadcastPattern, RejectsIncompatibleSymbolicDimensions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput(
      "x", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape({core::symbolic::SymDim("m"), core::symbolic::SymDim("b")}));
  builder.MakeInput(
      "y", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape({core::symbolic::SymDim("a"), core::symbolic::SymDim("b")}));
  builder.MakeNode("Shape", {"y"}, {"target"});
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  builder.MakeNode("Add", {"xe", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandBroadcastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ShapeBasedExpandBroadcastMatMulPattern, RemovesDynamicBatchExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput(
      "x", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape(
          {core::symbolic::SymDim("a"), core::symbolic::SymDim("b"), core::symbolic::SymDim("c")}));
  builder.MakeInput(
      "y", core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape({1, core::symbolic::SymDim("c"), core::symbolic::SymDim("d")}));
  NodeProto shape = MakeNode("Shape", {"x"}, {"batch"});
  AddAttribute<int64_t>(shape, "start", 0);
  AddAttribute<int64_t>(shape, "end", 1);
  builder.MakeNode("Shape", {"x"}, {"batch"}, "", "", shape.attribute());
  NodeProto matrix_shape = MakeNode("Shape", {"y"}, {"matrix"});
  AddAttribute<int64_t>(matrix_shape, "start", 1);
  builder.MakeNode("Shape", {"y"}, {"matrix"}, "", "", matrix_shape.attribute());
  NodeProto concat = MakeNode("Concat", {"batch", "matrix"}, {"target"});
  AddAttribute<int64_t>(concat, "axis", 0);
  builder.MakeNode("Concat", {"batch", "matrix"}, {"target"}, "", "", concat.attribute());
  builder.MakeNode("Expand", {"y", "target"}, {"ye"});
  builder.MakeNode("MatMul", {"x", "ye"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandBroadcastMatMulPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[4]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
}

TEST(ShapeBasedExpandBroadcastMatMulPattern, RejectsIncompatibleBatchDimensions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({core::symbolic::SymDim("a"), 2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({core::symbolic::SymDim("m"), 3, 4}));
  builder.MakeInput("target", core::symbolic::TensorType::kInt64, Shape({3}));
  builder.MakeNode("Expand", {"y", "target"}, {"ye"});
  builder.MakeNode("MatMul", {"x", "ye"}, {"out"});
  builder.MakeOutput("out", core::symbolic::TensorType::kFloat,
                     core::symbolic::SymShape({core::symbolic::SymDim("a"), 2, 4}));

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandBroadcastMatMulPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ShapeBasedStaticExpandPattern, ReplacesDynamicTargetWithConstantShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({2, 3, core::symbolic::SymDim("d"), 1}));
  AddInt64Initializer(builder, "four", {4});
  NodeProto shape = MakeNode("Shape", {"x"}, {"prefix"});
  AddAttribute<int64_t>(shape, "start", 0);
  AddAttribute<int64_t>(shape, "end", -1);
  builder.MakeNode("Shape", {"x"}, {"prefix"}, "", "", shape.attribute());
  NodeProto concat = MakeNode("Concat", {"prefix", "four"}, {"target"});
  AddAttribute<int64_t>(concat, "axis", 0);
  builder.MakeNode("Concat", {"prefix", "four"}, {"target"}, "", "", concat.attribute());
  builder.MakeNode("Expand", {"x", "target"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedStaticExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Expand");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(InitializerValues(builder, replacements[0].input()[1].value()),
            (std::vector<int64_t>{1, 1, 1, 4}));
}

TEST(ShapeBasedStaticExpandPattern, RejectsSymbolicChangedDimension) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    core::symbolic::SymShape({core::symbolic::SymDim("a"), 1}));
  builder.MakeInput("target", core::symbolic::TensorType::kInt64, Shape({2}));
  builder.MakeNode("Expand", {"x", "target"}, {"out"});
  builder.MakeOutput("out", core::symbolic::TensorType::kFloat,
                     core::symbolic::SymShape({core::symbolic::SymDim("b"), 4}));

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedStaticExpandPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ShapeBasedExpandSwapPattern, MovesBinaryBeforeExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({4, 1}));
  AddInt64Initializer(builder, "target", {4, 4});
  builder.MakeInitializer(MakeInitializer<float>("one", {1}, {1.0f}));
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  builder.MakeNode("Add", {"xe", "one"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandSwapPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[1], nullptr);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Add");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[1].value(), "target");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(ShapeBasedExpandSwapPattern, RejectsRedundantInputExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({4, 4}));
  AddInt64Initializer(builder, "target", {4, 4});
  builder.MakeInitializer(MakeInitializer<float>("one", {1}, {1.0f}));
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  builder.MakeNode("Add", {"xe", "one"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandSwapPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(ShapeBasedExpandCastWhereSwapPattern, MovesWhereBeforeExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 1}));
  AddInt64Initializer(builder, "target", {2, 3});
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  NodeProto cast = MakeNode("Cast", {"xe"}, {"condition"});
  AddAttribute<int64_t>(cast, "to", static_cast<int64_t>(TensorProto::DataType::BOOL));
  builder.MakeNode("Cast", {"xe"}, {"condition"}, "", "", cast.attribute());
  builder.MakeNode("Where", {"condition", "xe", "zero"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandCastWhereSwapPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Where");
  EXPECT_EQ(replacements[1].input()[1].value(), "x");
  EXPECT_EQ(replacements[2].op_type().value(), "Expand");
  EXPECT_EQ(replacements[2].input()[1].value(), "target");
  EXPECT_EQ(replacements[2].output()[0].value(), "out");
}

TEST(ShapeBasedExpandCastWhereSwapPattern, RejectsAdditionalExpandConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 1}));
  AddInt64Initializer(builder, "target", {2, 3});
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Expand", {"x", "target"}, {"xe"});
  NodeProto cast = MakeNode("Cast", {"xe"}, {"condition"});
  AddAttribute<int64_t>(cast, "to", static_cast<int64_t>(TensorProto::DataType::BOOL));
  builder.MakeNode("Cast", {"xe"}, {"condition"}, "", "", cast.attribute());
  builder.MakeNode("Where", {"condition", "xe", "zero"}, {"out"});
  builder.MakeNode("Identity", {"xe"}, {"saved"});
  builder.MakeOutput("out");
  builder.MakeOutput("saved");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedExpandCastWhereSwapPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(ExpandSwapPattern, MovesExpandPastUnary) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 5, 7}));
  AddInt64Initializer(builder, "shape", {3, 1, 1});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Exp", {"xe"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandSwapPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Exp");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].input()[1].value(), "shape");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(ExpandSwapPattern, RejectsNonUnaryConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 5, 7}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3, 5, 7}));
  AddInt64Initializer(builder, "shape", {3, 1, 1});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Add", {"xe", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandSwapPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SwapExpandUnsqueezePattern, MovesUnsqueezeBeforeExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 5, 7}));
  AddInt64Initializer(builder, "shape", {3, 1, 1});
  AddInt64Initializer(builder, "axes", {1});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Unsqueeze", {"xe", "axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapExpandUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "axes");
  EXPECT_EQ(replacements[1].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
  EXPECT_EQ(InitializerValues(builder, replacements[1].input()[1].value()),
            (std::vector<int64_t>{3, 1, 1, 1}));
}

TEST(SwapExpandUnsqueezePattern, RejectsNonUnsqueezeConsumer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 5, 7}));
  AddInt64Initializer(builder, "shape", {3, 1, 1});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Exp", {"xe"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapExpandUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SwapExpandUnsqueezePattern, RejectsNonConstantAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 5, 7}));
  builder.MakeInput("axes", core::symbolic::TensorType::kInt64, Shape({1}));
  AddInt64Initializer(builder, "shape", {3, 1, 1});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Unsqueeze", {"xe", "axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapExpandUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SwapExpandUnsqueezePattern, RejectsOpsetBelow13) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 11);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 5, 7}));
  AddInt64Initializer(builder, "shape", {3, 1, 1});
  AddInt64Initializer(builder, "axes", {1});
  builder.MakeNode("Expand", {"x", "shape"}, {"xe"});
  builder.MakeNode("Unsqueeze", {"xe", "axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapExpandUnsqueezePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ExpandUnsqueezeExpandPattern, FusesExpandUnsqueezeExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3, 4}));
  AddInt64Initializer(builder, "shape1", {5, 3, 4});
  AddInt64Initializer(builder, "axes", {1});
  AddInt64Initializer(builder, "shape2", {5, 2, 3, 4});
  builder.MakeNode("Expand", {"x", "shape1"}, {"xe"});
  builder.MakeNode("Unsqueeze", {"xe", "axes"}, {"xu"});
  builder.MakeNode("Expand", {"xu", "shape2"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandUnsqueezeExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "axes");
  EXPECT_EQ(replacements[1].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
  EXPECT_EQ(InitializerValues(builder, replacements[1].input()[1].value()),
            (std::vector<int64_t>{5, 2, 3, 4}));
}

TEST(ExpandUnsqueezeExpandPattern, RejectsMissingSecondExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3, 4}));
  AddInt64Initializer(builder, "shape1", {5, 3, 4});
  AddInt64Initializer(builder, "axes", {1});
  builder.MakeNode("Expand", {"x", "shape1"}, {"xe"});
  builder.MakeNode("Unsqueeze", {"xe", "axes"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandUnsqueezeExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(ExpandUnsqueezeExpandPattern, RejectsRankMismatch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3, 4}));
  AddInt64Initializer(builder, "shape1", {5, 3, 4});
  AddInt64Initializer(builder, "axes", {1, 2});
  AddInt64Initializer(builder, "shape2", {5, 2, 3, 4});
  builder.MakeNode("Expand", {"x", "shape1"}, {"xe"});
  builder.MakeNode("Unsqueeze", {"xe", "axes"}, {"xu"});
  builder.MakeNode("Expand", {"xu", "shape2"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ExpandUnsqueezeExpandPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
}

TEST(SwapExpandReshapePattern, MovesReshapeBeforeExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 4, 1}));
  AddInt64Initializer(builder, "expand_shape", {3, 1, 1});
  AddInt64Initializer(builder, "reshape_shape", {0, 1, -1});
  builder.MakeNode("Expand", {"x", "expand_shape"}, {"xe"});
  builder.MakeNode("Reshape", {"xe", "reshape_shape"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapExpandReshapePattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Expand");
  EXPECT_EQ(replacements[1].input()[1].value(), "expand_shape");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(SwapExpandReshapePattern, RejectsExpandTargetWithoutTrailingOnes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 1, 1}));
  AddInt64Initializer(builder, "expand_shape", {3, 2, 1});
  AddInt64Initializer(builder, "reshape_shape", {0, 1, -1});
  builder.MakeNode("Expand", {"x", "expand_shape"}, {"xe"});
  builder.MakeNode("Reshape", {"xe", "reshape_shape"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapExpandReshapePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

} // namespace
} // namespace Test
