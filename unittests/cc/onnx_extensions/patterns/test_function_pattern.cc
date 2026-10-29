// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/function_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

using core::builder::BuilderError;
using core::builder::GraphBuilder;
using core::builder::GraphGraph;
using onnx_patterns::FunctionPattern;

GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

FunctionProto Function(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs,
                       const std::vector<NodeProto> &nodes) {
  FunctionProto function;
  function.set_name("body");
  function.set_domain("patterns");
  for (const auto &input : inputs)
    function.add_input(input);
  for (const auto &output : outputs)
    function.add_output(output);
  for (const auto &node : nodes)
    function.add_node(node);
  auto &opset = *function.add_opset_import();
  opset.set_domain("");
  opset.set_version(18);
  return function;
}

FunctionProto DoubleNeg() {
  return Function({"x"}, {"y"},
                  {MakeNode("Neg", {"x"}, {"middle"}), MakeNode("Neg", {"middle"}, {"y"})});
}

FunctionProto Identity() {
  return Function({"input"}, {"output"}, {MakeNode("Identity", {"input"}, {"output"})});
}

void AddInput(GraphBuilder &builder, const std::string &name) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  builder.MakeInput(name, core::symbolic::TensorType::kFloat, shape);
}

NodeProto Transpose(const std::string &input, const std::string &output, bool reference,
                    const std::vector<int64_t> &perm = {0}) {
  NodeProto node = MakeNode("Transpose", {input}, {output});
  auto &attribute = *node.add_attribute();
  attribute.set_name("perm");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  if (reference)
    attribute.set_ref_attr_name("permutation");
  else
    for (const auto value : perm)
      attribute.add_ints(value);
  return node;
}

TEST(FunctionPattern, MatchesAndRevalidatesApply) {
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  AddInput(builder, "a");
  builder.MakeNode("Neg", {"a"}, {"b"});
  builder.MakeNode("Neg", {"b"}, {"result"});
  builder.MakeOutput("result");
  GraphGraph graph(builder);
  FunctionPattern pattern(DoubleNeg(), Identity());
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>{"Neg"});
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);
  const auto result = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].op_type(), "Identity");
  EXPECT_EQ(result[0].input(0), "a");
  EXPECT_EQ(result[0].output(0), "result");
  EXPECT_THROW(pattern.Apply(graph, {}), BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {match.nodes[1], match.nodes[0]}), BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {nullptr, match.nodes[1]}), BuilderError);
}

TEST(FunctionPattern, SelectsAnAnchorAndMatchesForwardBranches) {
  const auto body =
      Function({"x"}, {"left", "right"},
               {MakeNode("Neg", {"x"}, {"middle"}), MakeNode("Abs", {"middle"}, {"left"}),
                MakeNode("Relu", {"middle"}, {"right"})});
  for (const int anchor : {-1, 0, 1, 2}) {
    SCOPED_TRACE(anchor);
    GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    AddInput(builder, "input");
    builder.MakeNode("Neg", {"input"}, {"shared"});
    builder.MakeNode("Abs", {"shared"}, {"first"});
    builder.MakeNode("Relu", {"shared"}, {"second"});
    builder.MakeOutput("first");
    builder.MakeOutput("second");
    auto pattern = std::make_shared<FunctionPattern>(body, body, 1, "Anchored",
                                                     core::symbolic::Device::kUndefined, anchor);
    GraphGraph graph(builder,
                     std::vector<std::shared_ptr<core::builder::PatternOptimization>>{pattern});
    const std::size_t selected = anchor == -1 ? 2 : anchor;
    EXPECT_EQ(pattern->FastOpType(),
              std::set<std::string>{body.node()[selected].op_type().value()});
    const auto match = pattern->Match(graph, builder.Nodes()[selected]);
    ASSERT_EQ(match.pattern, pattern.get());
    EXPECT_EQ(match.insert_at, &builder.Nodes()[2]);
    for (std::size_t index = 0; index < 3; ++index)
      EXPECT_EQ(match.nodes[index], &builder.Nodes()[index]);
    EXPECT_EQ(pattern->Apply(graph, match.nodes).size(), 3u);
    auto invalid = match.nodes;
    invalid[selected] = nullptr;
    EXPECT_THROW(pattern->Apply(graph, invalid), BuilderError);
    EXPECT_EQ(graph.Optimize(1).rewrites().size(), 1u);
    EXPECT_EQ(builder.Outputs()[0].name(), "first");
    EXPECT_EQ(builder.Outputs()[1].name(), "second");
  }
  for (const int invalid : {-2, 3})
    EXPECT_THROW(
        FunctionPattern(body, body, 1, "Invalid", core::symbolic::Device::kUndefined, invalid),
        BuilderError);
}

TEST(FunctionPattern, OptimizesAndPreservesOutput) {
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  AddInput(builder, "a");
  builder.MakeNode("Neg", {"a"}, {"b"});
  builder.MakeNode("Neg", {"b"}, {"result"});
  builder.MakeOutput("result");
  std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns{
      std::make_shared<FunctionPattern>(DoubleNeg(), Identity())};
  GraphGraph graph(builder, patterns);
  graph.Optimize();
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input(0), "a");
  EXPECT_EQ(builder.Nodes()[0].output(0), "result");
}

TEST(FunctionPattern, RejectsEscapingIntermediate) {
  for (const bool graph_output : {false, true}) {
    GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    AddInput(builder, "a");
    builder.MakeNode("Neg", {"a"}, {"b"});
    builder.MakeNode("Neg", {"b"}, {"result"});
    builder.MakeOutput("result");
    if (graph_output)
      builder.MakeOutput("b");
    else {
      builder.MakeNode("Abs", {"b"}, {"shared"});
      builder.MakeOutput("shared");
    }
    GraphGraph graph(builder);
    FunctionPattern pattern(DoubleNeg(), Identity());
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
  }
}

TEST(FunctionPattern, CapturesRepeatedReferencesAndCopiesDefinitions) {
  auto body =
      Function({"x"}, {"y"}, {Transpose("x", "middle", true), Transpose("middle", "y", true)});
  body.add_attribute("permutation");
  FunctionPattern pattern(body, Identity());
  body.ref_node().clear();
  for (const bool equal : {false, true}) {
    GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(2));
    shape.PushBack(core::symbolic::SymDim(2));
    builder.MakeInput("a", core::symbolic::TensorType::kFloat, shape);
    builder.ImportNode(Transpose("a", "b", false, {1, 0}));
    builder.ImportNode(Transpose("b", "result", false,
                                 equal ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{0, 1}));
    builder.MakeOutput("result");
    GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern != nullptr, equal);
  }
}

TEST(FunctionPattern, TransfersReferencesUnderAnotherAttributeName) {
  auto body = Function({"x"}, {"y"}, {Transpose("x", "y", true)});
  body.add_attribute("permutation");
  NodeProto replacement_node = MakeNode("Custom", {"a"}, {"b"}, "custom");
  auto &attribute = *replacement_node.add_attribute();
  attribute.set_name("axes");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  attribute.set_ref_attr_name("permutation");
  auto replacement = Function({"a"}, {"b"}, {replacement_node});
  replacement.add_attribute("permutation");
  auto &opset = *replacement.add_opset_import();
  opset.set_domain("custom");
  opset.set_version(1);
  FunctionPattern pattern(body, replacement);
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"custom", 1}});
  AddInput(builder, "input");
  builder.ImportNode(Transpose("input", "output", false));
  builder.MakeOutput("output");
  GraphGraph graph(builder);
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto result = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(result[0].attribute_size(), 1);
  EXPECT_EQ(result[0].attribute(0).name(), "axes");
  EXPECT_TRUE(result[0].attribute(0).ref_attr_name().empty());
  ASSERT_EQ(result[0].attribute(0).ints_size(), 1);
  EXPECT_EQ(result[0].attribute(0).ints(0), 0);
}

TEST(FunctionPattern, MatchesBranchesAndMultipleOutputs) {
  const auto body =
      Function({"x"}, {"negative", "positive"},
               {MakeNode("Neg", {"x"}, {"negative"}), MakeNode("Neg", {"x"}, {"positive"})});
  const auto replacement = Function({"a"}, {"b", "b"}, {MakeNode("Neg", {"a"}, {"b"})});
  FunctionPattern pattern(body, replacement);
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  AddInput(builder, "input");
  builder.MakeNode("Neg", {"input"}, {"first"});
  builder.MakeNode("Neg", {"input"}, {"second"});
  builder.MakeOutput("first");
  builder.MakeOutput("second");
  GraphGraph graph(builder);
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto result = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].output(0), "first");
  EXPECT_EQ(result[1].output(0), "second");
  EXPECT_EQ(result[1].input(0), result[0].output(0));
}

TEST(FunctionPattern, RejectsInvalidDefinitions) {
  auto body = DoubleNeg();
  auto replacement = Identity();
  body.clear_name();
  EXPECT_THROW(FunctionPattern(body, replacement), std::invalid_argument);
  body = DoubleNeg();
  replacement.add_input("extra");
  EXPECT_THROW(FunctionPattern(body, replacement), BuilderError);
  replacement = Identity();
  replacement.add_output("output");
  EXPECT_THROW(FunctionPattern(body, replacement), BuilderError);
  body.ref_node()[0].ref_input()[0] = "undefined";
  EXPECT_THROW(FunctionPattern(body, Identity()), std::invalid_argument);
  body = DoubleNeg();
  body.ref_node()[1].ref_output()[0] = "x";
  EXPECT_THROW(FunctionPattern(body, Identity()), std::invalid_argument);
  body = Function({"x"}, {"y"}, {Transpose("x", "y", true)});
  EXPECT_THROW(FunctionPattern(body, Identity()), std::invalid_argument);
  body.add_attribute("permutation");
  replacement = body;
  replacement.ref_node()[0].ref_attribute()[0].set_ref_attr_name("uncaptured");
  replacement.add_attribute("uncaptured");
  EXPECT_THROW(FunctionPattern(body, replacement), BuilderError);
  body.ref_node()[0].ref_attribute()[0].set_type(AttributeProto::AttributeType::GRAPH);
  EXPECT_THROW(FunctionPattern(body, Identity()), BuilderError);
}

TEST(FunctionPattern, MatchesSiblingBranchesInEitherOrder) {
  const auto body =
      Function({"x"}, {"a", "b"}, {MakeNode("Neg", {"x"}, {"a"}), MakeNode("Abs", {"x"}, {"b"})});
  FunctionPattern pattern(body, body);
  for (const bool reverse : {false, true}) {
    GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    AddInput(builder, "input");
    if (reverse) {
      builder.MakeNode("Abs", {"input"}, {"positive"});
      builder.MakeNode("Neg", {"input"}, {"negative"});
    } else {
      builder.MakeNode("Neg", {"input"}, {"negative"});
      builder.MakeNode("Abs", {"input"}, {"positive"});
    }
    builder.MakeOutput("negative");
    builder.MakeOutput("positive");
    GraphGraph graph(builder);
    const auto match = pattern.Match(graph, *graph.NodeBefore("positive"));
    ASSERT_EQ(match.pattern, &pattern);
    EXPECT_EQ(match.insert_at, &builder.Nodes()[1]);
    const auto result = pattern.Apply(graph, match.nodes);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].input(0), "input");
    EXPECT_EQ(result[1].input(0), "input");
  }
}

TEST(FunctionPattern, RejectsDisconnectedTemplatesIncludingEmptyOptionalSlots) {
  const auto disconnected = Function(
      {"x", "z"}, {"a", "b"}, {MakeNode("Neg", {"x"}, {"a"}), MakeNode("Abs", {"z"}, {"b"})});
  EXPECT_THROW(FunctionPattern(disconnected, disconnected), BuilderError);
  const auto optional =
      Function({"x", "z"}, {"a", "b"},
               {MakeNode("Clip", {"x", ""}, {"a"}), MakeNode("Clip", {"z", ""}, {"b"})});
  EXPECT_THROW(FunctionPattern(optional, optional), BuilderError);
}

TEST(FunctionPattern, MatchesForwardBranchesAndBacktracksLocally) {
  const auto body =
      Function({"x"}, {"left_output", "right_output"},
               {MakeNode("Neg", {"x"}, {"left"}), MakeNode("Abs", {"left"}, {"left_output"}),
                MakeNode("Neg", {"x"}, {"right"}), MakeNode("Relu", {"right"}, {"right_output"})});
  FunctionPattern pattern(body, body);
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  AddInput(builder, "input");
  AddInput(builder, "unrelated");
  for (int i = 0; i < 128; ++i)
    builder.MakeNode("Abs", {"unrelated"}, {"noise_" + std::to_string(i)});
  builder.MakeNode("Neg", {"input"}, {"dead_end"});
  builder.MakeNode("Neg", {"input"}, {"left"});
  builder.MakeNode("Abs", {"left"}, {"left_output"});
  builder.MakeNode("Neg", {"input"}, {"right"});
  builder.MakeNode("Relu", {"right"}, {"right_output"});
  builder.MakeOutput("left_output");
  builder.MakeOutput("right_output");
  GraphGraph graph(builder);
  const auto match = pattern.Match(graph, *graph.NodeBefore("right_output"));
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 4u);
  EXPECT_EQ(match.nodes[0], graph.NodeBefore("left"));
  EXPECT_EQ(match.nodes[1], graph.NodeBefore("left_output"));
  EXPECT_EQ(match.nodes[2], graph.NodeBefore("right"));
  EXPECT_EQ(match.nodes[3], graph.NodeBefore("right_output"));
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 4u);
  EXPECT_EQ(replacement[1].output(0), "left_output");
  EXPECT_EQ(replacement[3].output(0), "right_output");
}

TEST(FunctionPattern, RejectsConfiguredAnchorsAndPredecessors) {
  FunctionPattern pattern(DoubleNeg(), Identity());
  for (const int configured : {0, 1}) {
    GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    AddInput(builder, "input");
    for (int index = 0; index < 2; ++index) {
      auto node =
          MakeNode("Neg", {index == 0 ? "input" : "middle"}, {index == 0 ? "middle" : "output"});
      if (index == configured) {
        auto *configuration = node.add_device_configurations();
        configuration->set_configuration_id("sharded");
        configuration->set_pipeline_stage(2);
        configuration->add_sharding_spec()->set_tensor_name(index == 0 ? "middle" : "output");
      }
      builder.ImportNode(node);
    }
    GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {&builder.Nodes()[0], &builder.Nodes()[1]}), BuilderError);
  }
}

TEST(FunctionPattern, RejectsInconsistentInputBindingsAndInterleavedConsumers) {
  auto body = Function({"x"}, {"y"}, {MakeNode("Add", {"x", "x"}, {"y"})});
  FunctionPattern pattern(body, Identity());
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  AddInput(builder, "a");
  AddInput(builder, "b");
  builder.MakeNode("Add", {"a", "b"}, {"different"});
  builder.MakeNode("Add", {"a", "a"}, {"same"});
  GraphGraph graph(builder);
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, &pattern);

  body = DoubleNeg();
  body.add_output("middle");
  auto replacement = Function({"x"}, {"x", "n"}, {MakeNode("Neg", {"x"}, {"n"})});
  FunctionPattern multi(body, replacement);
  GraphBuilder interleaved("g", SchemaLookup(), {{"", 18}});
  AddInput(interleaved, "x");
  interleaved.MakeNode("Neg", {"x"}, {"first"});
  interleaved.MakeNode("Abs", {"first"}, {"external"});
  interleaved.MakeNode("Neg", {"first"}, {"last"});
  GraphGraph indexed(interleaved);
  EXPECT_EQ(multi.Match(indexed, interleaved.Nodes()[2]).pattern, nullptr);
}

TEST(FunctionPattern, RejectsDifferentOpsetDomainArityAndLiterals) {
  auto body = Function({"x"}, {"y"}, {Transpose("x", "y", false)});
  FunctionPattern pattern(body, Identity());
  for (const int version : {18, 19}) {
    GraphBuilder builder("g", SchemaLookup(), {{"", version}, {"custom", 1}});
    AddInput(builder, "input");
    builder.ImportNode(Transpose("input", "output", false));
    builder.MakeNode("Transpose", {"input"}, {"default"});
    builder.MakeNode("Transpose", {"input"}, {"custom"}, "custom");
    GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern != nullptr, version == 18);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
  }
  auto literal = Function({"x"}, {"y"}, {Transpose("x", "y", false, {0, 1})});
  FunctionPattern literal_pattern(literal, Identity());
  GraphBuilder transposed("g", SchemaLookup(), {{"", 18}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(2));
  transposed.MakeInput("a", core::symbolic::TensorType::kFloat, shape);
  transposed.ImportNode(Transpose("a", "b", false, {1, 0}));
  GraphGraph indexed(transposed);
  EXPECT_EQ(literal_pattern.Match(indexed, transposed.Nodes()[0]).pattern, nullptr);

  auto add = Function({"x", "y"}, {"z"}, {MakeNode("Custom", {"x", "y"}, {"z"}, "custom")});
  auto &opset = *add.add_opset_import();
  opset.set_domain("custom");
  opset.set_version(1);
  FunctionPattern add_pattern(add, add);
  GraphBuilder builder("g", nullptr, {{"", 18}, {"custom", 1}});
  AddInput(builder, "a");
  AddInput(builder, "b");
  builder.MakeNode("Custom", {"a", "b", "a"}, {"output"}, "custom");
  builder.MakeNode("Custom", {"a", "b"}, {"first", "second"}, "custom");
  builder.MakeNode("Custom", {"a", ""}, {"optional"}, "custom");
  GraphGraph graph(builder);
  EXPECT_EQ(add_pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  EXPECT_EQ(add_pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
  EXPECT_EQ(add_pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);

  auto optional = Function({"x"}, {"z"}, {MakeNode("Custom", {"x", ""}, {"z"}, "custom")});
  optional.add_opset_import(opset);
  FunctionPattern optional_pattern(optional, optional);
  EXPECT_EQ(optional_pattern.Match(graph, builder.Nodes()[2]).pattern, &optional_pattern);
}

} // namespace
} // namespace Test
