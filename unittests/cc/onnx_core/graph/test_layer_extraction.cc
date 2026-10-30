// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/graph/graph_partition.h"
#include "onnx_core/graph/layer_extraction.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn ExtractionSchemaLookup() {
  onnx_shapes::RegisterShapeFunctions();
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

void AddRepeatedLayers(core::builder::GraphBuilder &builder) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(2));
  for (const std::string name : {"x", "w0", "w1"})
    builder.MakeInput(name, core::symbolic::TensorType::kFloat, shape);
  builder.MakeNode("MatMul", {"x", "w0"}, {"mm0"});
  builder.MakeNode("Relu", {"mm0"}, {"h0"});
  builder.MakeNode("MatMul", {"h0", "w1"}, {"mm1"});
  builder.MakeNode("Relu", {"mm1"}, {"h1"});
  builder.MakeOutput("h1");
}

} // namespace

TEST(LayerExtraction, ProposesWithoutMutationAndBuildsSharedFunctionCalls) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
  AddRepeatedLayers(builder);
  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const GraphProto before = builder.BuildGraph();

  const auto proposal = core::graph::ProposeRepeatedLayers(graph);
  ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
  const auto nodes = proposal.MatchedNodes(builder.Nodes());
  ASSERT_EQ(nodes.size(), 4u);
  EXPECT_EQ(nodes.front(), &builder.Nodes()[0]);
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
  EXPECT_TRUE(builder.LocalFunctions().empty());
  EXPECT_LE(builder.OpsetVersion("ai.rt"), 0);

  const auto replacement = core::graph::MakeRepeatedLayerReplacements(graph, nodes);
  ASSERT_EQ(replacement.size(), 2u);
  EXPECT_EQ(replacement[0].name(), "RepeatedLayer_n0_p0");
  EXPECT_EQ(replacement[1].name(), "RepeatedLayer_n1_p2");
  EXPECT_EQ(replacement[0].domain(), "ai.rt");
  EXPECT_EQ(replacement[1].domain(), "ai.rt");
  EXPECT_EQ(replacement[0].op_type(), replacement[1].op_type());
  ASSERT_EQ(replacement[0].input().size(), 2u);
  EXPECT_EQ(replacement[0].input(0), "x");
  EXPECT_EQ(replacement[0].input(1), "w0");
  EXPECT_EQ(replacement[0].output(0), "h0");
  EXPECT_EQ(replacement[1].input(0), "h0");
  EXPECT_EQ(replacement[1].input(1), "w1");
  EXPECT_EQ(replacement[1].output(0), "h1");
  ASSERT_EQ(builder.LocalFunctions().size(), 1u);
  const GraphProto body = builder.LocalFunctions()[0]->BuildGraph();
  ASSERT_EQ(body.node().size(), 2u);
  EXPECT_EQ(body.node(0).op_type(), "MatMul");
  EXPECT_EQ(body.node(1).op_type(), "Relu");
  EXPECT_EQ(builder.OpsetVersion("ai.rt"), 1);
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
}

TEST(LayerExtraction, RejectsInvalidIntervalsBeforeCreatingFunctions) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
  AddRepeatedLayers(builder);
  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const auto proposal = core::graph::ProposeRepeatedLayers(graph);
  ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
  auto nodes = proposal.MatchedNodes(builder.Nodes());
  const GraphProto before = builder.BuildGraph();

  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {}), core::builder::BuilderError);
  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {nullptr}),
               core::builder::BuilderError);
  nodes.pop_back();
  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, nodes),
               core::builder::BuilderError);
  nodes.push_back(&builder.Nodes()[3]);
  std::reverse(nodes.begin(), nodes.end());
  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, nodes),
               core::builder::BuilderError);
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
  EXPECT_TRUE(builder.LocalFunctions().empty());
  EXPECT_LE(builder.OpsetVersion("ai.rt"), 0);
}

TEST(LayerExtraction, DeclinesEmptyGraphsIncompatibleOpsetsAndExistingLayerBodies) {
  {
    core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto proposal = core::graph::ProposeRepeatedLayers(graph);
    EXPECT_EQ(proposal.status, core::graph::OperatorGroupStatus::kNoMatch);
    EXPECT_TRUE(proposal.groups.empty());
    EXPECT_TRUE(proposal.MatchedNodes(builder.Nodes()).empty());
  }
  {
    core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}, {"ai.rt", 2}});
    AddRepeatedLayers(builder);
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto proposal = core::graph::ProposeRepeatedLayers(graph);
    EXPECT_EQ(proposal.status, core::graph::OperatorGroupStatus::kIncompatibleOpset);
    EXPECT_TRUE(proposal.groups.empty());
    EXPECT_TRUE(proposal.MatchedNodes(builder.Nodes()).empty());
    EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {&builder.Nodes()[0]}),
                 core::builder::BuilderError);
    EXPECT_TRUE(builder.LocalFunctions().empty());
    EXPECT_EQ(builder.OpsetVersion("ai.rt"), 2);
  }
  {
    core::builder::GraphBuilder model("g", ExtractionSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
    auto &builder = model.MakeLocalFunction("Layer", "ai.rt");
    AddRepeatedLayers(builder);
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto proposal = core::graph::ProposeRepeatedLayers(graph);
    EXPECT_EQ(proposal.status, core::graph::OperatorGroupStatus::kInsideFunction);
    EXPECT_TRUE(proposal.groups.empty());
    EXPECT_TRUE(proposal.MatchedNodes(builder.Nodes()).empty());
    EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {&builder.Nodes()[0]}),
                 core::builder::BuilderError);
    EXPECT_TRUE(builder.LocalFunctions().empty());
    EXPECT_EQ(model.LocalFunctions().size(), 1u);
  }
}

TEST(OperatorGroups, BothAlgorithmsUseTheSameInterfacesAndReplacementConsumer) {
  for (const bool repeated : {false, true}) {
    core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
    AddRepeatedLayers(builder);
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto proposal = repeated ? core::graph::ProposeRepeatedLayers(graph)
                                   : core::graph::ProposePartitionGroups(graph, {1024, 2, 2});
    ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
    ASSERT_EQ(proposal.groups.size(), 2u);
    EXPECT_EQ(proposal.groups[0].node_indices, (std::vector<std::size_t>{0, 1}));
    EXPECT_EQ(proposal.groups[1].node_indices, (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(proposal.groups[0].inputs, (std::vector<std::string>{"x", "w0"}));
    EXPECT_EQ(proposal.groups[1].inputs, (std::vector<std::string>{"h0", "w1"}));
    EXPECT_EQ(proposal.groups[0].outputs, (std::vector<std::string>{"h0"}));
    EXPECT_EQ(proposal.groups[1].outputs, (std::vector<std::string>{"h1"}));
    const auto nodes = proposal.MatchedNodes(builder.Nodes());
    ASSERT_EQ(nodes.size(), 4u);
    EXPECT_EQ(nodes.front(), &builder.Nodes()[proposal.first]);
    EXPECT_EQ(nodes.back(), &builder.Nodes()[proposal.last]);

    std::vector<std::size_t> emitted;
    const auto replacement =
        core::graph::MakeOperatorGroupReplacements(graph, proposal, [&](std::size_t group) {
          emitted.push_back(group);
          return MakeNode("Group", proposal.groups[group].inputs, proposal.groups[group].outputs,
                          "test.groups");
        });
    EXPECT_EQ(emitted, (std::vector<std::size_t>{0, 1}));
    ASSERT_EQ(replacement.size(), 2u);
    EXPECT_EQ(replacement[0].input(0), "x");
    EXPECT_EQ(replacement[0].output(0), "h0");
    EXPECT_EQ(replacement[1].input(0), "h0");
    EXPECT_EQ(replacement[1].output(0), "h1");
    EXPECT_TRUE(builder.LocalFunctions().empty());
  }
}

TEST(OperatorGroups, RejectsInvalidCoverageBeforeInvokingTheFunctionBuilder) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
  AddRepeatedLayers(builder);
  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const auto proposal = core::graph::ProposeRepeatedLayers(graph);
  ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
  std::vector<core::graph::OperatorGroupProposal> invalid(5, proposal);
  invalid[0].schedule.pop_back();
  invalid[1].schedule.push_back(invalid[1].schedule.front());
  invalid[2].groups[1].node_indices[0] = 0;
  invalid[3].last = builder.Nodes().size();
  invalid[4].status = core::graph::OperatorGroupStatus::kNoMatch;
  std::size_t calls = 0;
  for (const auto &candidate : invalid)
    EXPECT_THROW(core::graph::MakeOperatorGroupReplacements(graph, candidate,
                                                            [&](std::size_t) {
                                                              ++calls;
                                                              return NodeProto{};
                                                            }),
                 core::builder::BuilderError);
  EXPECT_THROW(core::graph::MakeOperatorGroupReplacements(graph, proposal, {}),
               core::builder::BuilderError);
  EXPECT_EQ(calls, 0u);
  EXPECT_TRUE(builder.LocalFunctions().empty());
}

TEST(OperatorGroups, RetainsInterleavedNodesAndRejectsOutOfRangeIntervals) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(2));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  builder.MakeInput("w0", core::symbolic::TensorType::kFloat, shape);
  builder.MakeNode("MatMul", {"x", "w0"}, {"mm0"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &value = attributes.add();
  value.set_name("value_float");
  value.set_type(AttributeProto::FLOAT);
  value.set_f(0.25f);
  builder.MakeNode("Constant", {}, {"bias"}, "", "literal_bias", attributes);
  builder.MakeNode("Relu", {"mm0"}, {"h0"});
  builder.MakeOutput("h0");
  builder.MakeOutput("bias");
  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const core::graph::OperatorGroup group{{0, 2}, {"x", "w0"}, {"h0"}, {}};
  const auto proposal = core::graph::MakeIntervalGroupProposal({group}, builder.Nodes().size());
  const auto replacement =
      core::graph::MakeOperatorGroupReplacements(graph, proposal, [&](std::size_t) {
        return MakeNode("Group", group.inputs, group.outputs, "test.groups");
      });
  ASSERT_EQ(replacement.size(), 2u);
  EXPECT_TRUE(replacement[0].Equals(builder.Nodes()[1]));
  EXPECT_EQ(replacement[1].op_type(), "Group");
  auto invalid = group;
  invalid.node_indices.back() = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(core::graph::MakeIntervalGroupProposal({invalid}, builder.Nodes().size()),
               core::builder::BuilderError);
}

} // namespace Test
