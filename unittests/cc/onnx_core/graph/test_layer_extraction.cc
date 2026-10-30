// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/graph/layer_extraction.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"

#include <algorithm>
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
  ASSERT_EQ(proposal.status, core::graph::RepeatedLayerStatus::kFound);
  ASSERT_EQ(proposal.nodes.size(), 4u);
  EXPECT_EQ(proposal.insert_at, &builder.Nodes()[0]);
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
  EXPECT_TRUE(builder.LocalFunctions().empty());
  EXPECT_LE(builder.OpsetVersion("ai.rt"), 0);

  const auto replacement = core::graph::MakeRepeatedLayerReplacements(graph, proposal.nodes);
  ASSERT_EQ(replacement.size(), 2u);
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
  auto proposal = core::graph::ProposeRepeatedLayers(graph);
  ASSERT_EQ(proposal.status, core::graph::RepeatedLayerStatus::kFound);
  const GraphProto before = builder.BuildGraph();

  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {}), core::builder::BuilderError);
  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {nullptr}),
               core::builder::BuilderError);
  proposal.nodes.pop_back();
  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, proposal.nodes),
               core::builder::BuilderError);
  proposal.nodes.push_back(&builder.Nodes()[3]);
  std::reverse(proposal.nodes.begin(), proposal.nodes.end());
  EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, proposal.nodes),
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
    EXPECT_EQ(proposal.status, core::graph::RepeatedLayerStatus::kNoMatch);
    EXPECT_TRUE(proposal.nodes.empty());
    EXPECT_EQ(proposal.insert_at, nullptr);
  }
  {
    core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}, {"ai.rt", 2}});
    AddRepeatedLayers(builder);
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto proposal = core::graph::ProposeRepeatedLayers(graph);
    EXPECT_EQ(proposal.status, core::graph::RepeatedLayerStatus::kIncompatibleOpset);
    EXPECT_TRUE(proposal.nodes.empty());
    EXPECT_EQ(proposal.insert_at, nullptr);
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
    EXPECT_EQ(proposal.status, core::graph::RepeatedLayerStatus::kInsideLayer);
    EXPECT_TRUE(proposal.nodes.empty());
    EXPECT_EQ(proposal.insert_at, nullptr);
    EXPECT_THROW(core::graph::MakeRepeatedLayerReplacements(graph, {&builder.Nodes()[0]}),
                 core::builder::BuilderError);
    EXPECT_TRUE(builder.LocalFunctions().empty());
    EXPECT_EQ(model.LocalFunctions().size(), 1u);
  }
}

} // namespace Test
