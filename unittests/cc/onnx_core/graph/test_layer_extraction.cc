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
#include <cstring>
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
  EXPECT_EQ(replacement[0].op_type(), "Layer_MatMul_0");
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

TEST(LayerExtraction, NamesAfterFirstAnchorAndAvoidsGraphNameCollisions) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(2));
  for (const std::string name : {"x", "w0", "w1", "Layer_MatMul_0"})
    builder.MakeInput(name, core::symbolic::TensorType::kFloat, shape);
  builder.MakeNode("Relu", {"x"}, {"pre0"});
  builder.MakeNode("MatMul", {"pre0", "w0"}, {"h0"});
  builder.MakeNode("Relu", {"h0"}, {"pre1"});
  builder.MakeNode("MatMul", {"pre1", "w1"}, {"h1"});
  builder.MakeOutput("h1");
  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const auto proposal = core::graph::ProposeRepeatedLayers(graph);
  ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
  const auto replacement =
      core::graph::MakeRepeatedLayerReplacements(graph, proposal.MatchedNodes(builder.Nodes()));
  ASSERT_EQ(replacement.size(), 2u);
  EXPECT_EQ(replacement[0].op_type(), "Layer_MatMul_1");
  EXPECT_EQ(replacement[1].op_type(), "Layer_MatMul_1");
  ASSERT_EQ(builder.LocalFunctions().size(), 1u);
  const auto &body = builder.LocalFunctions()[0]->Nodes();
  ASSERT_EQ(body.size(), 2u);
  EXPECT_EQ(body[0].op_type(), "Relu");
  EXPECT_EQ(body[1].op_type(), "MatMul");
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

TEST(OperatorGroups, BothAlgorithmsShareMainOperatorNamingAndCollisionChecks) {
  for (const bool repeated : {false, true}) {
    const std::string prefix = repeated ? "Layer" : "PartitionGroup";
    core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(2));
    shape.PushBack(core::symbolic::SymDim(2));
    builder.MakeInput(prefix + "_MatMul_0", core::symbolic::TensorType::kFloat, shape);
    AddRepeatedLayers(builder);
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto proposal = repeated ? core::graph::ProposeRepeatedLayers(graph)
                                   : core::graph::ProposePartitionGroups(graph, {1024, 2, 2});
    ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
    const auto nodes = proposal.MatchedNodes(builder.Nodes());
    const auto replacement =
        repeated ? core::graph::MakeRepeatedLayerReplacements(graph, nodes)
                 : core::graph::MakePartitionGroupReplacements(graph, nodes, {1024, 2, 2});
    ASSERT_EQ(replacement.size(), 2u);
    EXPECT_EQ(replacement[0].op_type(), prefix + "_MatMul_1");
    EXPECT_EQ(replacement[1].op_type(), prefix + "_MatMul_1");
    EXPECT_EQ(builder.LocalFunctions().size(), 1u);
  }
}

TEST(OperatorGroups, NamingRejectsEmptyBodiesAndDoesNotOverflowTheCounter) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  AddRepeatedLayers(builder);
  std::size_t suffix = 0;
  EXPECT_THROW(core::graph::MakeOperatorGroupFunctionName(
                   builder, utils::RepeatedProtoField<NodeProto>{}, "PartitionGroup", suffix),
               core::builder::BuilderError);
  EXPECT_THROW(core::graph::MakeOperatorGroupFunctionName(builder, builder.Nodes(), "", suffix),
               core::builder::BuilderError);
  suffix = std::numeric_limits<std::size_t>::max();
  const std::string name = "PartitionGroup_MatMul_" + std::to_string(suffix);
  EXPECT_EQ(core::graph::MakeOperatorGroupFunctionName(builder, builder.Nodes(), "PartitionGroup",
                                                       suffix),
            name);
  auto &existing = builder.MakeLocalFunction(name, "ai.rt");
  AddRepeatedLayers(existing);
  EXPECT_THROW(core::graph::MakeOperatorGroupFunctionName(builder, builder.Nodes(),
                                                          "PartitionGroup", suffix),
               core::builder::BuilderError);
}

TEST(OperatorGroups, IndexesDistinctDefinitionsAndReusesEarlyAndLateMatches) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  builder.MakeNode("Neg", {"x"}, {"middle"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &alpha = attributes.add();
  alpha.set_name("alpha");
  alpha.set_type(AttributeProto::FLOAT);
  alpha.set_f(0.1f);
  builder.MakeNode("LeakyRelu", {"middle"}, {"y"}, "", "", attributes);
  builder.MakeOutput("y");
  const auto make_body = [&](size_t index) {
    auto body = builder.BuildGraph();
    body.mutable_node(1)->mutable_attribute(0)->set_f(static_cast<float>(index + 1) / 2000);
    return body;
  };
  core::graph::OperatorGroupFunctions functions(builder, "PartitionGroup");
  size_t suffix = 0;
  for (size_t index = 0; index < 128; ++index)
    ASSERT_EQ(functions.Add(make_body(index), suffix),
              "PartitionGroup_Neg_" + std::to_string(index));
  for (const size_t index : {0u, 63u, 127u, 0u})
    EXPECT_EQ(functions.Add(make_body(index), suffix),
              "PartitionGroup_Neg_" + std::to_string(index));
  EXPECT_EQ(builder.LocalFunctions().size(), 128u);
  EXPECT_EQ(suffix, 128u);
  EXPECT_EQ(functions.Stats().candidate_normalizations, 128u);
  EXPECT_EQ(functions.Stats().exact_comparisons, 4u);
}

TEST(OperatorGroups, FingerprintCollisionsStillRequireExactBodyEquality) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  AddRepeatedLayers(builder);
  core::graph::OperatorGroupFunctions functions(builder, "Layer");
  size_t suffix = 0;
  const GraphProto original = builder.BuildGraph();
  const std::string name = functions.Add(original, suffix);
  ASSERT_EQ(functions.Find(original), name);
  auto changed = builder.BuildGraph();
  changed.mutable_node(0)->set_domain("ai.onnx");
  const auto comparisons = functions.Stats().exact_comparisons;
  // NodeProto::Signature normalizes the domain, but exact body equality does not.
  EXPECT_TRUE(functions.Find(changed).empty());
  EXPECT_EQ(functions.Stats().exact_comparisons, comparisons + 1);
  EXPECT_EQ(functions.Find(original), name);
}

TEST(OperatorGroups, InlinesOnlySmallMaterializedConstantsCommonToAllFunctionInstances) {
  for (const bool repeated : {false, true})
    for (const int width : {31, 32})
      for (const int variant : {0, 1, 2, 3}) {
        SCOPED_TRACE(::testing::Message() << repeated << "," << width << "," << variant);
        core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
        const auto shape = [](int64_t rows, int64_t columns) {
          core::symbolic::SymShape result;
          result.PushBack(core::symbolic::SymDim(rows));
          result.PushBack(core::symbolic::SymDim(columns));
          return result;
        };
        builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape(1, width));
        std::string input = "x";
        for (int i = 0; i < 3; ++i) {
          const std::string suffix = std::to_string(i);
          builder.MakeInput("w" + suffix, core::symbolic::TensorType::kFloat, shape(width, width));
          std::vector<float> values(width, variant == 1 && i == 2 ? 2.0f : 1.0f);
          TensorProto bias = MakeInitializer<float>(("bias" + suffix).c_str(), {1, width}, values);
          if (variant == 2)
            builder.MakeInput("bias" + suffix, core::symbolic::TensorType::kFloat, shape(1, width));
          if (variant == 3) {
            bias.clear_float_data();
            bias.clear_raw_data();
            bias.set_data_location(TensorProto::EXTERNAL);
            auto &location = *bias.add_external_data();
            location.set_key("location");
            location.set_value("unavailable-small-constant.bin");
          } else if (i != 0) {
            bias.clear_float_data();
            bias.set_raw_data(values.data(), values.size() * sizeof(float));
          }
          builder.MakeInitializer(bias);
          builder.MakeNode("MatMul", {input, "w" + suffix}, {"mm" + suffix});
          builder.MakeNode("Add", {"mm" + suffix, "bias" + suffix}, {"h" + suffix});
          input = "h" + suffix;
        }
        builder.MakeOutput(input);
        core::builder::GraphGraph graph(
            builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
        const auto proposal = repeated ? core::graph::ProposeRepeatedLayers(graph)
                                       : core::graph::ProposePartitionGroups(graph, {65536, 2, 2});
        ASSERT_EQ(proposal.status, core::graph::OperatorGroupStatus::kFound);
        const auto nodes = proposal.MatchedNodes(builder.Nodes());
        const auto replacement =
            repeated ? core::graph::MakeRepeatedLayerReplacements(graph, nodes)
                     : core::graph::MakePartitionGroupReplacements(graph, nodes, {65536, 2, 2});
        ASSERT_EQ(replacement.size(), 3u);
        const bool inlined =
            width < 32 && variant != 2 && variant != 3 && (!repeated || variant != 1);
        for (std::size_t i = 0; i < replacement.size(); ++i) {
          const auto &call = replacement[i];
          EXPECT_EQ(call.input().size(), inlined ? 2u : 3u);
          const auto &body = builder.LocalFunction(call.op_type()).Nodes();
          ASSERT_EQ(body.size(), inlined ? 3u : 2u);
          EXPECT_EQ(body[0].op_type(), inlined ? "Constant" : "MatMul");
          if (inlined) {
            const auto &tensor = body[0].attribute(0).t();
            EXPECT_EQ(tensor.dims().values(), (std::vector<int64_t>{1, width}));
            EXPECT_FALSE(tensor.HasDeferredWeight());
            EXPECT_TRUE(tensor.external_data().empty());
            EXPECT_EQ(tensor.raw_data().size(), static_cast<std::size_t>(width) * sizeof(float));
            const std::vector<float> expected(width, variant == 1 && i == 2 ? 2.0f : 1.0f);
            EXPECT_EQ(std::memcmp(tensor.raw_data().data(), expected.data(),
                                  expected.size() * sizeof(float)),
                      0);
          }
        }
        if (repeated) {
          EXPECT_EQ(replacement[0].op_type(), replacement[1].op_type());
          EXPECT_EQ(replacement[0].op_type(), replacement[2].op_type());
        }
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

TEST(OperatorGroups, InternalizesOnlyShapesOfCorrespondingBoundaryInputs) {
  for (const std::string variant : {"common", "attributes", "source_port", "missing", "defaults"}) {
    SCOPED_TRACE(variant);
    core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
    core::symbolic::SymShape shape;
    for (const int64_t dim : {1, 2, 2})
      shape.PushBack(core::symbolic::SymDim(dim));
    std::vector<core::graph::OperatorGroup> groups;
    for (int i = 0; i < 3; ++i) {
      const auto suffix = std::to_string(i);
      for (const auto &prefix : {"x", "w", "z"})
        builder.MakeInput(prefix + suffix, core::symbolic::TensorType::kFloat, shape);
      utils::RepeatedProtoField<AttributeProto> attributes;
      const auto add_attribute = [&](const char *name, int64_t value) {
        auto &attribute = attributes.add();
        attribute.set_name(name);
        attribute.set_type(AttributeProto::INT);
        attribute.set_i(value);
      };
      if (variant != "defaults") {
        add_attribute("end", variant == "attributes" && i == 2 ? 4 : 3);
        add_attribute("start", 1);
      } else if (i == 2) {
        add_attribute("end", std::numeric_limits<int64_t>::max());
        add_attribute("start", 0);
      }
      const std::string source = variant == "missing"                 ? "z"
                                 : variant == "source_port" && i == 2 ? "w"
                                                                      : "x";
      builder.MakeNode("Shape", {source + suffix}, {"s" + suffix}, "", "", attributes);
      const auto first = builder.Nodes().size();
      builder.MakeNode("MatMul", {"x" + suffix, "w" + suffix}, {"mm" + suffix});
      builder.MakeNode("Reshape", {"mm" + suffix, "s" + suffix}, {"y" + suffix});
      builder.MakeOutput("y" + suffix);
      groups.push_back(
          {{first, first + 1}, {"x" + suffix, "w" + suffix, "s" + suffix}, {"y" + suffix}, {}});
    }
    core::builder::GraphGraph graph(
        builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
    const auto producers = core::graph::FindCommonOperatorGroupInputs(graph, groups);
    const bool internalized = variant == "common" || variant == "defaults";
    ASSERT_EQ(producers.size(), internalized ? 1u : 0u);
    for (const auto &group : groups) {
      GraphProto body;
      for (const auto &name : group.inputs)
        body.add_input()->set_name(name);
      for (const auto index : group.node_indices)
        *body.add_node() = utils::CopyWithSharedTensorStorage(builder.Nodes()[index]);
      const auto inputs = core::graph::InlineOperatorGroupInputs(body, group, producers);
      EXPECT_EQ(inputs.size(), internalized ? 2u : 3u);
      ASSERT_EQ(body.node().size(), internalized ? 3u : 2u);
      if (internalized) {
        const auto &node = body.node(0);
        EXPECT_EQ(node.op_type(), "Shape");
        EXPECT_EQ(node.input(0), group.inputs[0]);
        EXPECT_EQ(node.output(0), group.inputs[2]);
        ASSERT_EQ(node.attribute().size(), variant == "defaults" ? 0u : 2u);
        if (variant != "defaults") {
          EXPECT_EQ(node.attribute(0).name(), "start");
          EXPECT_EQ(node.attribute(0).i(), 1);
          EXPECT_EQ(node.attribute(1).name(), "end");
          EXPECT_EQ(node.attribute(1).i(), 3);
        }
      }
    }
  }
}

TEST(OperatorGroups, OrdersInternalizedShapesAfterTheirInternalizedConstantInputs) {
  core::builder::GraphBuilder builder("g", ExtractionSchemaLookup(), {{"", 28}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(2));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 2}, {1, 0, 0, 1}));
  builder.MakeNode("Shape", {"w"}, {"s"});
  builder.MakeNode("MatMul", {"x", "w"}, {"mm"});
  builder.MakeNode("Reshape", {"mm", "s"}, {"y"});
  builder.MakeOutput("y");
  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const core::graph::OperatorGroup group{{1, 2}, {"s", "w", "x"}, {"y"}, {}};
  const auto producers = core::graph::FindCommonOperatorGroupInputs(graph, {&group, 1});
  ASSERT_EQ(producers.size(), 2u);
  GraphProto body;
  for (const auto &name : group.inputs)
    body.add_input()->set_name(name);
  for (const auto index : group.node_indices)
    *body.add_node() = utils::CopyWithSharedTensorStorage(builder.Nodes()[index]);
  const auto inputs = core::graph::InlineOperatorGroupInputs(body, group, producers);
  EXPECT_EQ(inputs, (std::vector<std::string>{"x"}));
  ASSERT_EQ(body.node().size(), 4u);
  EXPECT_EQ(body.node(0).op_type(), "Constant");
  EXPECT_EQ(body.node(0).output(0), "w");
  EXPECT_EQ(body.node(1).op_type(), "Shape");
  EXPECT_EQ(body.node(1).input(0), "w");
  EXPECT_EQ(body.node(1).output(0), "s");
  EXPECT_EQ(body.node(2).op_type(), "MatMul");
  EXPECT_EQ(body.node(3).op_type(), "Reshape");
  const std::vector<core::graph::OperatorGroupInput> cycle{{0, 1, MakeNode("Shape", {""}, {""})},
                                                           {1, 0, MakeNode("Shape", {""}, {""})}};
  GraphProto invalid;
  invalid.add_input()->set_name("s");
  invalid.add_input()->set_name("w");
  const auto invalid_before = utils::CopyWithSharedTensorStorage(invalid);
  const core::graph::OperatorGroup invalid_group{{}, {"s", "w"}, {}, {}};
  EXPECT_THROW(core::graph::InlineOperatorGroupInputs(invalid, invalid_group, cycle),
               core::builder::BuilderError);
  EXPECT_TRUE(invalid.Equals(invalid_before));
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
