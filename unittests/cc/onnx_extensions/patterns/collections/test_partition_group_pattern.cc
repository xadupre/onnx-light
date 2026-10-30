// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/collections/partition_group_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

using core::builder::BuilderError;
using core::builder::GraphBuilder;
using core::builder::GraphGraph;
using onnx_patterns::PartitionGroupPattern;

GraphBuilder::SchemaLookupFn SchemaLookup() {
  onnx_shapes::RegisterShapeFunctions();
  return
      [](const std::string &name) { return onnx_op::GetAllOnnxOpSchemasWithHistory(name, false); };
}

core::symbolic::SymShape Shape(std::initializer_list<int64_t> dimensions = {}) {
  core::symbolic::SymShape shape;
  for (int64_t dimension : dimensions)
    shape.PushBack(core::symbolic::SymDim(dimension));
  return shape;
}

GraphBuilder Chain(size_t count = 2, int64_t width = 1) {
  GraphBuilder builder("chain", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({width}));
  std::string input = "x";
  for (size_t i = 0; i < count; ++i) {
    const std::string output = i + 1 == count ? "y" : "v" + std::to_string(i);
    builder.MakeNode(i % 2 == 0 ? "Neg" : "Abs", {input}, {output});
    input = output;
  }
  builder.MakeOutput("y");
  return builder;
}

RewritingProto Optimize(GraphBuilder &builder, uint64_t bytes = 1024, uint64_t maximum = 128) {
  std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns{
      std::make_shared<PartitionGroupPattern>(0, bytes, 2, maximum)};
  GraphGraph graph(builder, patterns);
  return graph.Optimize();
}

TEST(PartitionGroupPattern, CallsCoreSelectionAndPreservesShapeAndNames) {
  auto builder = Chain(4);
  EXPECT_FALSE(Optimize(builder, 8, 2).rewrites().empty());
  ASSERT_EQ(builder.Nodes().size(), 2u);
  ASSERT_EQ(builder.LocalFunctions().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].domain(), "ai.rt");
  EXPECT_EQ(builder.Nodes()[0].op_type(), "PartitionGroup_Neg_0");
  EXPECT_EQ(builder.Nodes()[1].op_type(), "PartitionGroup_Neg_1");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[0].output(0), "v1");
  EXPECT_EQ(builder.Nodes()[1].input(0), "v1");
  EXPECT_EQ(builder.Nodes()[1].output(0), "y");
  EXPECT_TRUE(builder.HasShape("y"));
  VerifyModel(builder.ToModel());
  EXPECT_TRUE(Optimize(builder, 8, 2).rewrites().empty());
  EXPECT_EQ(builder.LocalFunctions().size(), 2u);
}

TEST(PartitionGroupPattern, RunsOnlyAtFirstNodeAndRevalidatesApply) {
  auto builder = Chain();
  GraphGraph graph(builder);
  PartitionGroupPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_THROW(pattern.Apply(graph, {}), BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {nullptr}), BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {match.nodes[1], match.nodes[0]}), BuilderError);
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type(), "PartitionGroup_Neg_0");
}

TEST(PartitionGroupPattern, LimitsAndDefaultSelectionArePreserved) {
  onnx_patterns::RegisterPatterns();
  const auto names = core::builder::RegisteredPatternNames(false);
  EXPECT_EQ(std::find(names.begin(), names.end(), "PartitionGroup"), names.end());
  EXPECT_EQ(onnx_patterns::CreatePattern("PartitionGroup")->Name(), "PartitionGroup");
  EXPECT_THROW(PartitionGroupPattern(0, 1024, 1, 128), std::invalid_argument);
  EXPECT_THROW(PartitionGroupPattern(0, 1024, 3, 2), std::invalid_argument);
  auto builder = Chain();
  EXPECT_TRUE(Optimize(builder, 7).rewrites().empty());
  EXPECT_TRUE(builder.LocalFunctions().empty());
  EXPECT_FALSE(Optimize(builder, 8).rewrites().empty());
  auto large = Chain(4);
  EXPECT_FALSE(Optimize(large, 8, 4).rewrites().empty());
  ASSERT_EQ(large.Nodes().size(), 1u);
  EXPECT_EQ(large.LocalFunctions()[0]->Nodes().size(), 4u);
}

TEST(PartitionGroupPattern, KeepsFanoutAndIntermediateGraphOutputs) {
  auto builder = Chain(3);
  builder.MakeOutput("v0");
  EXPECT_FALSE(Optimize(builder, 12).rewrites().empty());
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].output_size(), 2);
  EXPECT_EQ(builder.Nodes()[0].output(0), "v0");
  EXPECT_EQ(builder.Nodes()[0].output(1), "y");
  EXPECT_EQ(builder.LocalFunctions()[0]->Outputs().size(), 2u);
  VerifyModel(builder.ToModel());
}

TEST(PartitionGroupPattern, KeepsInterleavedConstantsBeforeCallsAndMetadataInBodies) {
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1}));
  builder.MakeNode("Neg", {"x"}, {"middle"}, "", "negative");
  NodeProto constant = MakeNode("Constant", {}, {"weight"});
  auto &value = *constant.add_attribute();
  value.set_name("value");
  value.set_type(AttributeProto::TENSOR);
  value.mutable_t()->set_data_type(TensorProto::FLOAT);
  value.mutable_t()->add_dims(1);
  value.mutable_t()->add_float_data(2);
  builder.ImportNode(constant);
  builder.MakeNode("Add", {"middle", "weight"}, {"y"});
  builder.MakeOutput("y");
  GraphGraph graph(builder);
  PartitionGroupPattern pattern(0, 8);
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 2u);
  EXPECT_EQ(replacement[0].op_type(), "Constant");
  EXPECT_EQ(replacement[1].input_size(), 1);
  EXPECT_EQ(replacement[1].input(0), "x");
  const auto function = builder.LocalFunctions()[0]->ToFunction("ai.rt");
  ASSERT_EQ(function.node_size(), 3);
  EXPECT_EQ(function.node(0).op_type(), "Constant");
  EXPECT_EQ(function.node(1).name(), "negative");
}

TEST(PartitionGroupPattern, DoesNotRewriteFunctionBodiesOrConflictingDomainVersions) {
  GraphBuilder model("model", SchemaLookup(), {{"", 18}});
  auto &body = model.MakeLocalFunction("Body", "local");
  body.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1}));
  body.MakeNode("Neg", {"x"}, {"middle"});
  body.MakeNode("Abs", {"middle"}, {"y"});
  body.MakeOutput("y");
  GraphGraph graph(body);
  EXPECT_EQ(PartitionGroupPattern().Match(graph, body.Nodes()[0]).pattern, nullptr);
  auto conflict = Chain();
  conflict.SetOpsetVersion("ai.rt", 2);
  GraphGraph conflicting_graph(conflict);
  EXPECT_EQ(PartitionGroupPattern().Match(conflicting_graph, conflict.Nodes()[0]).pattern, nullptr);
}

TEST(PartitionGroupPattern, KeepsInferredShapeTagsOnFunctionInputsAndValues) {
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("index", {}, {0}));
  builder.MakeInitializer(MakeInitializer<int64_t>("one", {}, {1}));
  builder.MakeNode("Shape", {"x"}, {"shape"});
  builder.MakeNode("Gather", {"shape", "index"}, {"dimension"});
  builder.MakeNode("Add", {"dimension", "one"}, {"y"});
  builder.MakeOutput("y");
  const auto tags = builder.Compute().ValueTags();
  ASSERT_EQ(tags.at("shape"), "shape");
  ASSERT_FALSE(Optimize(builder, 24, 2).rewrites().empty());
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Shape");
  ASSERT_EQ(builder.LocalFunctions().size(), 1u);
  const auto &body = *builder.LocalFunctions()[0];
  for (const auto *name : {"shape", "dimension", "y"})
    EXPECT_EQ(body.Compute().ValueTags().at(name), tags.at(name));
  EXPECT_EQ(builder.Compute().ValueTags().at("y"), tags.at("y"));
  VerifyModel(builder.ToModel());
}

TEST(PartitionGroupPattern, PartitionsEligibleNeighborsOfDeviceConfiguredNodesAndExistingGroups) {
  for (const bool existing_group : {false, true}) {
    auto builder = Chain(3);
    if (existing_group) {
      builder.SetOpsetVersion("ai.rt", 1);
      auto &function = builder.MakeLocalFunction("PartitionGroup_Neg_existing", "ai.rt");
      function.MakeInput("input", core::symbolic::TensorType::kFloat, Shape({1}));
      function.MakeNode("Neg", {"input"}, {"output"});
      function.MakeOutput("output");
      auto model = builder.ToModel();
      auto *first = model.mutable_graph()->mutable_node(0);
      first->set_domain("ai.rt");
      first->set_op_type("PartitionGroup_Neg_existing");
      builder = GraphBuilder(model, SchemaLookup());
    } else {
      auto model = builder.ToModel();
      model.mutable_graph()->mutable_node(0)->add_device_configurations()->set_configuration_id(
          "device");
      builder = GraphBuilder(model, SchemaLookup());
    }
    EXPECT_FALSE(Optimize(builder, 8).rewrites().empty());
    ASSERT_EQ(builder.Nodes().size(), 2u);
    EXPECT_EQ(builder.Nodes()[1].domain(), "ai.rt");
    EXPECT_EQ(builder.Nodes()[1].input(0), "v0");
    EXPECT_EQ(builder.Nodes()[1].output(0), "y");
    EXPECT_EQ(builder.Nodes()[0].device_configurations().empty(), existing_group);
    VerifyModel(builder.ToModel());
  }
}

TEST(PartitionGroupPattern, RequiresInferenceOrCompleteCustomOutputAnnotations) {
  for (const bool annotated : {false, true}) {
    GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"custom", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1}));
    builder.MakeNode("Foo", {"x"}, {"t"}, "custom");
    if (annotated)
      builder.MakeOutput("t", core::symbolic::TensorType::kFloat, Shape({1}));
    builder.MakeNode("Relu", {"t"}, {"y"});
    builder.MakeOutput("y", core::symbolic::TensorType::kFloat, Shape({1}));
    GraphGraph graph(builder);
    PartitionGroupPattern pattern;
    const auto match = pattern.Match(graph, builder.Nodes()[0]);
    if (annotated) {
      ASSERT_EQ(match.pattern, &pattern);
      EXPECT_NO_THROW(pattern.Apply(graph, match.nodes));
      ASSERT_EQ(builder.LocalFunctions().size(), 1u);
      EXPECT_EQ(builder.LocalFunctions()[0]->Nodes()[0].op_type(), "Foo");
    } else {
      EXPECT_EQ(match.pattern, nullptr);
      EXPECT_TRUE(builder.LocalFunctions().empty());
    }
  }
}

TEST(PartitionGroupPattern, PreservesCapturedConstantAndOverridableInitializerClassification) {
  for (const bool overridable : {false, true})
    for (const bool capped : {false, true}) {
      SCOPED_TRACE(::testing::Message() << overridable << "," << capped);
      GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
      builder.MakeInput("index", core::symbolic::TensorType::kInt64, Shape());
      builder.MakeInput("condition", core::symbolic::TensorType::kBool, Shape());
      if (overridable)
        builder.MakeInput("weight", core::symbolic::TensorType::kFloat, Shape({64}));
      TensorProto weight;
      weight.set_name("weight");
      weight.set_data_type(TensorProto::FLOAT);
      weight.add_dims(64);
      weight.set_data_location(TensorProto::EXTERNAL);
      auto &location = *weight.add_external_data();
      location.set_key("location");
      location.set_value("unavailable-captured-weight.bin");
      builder.MakeInitializer(weight);
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const auto *name : {"then_branch", "else_branch"}) {
        auto &branch = builder.MakeSubgraph(name);
        branch.MakeNode("Gather", {"weight", "index"}, {"element"});
        branch.MakeNode("Neg", {"element"}, {"result"});
        branch.MakeOutput("result");
        auto &reference = attributes.add();
        reference.set_name(std::string(name) + "_ref");
        reference.set_type(AttributeProto::STRING);
        reference.set_s(name);
      }
      builder.MakeNode("If", {"condition"}, {"y"}, "", "", attributes);
      builder.MakeOutput("y");
      std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns{
          std::make_shared<PartitionGroupPattern>(
              0, capped ? std::optional<uint64_t>{12} : std::nullopt, 2, 128,
              capped ? std::nullopt : std::optional<uint64_t>{0})};
      GraphGraph graph(builder, patterns);
      const auto plan = graph.Optimize();
      const bool grouped = capped ? !overridable : overridable;
      EXPECT_EQ(plan.rewrites().empty(), !grouped);
      for (const auto *name : {"then_branch", "else_branch"}) {
        const auto &nodes = builder.Subgraph(name).Nodes();
        ASSERT_EQ(nodes.size(), grouped ? 1u : 2u);
        if (grouped) {
          EXPECT_EQ(nodes[0].domain(), "ai.rt");
          EXPECT_EQ(nodes[0].input().size(), 2u);
          EXPECT_EQ(nodes[0].input(0), "weight");
          EXPECT_EQ(nodes[0].input(1), "index");
        }
      }
      VerifyModel(builder.ToModel());
    }
}

TEST(PartitionGroupPattern, RequiresInferenceOrCompleteAnnotationsForNestedNodes) {
  for (const bool annotated : {false, true}) {
    GraphBuilder source("g", SchemaLookup(), {{"", 18}, {"custom", 1}});
    source.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1}));
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const auto *name : {"then_branch", "else_branch"}) {
      auto &branch = source.MakeSubgraph(name);
      branch.MakeNode("Foo", {"x"}, {"t"}, "custom");
      branch.MakeNode("Relu", {"t"}, {"result"});
      branch.MakeOutput("result", core::symbolic::TensorType::kFloat, Shape({1}));
      auto &reference = attributes.add();
      reference.set_name(std::string(name) + "_ref");
      reference.set_type(AttributeProto::STRING);
      reference.set_s(name);
    }
    source.MakeNode("Container", {"x"}, {"selected"}, "custom", "", attributes);
    source.MakeOutput("selected", core::symbolic::TensorType::kFloat, Shape({1}));
    source.MakeNode("Neg", {"selected"}, {"y"});
    source.MakeOutput("y");
    auto model = source.ToModel();
    if (annotated)
      for (auto &attribute : model.mutable_graph()->mutable_node(0)->ref_attribute())
        if (attribute.has_g()) {
          auto &value = *attribute.mutable_g()->add_value_info();
          value.set_name("t");
          auto *tensor = value.mutable_type()->mutable_tensor_type();
          tensor->set_elem_type(TensorProto::FLOAT);
          tensor->mutable_shape()->add_dim()->set_dim_value(1);
        }
    GraphBuilder builder(model, SchemaLookup());
    GraphGraph graph(builder);
    PartitionGroupPattern pattern;
    const auto match = pattern.Match(graph, builder.Nodes()[0]);
    if (annotated) {
      ASSERT_EQ(match.pattern, &pattern);
      EXPECT_NO_THROW(pattern.Apply(graph, match.nodes));
      ASSERT_EQ(builder.LocalFunctions().size(), 1u);
      EXPECT_EQ(builder.LocalFunctions()[0]->Nodes()[0].op_type(), "Container");
    } else {
      EXPECT_EQ(match.pattern, nullptr);
      EXPECT_TRUE(builder.LocalFunctions().empty());
    }
  }
}

} // namespace
