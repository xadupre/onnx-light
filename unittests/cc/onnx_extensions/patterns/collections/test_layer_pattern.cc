// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_lib/checker.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn LayerSchemaLookup() {
  onnx_shapes::RegisterShapeFunctions();
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape LayerShape(std::initializer_list<int64_t> dimensions) {
  core::symbolic::SymShape shape;
  for (int64_t dimension : dimensions)
    shape.PushBack(core::symbolic::SymDim(dimension));
  return shape;
}

core::symbolic::SymShape DynamicLayerShape(int64_t width) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim("batch"));
  shape.PushBack(core::symbolic::SymDim(width));
  return shape;
}

utils::RepeatedProtoField<AttributeProto> LayerIntAttribute(const std::string &name,
                                                            int64_t value) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::INT);
  attribute.set_i(value);
  return attributes;
}

RewritingProto OptimizeLayers(core::builder::GraphBuilder &builder) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(onnx_patterns::CreatePattern("RepeatedLayer"));
  core::builder::GraphGraph graph(builder, std::move(patterns));
  return graph.Optimize();
}

bool IsLayer(const NodeProto &node) {
  return node.op_type().value() == "Layer" && node.domain().value() == "ai.rt";
}

const GraphProto *LayerBody(const NodeProto &node) {
  for (const auto &attribute : node.attribute())
    if (attribute.name().value() == "body" && attribute.has_g())
      return &attribute.g();
  return nullptr;
}

void ExpectClosedGraph(const GraphProto &graph, std::set<std::string> available = {}) {
  std::set<std::string> formal_inputs;
  for (const auto &input : graph.input())
    EXPECT_TRUE(formal_inputs.insert(input.name().value()).second);
  available.insert(formal_inputs.begin(), formal_inputs.end());
  for (const auto &initializer : graph.initializer())
    available.insert(initializer.name().value());
  for (const auto &node : graph.node()) {
    for (const auto &input : node.input()) {
      if (!input.value().empty()) {
        EXPECT_TRUE(available.count(input.value()))
            << node.op_type().value() << ": " << input.value();
      }
    }
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g())
        ExpectClosedGraph(attribute.g(), available);
      for (const auto &nested : attribute.graphs())
        ExpectClosedGraph(nested, available);
    }
    for (const auto &output : node.output()) {
      if (!output.value().empty()) {
        EXPECT_TRUE(available.insert(output.value()).second) << output.value();
      }
    }
  }
  for (const auto &output : graph.output())
    EXPECT_TRUE(available.count(output.name().value())) << output.name().value();
}

std::size_t ExpectLayerContracts(const GraphProto &graph, std::set<std::string> available = {}) {
  std::size_t count = 0;
  for (const auto &node : graph.node()) {
    if (!IsLayer(node))
      continue;
    ++count;
    const GraphProto *body = LayerBody(node);
    EXPECT_NE(body, nullptr);
    if (!body)
      continue;
    EXPECT_EQ(body->input().size(), node.input().size());
    EXPECT_EQ(body->output().size(), node.output().size());
    EXPECT_TRUE(body->initializer().empty());
    std::set<std::string> inputs;
    std::set<std::string> outputs;
    for (const auto &input : body->input()) {
      EXPECT_TRUE(input.has_type());
      EXPECT_TRUE(inputs.insert(input.name().value()).second);
    }
    for (const auto &output : body->output()) {
      EXPECT_TRUE(output.has_type());
      EXPECT_TRUE(outputs.insert(output.name().value()).second);
    }
    for (const auto &inner : body->node())
      EXPECT_FALSE(IsLayer(inner));
    ExpectClosedGraph(*body);
  }
  ExpectClosedGraph(graph, std::move(available));
  return count;
}

void CollectOriginalOperators(const GraphProto &graph,
                              std::map<std::string, std::string> &operators,
                              const std::map<std::string, std::string> &bindings = {}) {
  auto resolve = [&](const std::string &name) {
    const auto found = bindings.find(name);
    return found == bindings.end() ? name : found->second;
  };
  for (const auto &node : graph.node()) {
    if (IsLayer(node)) {
      const GraphProto *body = LayerBody(node);
      ASSERT_NE(body, nullptr);
      ASSERT_EQ(body->input().size(), node.input().size());
      ASSERT_EQ(body->output().size(), node.output().size());
      std::map<std::string, std::string> arguments;
      for (std::size_t index = 0; index < body->input().size(); ++index)
        arguments[body->input()[index].name().value()] = resolve(node.input()[index].value());
      for (std::size_t index = 0; index < body->output().size(); ++index)
        arguments[body->output()[index].name().value()] = resolve(node.output()[index].value());
      CollectOriginalOperators(*body, operators, arguments);
    } else {
      ASSERT_FALSE(node.name().value().empty());
      std::string signature = node.domain().value() + ":" + node.op_type().value();
      for (const auto &input : node.input())
        signature += "|in:" + resolve(input.value());
      for (const auto &output : node.output())
        signature += "|out:" + resolve(output.value());
      EXPECT_TRUE(operators.emplace(node.name().value(), signature).second)
          << "Duplicated original node " << node.name().value();
    }
  }
}

void ExpectOriginalOperators(const GraphProto &before, const GraphProto &after) {
  std::map<std::string, std::string> expected;
  std::map<std::string, std::string> actual;
  CollectOriginalOperators(before, expected);
  CollectOriginalOperators(after, actual);
  EXPECT_EQ(actual, expected);
}

void ExpectCoherentLayerMetadata(const core::builder::GraphBuilder &builder) {
  const GraphProto graph = builder.BuildGraph();
  core::compute::ComputeContext expected;
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  EXPECT_EQ(builder.Compute().ValueTags(), expected.ValueTags());
  EXPECT_EQ(builder.Compute().NodeTags(), expected.NodeTags());
  EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
  ASSERT_EQ(builder.Compute().Size(), builder.Nodes().size());

  std::set<std::string> inputs;
  for (const auto &input : builder.Inputs()) {
    inputs.insert(input.name().value());
    EXPECT_TRUE(builder.Shapes().Has(input.name().value()));
  }
  for (std::size_t index = 0; index < builder.Nodes().size(); ++index) {
    const auto &node = builder.Nodes()[index];
    for (const auto &reuse : builder.Compute().NodeReuse(index)) {
      ASSERT_GE(reuse.input_index, 0);
      ASSERT_LT(reuse.input_index, node.input_size());
      EXPECT_FALSE(inputs.count(node.input()[reuse.input_index].value()));
    }
  }
  GraphProto annotated = graph;
  builder.Compute().WriteToGraph(annotated);
  for (const auto &node : annotated.node())
    for (const auto &metadata : node.metadata_props())
      if (metadata.key().value() == core::compute::kReleaseAfterMetadataKey) {
        const std::string released = ";" + metadata.value().value() + ";";
        for (const auto &input : inputs)
          EXPECT_EQ(released.find(";" + input + ";"), std::string::npos);
      }
}

void AddLayerWeight(core::builder::GraphBuilder &builder, const std::string &name, int64_t input,
                    int64_t output, float value = 1.0f) {
  builder.MakeInitializer(
      MakeInitializer<float>(name.c_str(), {input, output},
                             std::vector<float>(static_cast<std::size_t>(input * output), value)));
}

void AddMatMulLayer(core::builder::GraphBuilder &builder, const std::string &input,
                    const std::string &weight, const std::string &prefix) {
  builder.MakeNode("MatMul", {input, weight}, {prefix + "_mm"}, "", prefix + "_matmul");
  builder.MakeNode("Relu", {prefix + "_mm"}, {prefix}, "", prefix + "_relu");
}

TEST(RepeatedLayerPattern, DynamicMatMulLayersKeepDistinctWeightsExternalAndAreIdempotent) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, DynamicLayerShape(4));
  AddLayerWeight(builder, "w0", 4, 4, 0.5f);
  AddLayerWeight(builder, "w1", 4, 4, 1.5f);
  AddLayerWeight(builder, "w2", 4, 4, 2.5f);
  AddMatMulLayer(builder, "x", "w0", "h0");
  AddMatMulLayer(builder, "h0", "w1", "h1");
  AddMatMulLayer(builder, "h1", "w2", "h2");
  builder.MakeOutput("h2");
  const GraphProto before = builder.BuildGraph();
  ASSERT_TRUE(builder.Shapes().Has("h2"));
  const auto shape = builder.GetShape("h2");

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  ASSERT_TRUE(builder.Shapes().Has("h2"));
  const GraphProto after = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  ExpectCoherentLayerMetadata(builder);
  EXPECT_EQ(builder.GetShape("h2"), shape);
  EXPECT_EQ(builder.Initializers().size(), 3u);
  EXPECT_EQ(builder.OpsetVersion("ai.rt"), 1);
  std::set<std::string> external_weights;
  for (const auto &node : after.node())
    if (IsLayer(node))
      for (const auto &input : node.input())
        if (input.value().starts_with("w"))
          external_weights.insert(input.value());
  EXPECT_EQ(external_weights, (std::set<std::string>{"w0", "w1", "w2"}));
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, NestedExtractionExportsRequiredImportAndPreservesReplayUndo) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
  AddLayerWeight(builder, "w0", 4, 4, 0.5f);
  AddLayerWeight(builder, "w1", 4, 4, 1.5f);
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    if (std::string(name) == "then_branch") {
      AddMatMulLayer(branch, "x", "w0", "h0");
      AddMatMulLayer(branch, "h0", "w1", "h1");
      branch.MakeOutput("h1");
    } else {
      branch.MakeNode("Relu", {"x"}, {"other"});
      branch.MakeOutput("other");
    }
    auto &reference = attributes.add();
    reference.set_name(std::string(name) + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s(name);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(onnx_patterns::CreatePattern("RepeatedLayer"));
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  optimizer.Canonicalize();
  const auto original = builder.BuildGraph();
  const auto original_version = builder.OpsetVersion("ai.rt");
  const auto plan = optimizer.Optimize();
  EXPECT_GE(ExpectLayerContracts(builder.Subgraph("then_branch").BuildGraph(), {"x", "w0", "w1"}),
            2u);
  EXPECT_EQ(builder.OpsetVersion("ai.rt"), 1);
  const auto optimized = builder.ToModel();
  const auto optimized_graph = builder.BuildGraph();
  EXPECT_NO_THROW(VerifyModel(optimized));
  EXPECT_NO_THROW(checker::check_model(optimized));
  bool recorded = false;
  for (const auto &opset : plan.required_opset_imports())
    recorded |= opset.domain() == "ai.rt" && opset.version() == 1;
  EXPECT_TRUE(recorded);
  auto restored = builder.UndoRewritings(plan);
  EXPECT_EQ(restored.OpsetVersion("ai.rt"), original_version);
  std::string difference;
  EXPECT_TRUE(restored.BuildGraph().Equals(original, &difference)) << difference;
  auto replayed = restored.ApplyRewritings(plan);
  EXPECT_TRUE(replayed.BuildGraph().Equals(optimized_graph, &difference)) << difference;
  EXPECT_EQ(replayed.OpsetVersion("ai.rt"), 1);
  EXPECT_NO_THROW(VerifyModel(replayed.ToModel()));
  EXPECT_NO_THROW(checker::check_model(replayed.ToModel()));
}

TEST(RepeatedLayerPattern, PreservesMapBoundaryDescriptorsWithTensorPlaceholders) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"local", 1}});
  ValueInfoProto map;
  map.set_name("map");
  auto *map_type = map.mutable_type()->mutable_map_type();
  map_type->set_key_type(TensorProto::INT64);
  map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  map_type->mutable_value_type()->mutable_tensor_type()->mutable_shape();
  builder.MakeInput(map);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  auto &function = builder.MakeLocalFunction("WithMap", "local");
  function.MakeInput(map);
  function.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  function.MakeInput("w", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
  function.MakeNode("MatMul", {"x", "w"}, {"mm"});
  function.MakeOutput("mm");
  function.MakeOutput(map);
  for (int i = 0; i < 2; ++i) {
    const std::string prefix = "h" + std::to_string(i);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(i + 1));
    builder.MakeNode("WithMap", {"map", "x", prefix + "_w"}, {prefix + "_mm", prefix + "_map"},
                     "local", prefix + "_call");
    builder.MakeNode("Relu", {prefix + "_mm"}, {prefix}, "", prefix + "_relu");
    builder.MakeOutput(prefix);
    ValueInfoProto output = map;
    output.set_name(prefix + "_map");
    builder.MakeOutput(output);
  }
  OptimizeLayers(builder);
  const auto graph = builder.BuildGraph();
  ASSERT_GE(ExpectLayerContracts(graph), 2u);
  size_t maps = 0;
  for (const auto &node : graph.node()) {
    if (!IsLayer(node))
      continue;
    const auto *body = LayerBody(node);
    for (std::size_t i = 0; i < body->input().size(); ++i)
      if (node.input(i) == "map") {
        EXPECT_TRUE(body->input(i).type().has_map_type());
        ++maps;
      }
    for (const auto &output : body->output())
      if (output.name().value().ends_with("_map")) {
        EXPECT_TRUE(output.type().has_map_type());
        ++maps;
      }
  }
  EXPECT_EQ(maps, 4u);
}

TEST(RepeatedLayerPattern, KeepsUnsupportedOptionalBoundariesOutsideLayers) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int i = 0; i < 2; ++i) {
    const std::string prefix = "h" + std::to_string(i);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(i + 1));
    AddMatMulLayer(builder, "x", prefix + "_w", prefix);
    builder.MakeNode("Optional", {prefix}, {prefix + "_optional"}, "", prefix + "_optional");
    ValueInfoProto output;
    output.set_name(prefix + "_optional");
    auto *tensor =
        output.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(2);
    tensor->mutable_shape()->add_dim()->set_dim_value(4);
    builder.MakeOutput(output);
  }
  OptimizeLayers(builder);
  const auto graph = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(graph), 2u);
  size_t optional = 0;
  for (const auto &node : graph.node()) {
    optional += node.op_type() == "Optional" ? 1 : 0;
    if (!IsLayer(node))
      continue;
    const auto *body = LayerBody(node);
    for (const auto &formal : body->input())
      EXPECT_FALSE(formal.type().has_optional_type());
    for (const auto &formal : body->output())
      EXPECT_FALSE(formal.type().has_optional_type());
  }
  EXPECT_EQ(optional, 2u);
  for (const auto &output : graph.output())
    EXPECT_TRUE(output.type().has_optional_type());
}

TEST(RepeatedLayerPattern, BoundedOccurrenceBatchesDoNotLeaveSingletonRemainders) {
  for (int count : {33, 65}) {
    SCOPED_TRACE(count);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1, 4}));
    AddLayerWeight(builder, "shared_weight", 4, 4);
    std::string input = "x";
    for (int index = 0; index < count; ++index) {
      const std::string prefix = "h" + std::to_string(index);
      AddMatMulLayer(builder, input, "shared_weight", prefix);
      input = prefix;
    }
    builder.MakeOutput(input);
    const GraphProto before = builder.BuildGraph();
    const auto shape = builder.GetShape(input);

    EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
    ASSERT_TRUE(builder.Shapes().Has(input));
    EXPECT_EQ(builder.GetShape(input), shape);
    const GraphProto after = builder.BuildGraph();
    const std::size_t layers = ExpectLayerContracts(after);
    EXPECT_GE(layers, 2u);
    EXPECT_LE(layers, static_cast<std::size_t>(count));
    ExpectOriginalOperators(before, after);
    EXPECT_EQ(builder.Nodes().size(), layers);
    std::size_t wrapped_nodes = 0;
    for (const auto &node : after.node()) {
      ASSERT_TRUE(IsLayer(node));
      const GraphProto *body = LayerBody(node);
      ASSERT_NE(body, nullptr);
      wrapped_nodes += body->node().size();
    }
    EXPECT_EQ(wrapped_nodes, static_cast<std::size_t>(count * 2));
    EXPECT_EQ(builder.Initializers().size(), 1u);
    ExpectCoherentLayerMetadata(builder);
    EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
    EXPECT_TRUE(after.Equals(builder.BuildGraph()));
  }
}

TEST(RepeatedLayerPattern, BoundsSearchAcrossIncompatibleAnchorFamily) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  constexpr int count = 128;
  std::string last;
  for (int index = 1; index <= count; ++index) {
    // The final compatible pair must remain unvisited after the shared budget is exhausted.
    const int width = index == count ? count - 1 : index;
    const std::string suffix = std::to_string(index);
    const std::string input = "x" + suffix;
    const std::string weight = "w" + suffix;
    const std::string output = "h" + suffix;
    builder.MakeInput(input, core::symbolic::TensorType::kFloat, LayerShape({1, width}));
    AddLayerWeight(builder, weight, width, 0);
    builder.MakeNode("MatMul", {input, weight}, {output + "_mm"}, "", output + "_matmul");
    builder.MakeNode("Relu", {output + "_mm"}, {output}, "", output + "_relu");
    last = output;
  }
  builder.MakeOutput(last);

  OptimizeLayers(builder);
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 0u);
}

TEST(RepeatedLayerPattern, IncludesInternalActivationInLayerPeakMemory) {
  constexpr int64_t width = 16;
  constexpr int64_t expanded_width = 1024;
  core::builder::GraphBuilder body("body", LayerSchemaLookup(), {{"", 28}});
  body.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1, width}));
  body.MakeInput("w1", core::symbolic::TensorType::kFloat, LayerShape({width, expanded_width}));
  body.MakeInput("w2", core::symbolic::TensorType::kFloat, LayerShape({expanded_width, width}));
  body.MakeNode("MatMul", {"x", "w1"}, {"hidden"});
  body.MakeNode("MatMul", {"hidden", "w2"}, {"body_result"});
  body.MakeOutput("body_result", core::symbolic::TensorType::kFloat, LayerShape({1, width}));

  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  builder.MakeInput("caller_x", core::symbolic::TensorType::kFloat, LayerShape({1, width}));
  builder.MakeInput("caller_w1", core::symbolic::TensorType::kFloat,
                    LayerShape({width, expanded_width}));
  builder.MakeInput("caller_w2", core::symbolic::TensorType::kFloat,
                    LayerShape({expanded_width, width}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("body");
  attribute.set_type(AttributeProto::GRAPH);
  *attribute.mutable_g() = body.BuildGraph();
  builder.MakeNode("Layer", {"caller_x", "caller_w1", "caller_w2"}, {"result"}, "ai.rt",
                   "large_layer", attributes);
  builder.MakeOutput("result", core::symbolic::TensorType::kFloat, LayerShape({1, width}));

  const auto &profile = builder.Compute().NodeMemory(0);
  EXPECT_EQ(std::get<int64_t>(std::get<core::expressions::DimType>(
                profile.at(core::compute::kNodeMemoryAlreadyAllocatedBytesKey))),
            135232);
  EXPECT_EQ(std::get<int64_t>(std::get<core::expressions::DimType>(
                profile.at(core::compute::kNodeMemoryOutputAllocationBytesKey))),
            64);
  EXPECT_EQ(std::get<int64_t>(std::get<core::expressions::DimType>(
                profile.at(core::compute::kNodeMemoryTotalBytesKey))),
            135296);

  ASSERT_EQ(builder.Subgraphs().size(), 1u);
  builder.Subgraphs().front()->MakeNode("Add", {"hidden", "hidden"}, {"body_extra"});
  EXPECT_EQ(std::get<int64_t>(std::get<core::expressions::DimType>(
                builder.Compute().NodeMemory(0).at(core::compute::kNodeMemoryTotalBytesKey))),
            139392);
}

TEST(RepeatedLayerPattern, KeepsCaptureOnlyIntermediatesLiveThroughNestedBranches) {
  core::builder::GraphBuilder body("body", LayerSchemaLookup(), {{"", 28}});
  body.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1024}));
  body.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
  body.MakeNode("Neg", {"x"}, {"a"});
  body.MakeNode("Abs", {"x"}, {"b"});
  utils::RepeatedProtoField<AttributeProto> branches;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &attribute = branches.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::GRAPH);
    auto *graph = attribute.mutable_g();
    graph->set_name(name);
    auto *add = graph->add_node();
    add->set_op_type("Add");
    add->add_input("a");
    add->add_input("b");
    add->add_output(name + "_result");
    auto *output = graph->add_output();
    output->set_name(name + "_result");
    auto *type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(1024);
  }
  body.MakeNode("If", {"condition"}, {"body_result"}, "", "", branches);
  body.MakeOutput("body_result");

  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  builder.MakeInput("caller_x", core::symbolic::TensorType::kFloat, LayerShape({1024}));
  builder.MakeInput("caller_condition", core::symbolic::TensorType::kBool, LayerShape({}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &attribute = attributes.add();
  attribute.set_name("body");
  attribute.set_type(AttributeProto::GRAPH);
  *attribute.mutable_g() = body.BuildGraph();
  builder.MakeNode("Layer", {"caller_x", "caller_condition"}, {"result"}, "ai.rt", "", attributes);
  builder.MakeOutput("result");

  auto internal_bytes = [](const core::compute::ComputeContext &context) {
    return std::get<int64_t>(
        std::get<core::compute::TaggedMemory>(
            context.NodeMemory(0).at(core::compute::kNodeMemoryIntermediatesKey))
            .at(""));
  };
  EXPECT_EQ(internal_bytes(builder.Compute()), 8192);
  ModelProto model;
  model.add_opset("", 28);
  model.add_opset("ai.rt", 1);
  *model.mutable_graph() = builder.BuildGraph();
  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  context.ComputeInPlaceReuseGraph(model.graph(), context.Shapes());
  EXPECT_EQ(internal_bytes(context), 8192);
}

TEST(RepeatedLayerPattern, PreservesResidualFanoutSharedWeightsAndIntermediateGraphOutputs) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  AddLayerWeight(builder, "shared", 4, 4);
  std::string input = "x";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddMatMulLayer(builder, input, "shared", prefix);
    builder.MakeNode("Add", {prefix, input}, {prefix + "_residual"}, "", prefix + "_add");
    input = prefix + "_residual";
  }
  builder.MakeNode("Neg", {"h1_mm"}, {"side"}, "", "side_neg");
  builder.MakeOutput("h0_mm");
  builder.MakeOutput("side");
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();
  ASSERT_TRUE(builder.Shapes().Has("h0_mm"));
  const auto intermediate_shape = builder.GetShape("h0_mm");

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  ASSERT_TRUE(builder.Shapes().Has("h0_mm"));
  const GraphProto after = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  ExpectCoherentLayerMetadata(builder);
  ASSERT_EQ(after.output().size(), before.output().size());
  for (std::size_t index = 0; index < before.output().size(); ++index)
    EXPECT_EQ(after.output()[index].name().value(), before.output()[index].name().value());
  EXPECT_EQ(builder.GetShape("h0_mm"), intermediate_shape);
  EXPECT_EQ(builder.Initializers().size(), 1u);
}

TEST(RepeatedLayerPattern, AttentionLayersKeepOptionalInputsAndAttributes) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("q", core::symbolic::TensorType::kFloat, LayerShape({1, 2, 3, 4}));
  builder.MakeInput("k", core::symbolic::TensorType::kFloat, LayerShape({1, 2, 3, 4}));
  builder.MakeInput("v", core::symbolic::TensorType::kFloat, LayerShape({1, 2, 3, 4}));
  std::string input = "q";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "a" + std::to_string(index);
    builder.MakeNode("Attention", {input, "k", "v", ""}, {prefix + "_attention"}, "",
                     prefix + "_attention_node", LayerIntAttribute("is_causal", 1));
    builder.MakeNode("Relu", {prefix + "_attention"}, {prefix}, "", prefix + "_relu");
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  std::size_t attention_count = 0;
  for (const auto &node : after.node()) {
    const GraphProto *body = LayerBody(node);
    if (!body)
      continue;
    for (const auto &inner : body->node()) {
      if (inner.op_type().value() != "Attention")
        continue;
      ++attention_count;
      ASSERT_EQ(inner.input_size(), 4);
      EXPECT_TRUE(inner.input()[3].value().empty());
      EXPECT_EQ(GetAttributeOr<int64_t>(inner, "is_causal", 0), 1);
    }
  }
  EXPECT_EQ(attention_count, 3u);
}

TEST(RepeatedLayerPattern, ParallelProjectionAnchorsDoNotCreateOverlappingBodies) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, DynamicLayerShape(4));
  AddLayerWeight(builder, "left_weight", 4, 4, 0.5f);
  AddLayerWeight(builder, "right_weight", 4, 4, 1.5f);
  std::string input = "x";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    builder.MakeNode("MatMul", {input, "left_weight"}, {prefix + "_left"}, "", prefix + "_left_mm");
    builder.MakeNode("MatMul", {input, "right_weight"}, {prefix + "_right"}, "",
                     prefix + "_right_mm");
    builder.MakeNode("Add", {prefix + "_left", prefix + "_right"}, {prefix + "_sum"}, "",
                     prefix + "_add");
    builder.MakeNode("Relu", {prefix + "_sum"}, {prefix}, "", prefix + "_relu");
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, SameShapedSequentialProjectionRolesFormWholeLayers) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, DynamicLayerShape(4));
  std::string input = "x";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    AddLayerWeight(builder, prefix + "_v", 4, 4, static_cast<float>(index + 4));
    AddMatMulLayer(builder, input, prefix + "_w", prefix + "_activated");
    builder.MakeNode("MatMul", {prefix + "_activated", prefix + "_v"}, {prefix + "_projection"}, "",
                     prefix + "_second_matmul");
    builder.MakeNode("Add", {prefix + "_projection", input}, {prefix}, "", prefix + "_residual");
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  ASSERT_EQ(ExpectLayerContracts(after), 3u);
  ExpectOriginalOperators(before, after);
  for (const auto &node : after.node()) {
    const GraphProto *body = LayerBody(node);
    if (!body)
      continue;
    EXPECT_EQ(std::count_if(body->node().begin(), body->node().end(),
                            [](const NodeProto &inner) { return inner.op_type() == "MatMul"; }),
              2);
  }
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, AttentionAndOutputProjectionStayInTheSameLayer) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  const auto shape = LayerShape({1, 2, 3, 4});
  for (const char *name : {"q", "k", "v"})
    builder.MakeInput(name, core::symbolic::TensorType::kFloat, shape);
  std::string input = "q";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("Attention", {input, "k", "v"}, {prefix + "_attention"}, "",
                     prefix + "_attention_node");
    builder.MakeNode("MatMul", {prefix + "_attention", prefix + "_w"}, {prefix + "_projection"}, "",
                     prefix + "_projection_node");
    builder.MakeNode("Add", {prefix + "_projection", input}, {prefix}, "", prefix + "_residual");
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  ASSERT_EQ(ExpectLayerContracts(after), 3u);
  ExpectOriginalOperators(before, after);
  for (const auto &node : after.node()) {
    const GraphProto *body = LayerBody(node);
    if (!body)
      continue;
    EXPECT_EQ(std::count_if(body->node().begin(), body->node().end(),
                            [](const NodeProto &inner) { return inner.op_type() == "Attention"; }),
              1);
    EXPECT_EQ(std::count_if(body->node().begin(), body->node().end(),
                            [](const NodeProto &inner) { return inner.op_type() == "MatMul"; }),
              1);
  }
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, LocalCompositeCallsRetainTheirDefinitions) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"local.test", 1}});
  auto &function = builder.MakeLocalFunction("Projection", "local.test");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, DynamicLayerShape(4));
  function.MakeInput("w", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
  function.MakeNode("MatMul", {"a", "w"}, {"m"});
  function.MakeNode("Relu", {"m"}, {"b"});
  function.MakeOutput("b");
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, DynamicLayerShape(4));
  AddLayerWeight(builder, "shared", 4, 4);
  std::string input = "x";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    builder.MakeNode("Projection", {input, "shared"}, {prefix + "_projected"}, "local.test",
                     prefix + "_call");
    builder.MakeNode("Add", {prefix + "_projected", input}, {prefix}, "", prefix + "_add");
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();
  ASSERT_TRUE(builder.Shapes().Has(input));
  const auto shape = builder.GetShape(input);

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  ASSERT_TRUE(builder.Shapes().Has(input));
  const GraphProto after = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  EXPECT_TRUE(builder.HasLocalFunction("Projection"));
  EXPECT_EQ(builder.LocalFunction("Projection").Nodes().size(), 2u);
  EXPECT_EQ(builder.GetShape(input), shape);
  EXPECT_EQ(builder.ToModel().functions().size(), 1u);
}

TEST(RepeatedLayerPattern, DefaultPatternsDoNotWrapButExplicitSelectionDoes) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  AddLayerWeight(builder, "w", 4, 4);
  AddMatMulLayer(builder, "x", "w", "h0");
  AddMatMulLayer(builder, "h0", "w", "h1");
  builder.MakeOutput("h1");
  onnx_patterns::RegisterPatterns();
  core::builder::GraphGraph defaults(builder, core::builder::CreateRegisteredPatterns());
  defaults.Optimize();
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 0u);
  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_GE(ExpectLayerContracts(builder.BuildGraph()), 2u);
}

TEST(RepeatedLayerPattern, RejectsDifferentAttributesAndDomains) {
  for (bool different_domain : {false, true}) {
    SCOPED_TRACE(different_domain);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"other", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
    AddLayerWeight(builder, "w", 4, 4);
    builder.MakeNode("MatMul", {"x", "w"}, {"m0"}, "", "matmul0");
    utils::RepeatedProtoField<AttributeProto> attributes;
    auto &alpha = attributes.add();
    alpha.set_name("alpha");
    alpha.set_type(AttributeProto::FLOAT);
    alpha.set_f(0.1f);
    builder.MakeNode("LeakyRelu", {"m0"}, {"h0"}, "", "activation0", attributes);
    builder.MakeNode("MatMul", {"h0", "w"}, {"m1"}, "", "matmul1");
    if (!different_domain)
      attributes[0].set_f(0.2f);
    builder.MakeNode("LeakyRelu", {"m1"}, {"h1"}, different_domain ? "other" : "", "activation1",
                     attributes);
    builder.MakeOutput("h1", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
    EXPECT_EQ(builder.GetShape("h0"), builder.GetShape("h1"));
    const GraphProto before = builder.BuildGraph();
    EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
    EXPECT_TRUE(before.Equals(builder.BuildGraph()));
  }
}

GraphProto CapturingLayerBranch(const std::string &capture) {
  GraphProto branch;
  branch.set_name("capturing_branch");
  branch.add_node(MakeNode("Identity", {capture}, {"branch_result"}, "", "capture_identity"));
  auto *output = branch.add_output();
  output->set_name("branch_result");
  auto *tensor = output->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
  tensor->mutable_shape()->add_dim()->set_dim_value(4);
  return branch;
}

utils::RepeatedProtoField<AttributeProto> LayerBranches(const GraphProto &branch) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const char *name : {"then_branch", "else_branch"}) {
    auto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::GRAPH);
    *attribute.mutable_g() = branch;
  }
  return attributes;
}

TEST(RepeatedLayerPattern, ExportsValuesCapturedOnlyByDoublyNestedBranches) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
  AddLayerWeight(builder, "w", 4, 4);
  AddMatMulLayer(builder, "x", "w", "h0");
  AddMatMulLayer(builder, "h0", "w", "h1");
  AddMatMulLayer(builder, "h1", "w", "h2");
  GraphProto outer = CapturingLayerBranch("h0_mm");
  outer.clear_node();
  auto nested_if = MakeNode("If", {"condition"}, {"branch_result"}, "", "nested_if");
  for (const auto &attribute : LayerBranches(CapturingLayerBranch("h0_mm")))
    *nested_if.add_attribute() = attribute;
  outer.add_node(nested_if);
  builder.MakeNode("If", {"condition"}, {"side"}, "", "outer_if", LayerBranches(outer));
  builder.MakeOutput("h2");
  builder.MakeOutput("side");
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  EXPECT_GE(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  bool exposes_capture = false;
  for (const auto &node : after.node())
    if (IsLayer(node))
      for (const auto &output : node.output())
        exposes_capture |= output.value() == "h0_mm";
  EXPECT_TRUE(exposes_capture);
}

TEST(RepeatedLayerPattern, ExtractsRenamedIfBranchesWithExplicitNestedCaptureInputs) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("bias", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
  std::string input = "x";
  for (int index = 0; index < 2; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {input, prefix + "_w"}, {prefix + "_mm"}, "", prefix + "_matmul");
    GraphProto branch;
    branch.set_name(prefix + "_renamed_branch");
    branch.add_node(
        MakeNode("Identity", {prefix + "_mm"}, {prefix + "_captured"}, "", prefix.c_str()));
    branch.add_node(MakeNode("Add", {prefix + "_captured", "bias"}, {prefix + "_branch_result"}, "",
                             (prefix + "_branch_add").c_str()));
    auto *output = branch.add_output();
    output->set_name(prefix + "_branch_result");
    auto *tensor = output->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(2);
    tensor->mutable_shape()->add_dim()->set_dim_value(4);
    builder.MakeNode("If", {"condition"}, {prefix}, "", prefix + "_if", LayerBranches(branch));
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();
  const auto shape = builder.GetShape(input);

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  ASSERT_TRUE(builder.Shapes().Has(input));
  EXPECT_EQ(builder.GetShape(input), shape);
  const GraphProto after = builder.BuildGraph();
  ASSERT_EQ(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  for (const auto &node : after.node()) {
    ASSERT_TRUE(IsLayer(node));
    const GraphProto *body = LayerBody(node);
    ASSERT_NE(body, nullptr);
    ASSERT_EQ(body->node().size(), 2u);
    EXPECT_EQ(body->node()[0].op_type(), "MatMul");
    EXPECT_EQ(body->node()[1].op_type(), "If");
    EXPECT_EQ(node.input_size(), 4);
    for (const std::string &capture : {std::string("condition"), std::string("bias")}) {
      bool found = false;
      for (std::size_t index = 0; index < node.input().size(); ++index) {
        if (node.input()[index].value() == capture) {
          found = true;
          const std::string &formal = body->input()[index].name().value();
          if (capture == "condition") {
            EXPECT_EQ(body->node()[1].input()[0].value(), formal);
          } else {
            for (const auto &attribute : body->node()[1].attribute())
              if (attribute.has_g()) {
                const auto &branch_nodes = attribute.g().node();
                ASSERT_FALSE(branch_nodes.empty());
                const auto &addition = branch_nodes[branch_nodes.size() - 1];
                ASSERT_EQ(addition.op_type(), "Add");
                EXPECT_EQ(addition.input()[1].value(), formal);
              }
          }
        }
      }
      EXPECT_TRUE(found) << "Missing explicit captured input " << capture;
    }
  }
  ExpectCoherentLayerMetadata(builder);
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, EqualEdgeCutsPreferSmallerKnownTensors) {
  for (bool widen : {false, true}) {
    SCOPED_TRACE(widen);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    for (int index = 0; index < 2; ++index) {
      const std::string prefix = "h" + std::to_string(index);
      AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
      AddMatMulLayer(builder, "x", prefix + "_w", prefix);
      builder.MakeNode("Cast", {prefix}, {prefix + "_cast"}, "", prefix + "_cast_node",
                       LayerIntAttribute("to", widen ? TensorProto::DOUBLE : TensorProto::FLOAT16));
      builder.MakeOutput(prefix + "_cast");
    }
    const GraphProto before = builder.BuildGraph();

    EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
    const GraphProto after = builder.BuildGraph();
    ASSERT_EQ(ExpectLayerContracts(after), 2u);
    ExpectOriginalOperators(before, after);
    std::size_t wrapped_casts = 0;
    for (const auto &node : after.node()) {
      const GraphProto *body = LayerBody(node);
      if (body)
        for (const auto &inner : body->node())
          wrapped_casts += inner.op_type().value() == "Cast";
    }
    EXPECT_EQ(wrapped_casts, widen ? 0u : 2u);
  }
}

TEST(RepeatedLayerPattern, FewerCrossingEdgesTakePrecedenceOverLargerRegions) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("bias", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  AddLayerWeight(builder, "w", 4, 4);
  std::string input = "x";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddMatMulLayer(builder, input, "w", prefix);
    builder.MakeNode("Add", {prefix, "bias"}, {prefix + "_biased"}, "", prefix + "_add");
    input = prefix + "_biased";
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  EXPECT_EQ(ExpectLayerContracts(after), 3u);
  ExpectOriginalOperators(before, after);
  for (const auto &node : after.node()) {
    const GraphProto *body = LayerBody(node);
    if (body) {
      for (const auto &inner : body->node())
        EXPECT_NE(inner.op_type().value(), "Add");
    }
  }
}

TEST(RepeatedLayerPattern, RejectsContractionThatReentersThroughAnUnmatchedSideBranch) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  std::string input = "x";
  for (int index = 0; index < 2; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {input, prefix + "_w"}, {prefix + "_mm"}, "", prefix + "_matmul");
    builder.MakeNode(index == 0 ? "Neg" : "Abs", {prefix + "_mm"}, {prefix + "_side"}, "",
                     prefix + "_unmatched_branch");
    builder.MakeNode("Add", {prefix + "_mm", prefix + "_side"}, {prefix}, "", prefix + "_reentry");
    input = prefix;
  }
  builder.MakeOutput(input);
  const GraphProto before = builder.BuildGraph();

  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 0u);
  ExpectCoherentLayerMetadata(builder);
}

TEST(RepeatedLayerPattern, UnknownSizeCutsRemainValidAndDeterministic) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, DynamicLayerShape(4));
  for (int index = 0; index < 2; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    AddMatMulLayer(builder, "x", prefix + "_w", prefix);
    builder.MakeNode("Cast", {prefix}, {prefix + "_cast"}, "", prefix + "_cast_node",
                     LayerIntAttribute("to", TensorProto::DOUBLE));
    builder.MakeOutput(prefix + "_cast");
  }
  const GraphProto before = builder.BuildGraph();
  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  ASSERT_EQ(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  for (const auto &node : after.node()) {
    const GraphProto *body = LayerBody(node);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(body->node().size(), 3u);
  }
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, KnownSizeBoundaryBeatsEqualEdgeCutWithUnknownNonZeroSize) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 2; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    AddMatMulLayer(builder, "x", prefix + "_w", prefix);
    builder.MakeNode("NonZero", {prefix}, {prefix + "_nonzero"}, "", prefix + "_nonzero_node");
    builder.MakeOutput(prefix + "_nonzero");
    ASSERT_TRUE(builder.Shapes().Has(prefix + "_nonzero"));
    ASSERT_EQ(builder.GetShape(prefix + "_nonzero").Shape().Rank(), 2u);
    EXPECT_FALSE(builder.GetShape(prefix + "_nonzero").Shape()[1].IsInt());
  }
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  ASSERT_EQ(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  std::size_t external_nonzero = 0;
  for (const auto &node : after.node()) {
    external_nonzero += node.op_type() == "NonZero";
    const GraphProto *body = LayerBody(node);
    if (body) {
      ASSERT_EQ(body->node().size(), 2u);
      for (const auto &inner : body->node())
        EXPECT_NE(inner.op_type(), "NonZero");
    }
  }
  EXPECT_EQ(external_nonzero, 2u);
}

TEST(RepeatedLayerPattern, TypedUnknownRankBoundariesRemainUnshaped) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  ValueInfoProto input;
  input.set_name("x");
  input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  builder.MakeInput(input);
  for (int index = 0; index < 2; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    AddMatMulLayer(builder, "x", prefix + "_w", prefix);
    builder.MakeNode("Cast", {prefix}, {prefix + "_cast"}, "", prefix + "_cast_node",
                     LayerIntAttribute("to", TensorProto::DOUBLE));
    ValueInfoProto output;
    output.set_name(prefix + "_cast");
    output.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
    builder.MakeOutput(output);
  }
  const GraphProto before = builder.BuildGraph();

  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  ASSERT_EQ(ExpectLayerContracts(after), 2u);
  ExpectOriginalOperators(before, after);
  for (const auto &node : after.node()) {
    const GraphProto *body = LayerBody(node);
    ASSERT_NE(body, nullptr);
    ASSERT_EQ(body->node().size(), 3u);
    bool found_unknown_rank = false;
    for (std::size_t index = 0; index < node.input().size(); ++index) {
      if (node.input()[index].value() == "x") {
        found_unknown_rank = true;
        const auto &formal = body->input()[index].type().tensor_type();
        EXPECT_EQ(formal.elem_type(), TensorProto::FLOAT);
        EXPECT_FALSE(formal.has_shape());
      }
    }
    EXPECT_TRUE(found_unknown_rank);
    ASSERT_EQ(body->output().size(), 1u);
    EXPECT_EQ(body->output()[0].type().tensor_type().elem_type(), TensorProto::DOUBLE);
    EXPECT_FALSE(body->output()[0].type().tensor_type().has_shape());
  }
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, RejectsExtractionWithAnEntirelyUnknownBoundaryType) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  ValueInfoProto input;
  input.set_name("x");
  builder.MakeInput(input);
  for (int index = 0; index < 2; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddLayerWeight(builder, prefix + "_w", 4, 4, static_cast<float>(index + 1));
    AddMatMulLayer(builder, "x", prefix + "_w", prefix);
    builder.MakeNode("Cast", {prefix}, {prefix + "_cast"}, "", prefix + "_cast_node",
                     LayerIntAttribute("to", TensorProto::DOUBLE));
    ValueInfoProto output;
    output.set_name(prefix + "_cast");
    output.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
    builder.MakeOutput(output);
  }
  const GraphProto before = builder.BuildGraph();

  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 0u);
}

TEST(RepeatedLayerPattern, LeavesPreexistingLayerBodiesUnwrapped) {
  for (bool nested : {false, true}) {
    SCOPED_TRACE(nested);
    core::builder::GraphBuilder body("existing_body", LayerSchemaLookup(), {{"", 28}});
    body.MakeInput("activation", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    body.MakeInput("weight", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
    AddMatMulLayer(body, "activation", "weight", "h0");
    AddMatMulLayer(body, "h0", "weight", "h1");
    AddMatMulLayer(body, "h1", "weight", "h2");
    body.MakeOutput("h2", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    GraphProto body_proto = body.BuildGraph();
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    AddLayerWeight(builder, "w", 4, 4);
    std::vector<std::string> inputs{"x", "w"};
    if (nested) {
      body_proto.clear_input();
      core::builder::GraphBuilder enclosing("enclosing_body", LayerSchemaLookup(), {{"", 28}});
      enclosing.MakeInput("activation", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
      enclosing.MakeInput("weight", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
      enclosing.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
      enclosing.MakeNode("If", {"condition"}, {"chosen"}, "", "nested_if",
                         LayerBranches(body_proto));
      enclosing.MakeOutput("chosen", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
      body_proto = enclosing.BuildGraph();
      builder.MakeInput("outer_condition", core::symbolic::TensorType::kBool, LayerShape({}));
      inputs.push_back("outer_condition");
    }
    utils::RepeatedProtoField<AttributeProto> attributes;
    auto &attribute = attributes.add();
    attribute.set_name("body");
    attribute.set_type(AttributeProto::GRAPH);
    *attribute.mutable_g() = body_proto;
    builder.MakeNode("Layer", inputs, {"result"}, "ai.rt", "already_wrapped", attributes);
    builder.MakeOutput("result");
    const GraphProto before = builder.BuildGraph();

    EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
    const GraphProto after = builder.BuildGraph();
    EXPECT_EQ(ExpectLayerContracts(after), 1u);
    EXPECT_TRUE(before.Equals(after));
    EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
    EXPECT_TRUE(after.Equals(builder.BuildGraph()));
  }
}

TEST(RepeatedLayerPattern, ExtractedLayersExecuteLikeTheOriginalGraphAcrossRepeatedRuns) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1, 2}));
  builder.MakeInitializer(MakeInitializer<float>("w0", {2, 2}, {1, -1, 2, 1}));
  builder.MakeInitializer(MakeInitializer<float>("w1", {2, 2}, {0.5f, 1, -2, 1}));
  builder.MakeInitializer(MakeInitializer<float>("w2", {2, 2}, {1, -1, 2, 1}));
  AddMatMulLayer(builder, "x", "w0", "h0");
  AddMatMulLayer(builder, "h0", "w1", "h1");
  AddMatMulLayer(builder, "h1", "w2", "h2");
  builder.MakeOutput("h0_mm");
  builder.MakeOutput("h2");
  const ModelProto original = builder.ToModel();
  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  const ModelProto extracted = builder.ToModel();
  EXPECT_EQ(ExpectLayerContracts(extracted.graph()), 3u);

  auto execute = [](const ModelProto &model) {
    core::runtime::RuntimeSession session(model);
    core::runtime::RuntimeContext context;
    int matmul_calls = 0;
    int relu_calls = 0;
    context.RegisterCustomKernel(
        "", "MatMul", [&](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
          ++matmul_calls;
          const auto &left = runtime.Get(node.input(0));
          const auto &right = runtime.Get(node.input(1));
          ASSERT_EQ(left.shape.size(), 2u);
          ASSERT_EQ(right.shape.size(), 2u);
          ASSERT_EQ(left.shape[1], right.shape[0]);
          const int64_t rows = left.shape[0];
          const int64_t inner = left.shape[1];
          const int64_t columns = right.shape[1];
          std::vector<float> values(static_cast<std::size_t>(rows * columns), 0.0f);
          for (int64_t row = 0; row < rows; ++row)
            for (int64_t column = 0; column < columns; ++column)
              for (int64_t index = 0; index < inner; ++index)
                values[static_cast<std::size_t>(row * columns + column)] +=
                    left.AsFloat()[row * inner + index] * right.AsFloat()[index * columns + column];
          runtime.Put(node.output(0),
                      core::runtime::Tensor::FromFloat(node.output(0), {rows, columns}, values));
        });
    context.RegisterCustomKernel(
        "", "Relu", [&](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
          ++relu_calls;
          const auto &input = runtime.Get(node.input(0));
          std::vector<float> values(input.AsFloat(), input.AsFloat() + input.element_count());
          for (auto &value : values)
            value = std::max(value, 0.0f);
          runtime.Put(node.output(0),
                      core::runtime::Tensor::FromFloat(node.output(0), input.shape, values));
        });
    std::vector<std::vector<float>> results;
    for (const std::vector<float> &feed : {std::vector<float>{-2, 3}, std::vector<float>{1, 2}}) {
      context.Put("x", core::runtime::Tensor::FromFloat("x", {1, 2}, feed));
      session.Run(context);
      std::vector<float> values;
      for (const char *name : {"h0_mm", "h2"}) {
        const auto &output = context.Get(name);
        values.insert(values.end(), output.AsFloat(), output.AsFloat() + output.element_count());
      }
      results.push_back(std::move(values));
      EXPECT_EQ(context.Get("x").AsFloat()[0], feed[0]);
      EXPECT_EQ(context.Get("x").AsFloat()[1], feed[1]);
    }
    EXPECT_EQ(matmul_calls, 6);
    EXPECT_EQ(relu_calls, 6);
    return results;
  };
  const auto baseline = execute(original);
  EXPECT_EQ(baseline, (std::vector<std::vector<float>>{{4, 5, 18, 9}, {5, 1, 12.5f, 5.5f}}));
  EXPECT_EQ(execute(extracted), baseline);
}

} // namespace
} // namespace Test
