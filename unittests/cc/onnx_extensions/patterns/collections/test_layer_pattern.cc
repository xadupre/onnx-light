// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/shapes/dispatch_table.h"
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
#include <utility>
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

std::map<std::string, GraphProto> &LayerBodies() {
  static std::map<std::string, GraphProto> layer_bodies;
  return layer_bodies;
}

void CacheLayerBodies(core::builder::GraphBuilder &builder) {
  auto &layer_bodies = LayerBodies();
  layer_bodies.clear();
  const auto collect = [&](const auto &self, core::builder::GraphBuilder &scope) -> void {
    for (const auto &function_builder : scope.LocalFunctions()) {
      if (function_builder->name() == "Layer" || function_builder->name().find("Layer_") == 0) {
        FunctionProto function = function_builder->ToFunction("ai.rt");
        GraphProto body;
        std::unordered_map<std::string, ValueInfoProto> value_info;
        for (const auto &value : function.value_info())
          value_info.emplace(value.name(), value);
        for (const auto &name : function.input()) {
          ValueInfoProto value;
          value.set_name(name);
          if (const auto found = value_info.find(name); found != value_info.end())
            value = found->second;
          *body.add_input() = std::move(value);
        }
        for (const auto &name : function.output()) {
          ValueInfoProto value;
          value.set_name(name);
          if (const auto found = value_info.find(name); found != value_info.end())
            value = found->second;
          *body.add_output() = std::move(value);
        }
        for (const auto &value : function.value_info())
          if (std::none_of(function.input().begin(), function.input().end(),
                           [&](const std::string &name) { return name == value.name(); }) &&
              std::none_of(function.output().begin(), function.output().end(),
                           [&](const std::string &name) { return name == value.name(); }))
            *body.add_value_info() = value;
        for (const auto &node : function.node())
          *body.add_node() = node;
        layer_bodies[function.name()] = std::move(body);
      }
      self(self, *function_builder);
    }
    for (const auto &subgraph : scope.Subgraphs())
      self(self, *subgraph);
  };
  collect(collect, builder);
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
  RewritingProto rewrites = graph.Optimize();
  CacheLayerBodies(builder);
  return rewrites;
}

bool IsLayer(const NodeProto &node) {
  return node.domain().value() == "ai.rt" &&
         (node.op_type() == "Layer" || node.op_type().find("Layer_") == 0);
}

const GraphProto *LayerBody(const NodeProto &node) {
  auto &layer_bodies = LayerBodies();
  const auto found = layer_bodies.find(node.op_type());
  return found == layer_bodies.end() ? nullptr : &found->second;
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
                              std::vector<std::string> &order,
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
      CollectOriginalOperators(*body, operators, order, arguments);
    } else {
      ASSERT_FALSE(node.name().value().empty());
      order.push_back(node.name());
      std::string signature =
          node.domain().value() + ":" + node.op_type().value() + ":" + node.overload().value();
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
  std::vector<std::string> expected_order, actual_order;
  CollectOriginalOperators(before, expected, expected_order);
  CollectOriginalOperators(after, actual, actual_order);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(actual_order, expected_order);
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
  const ModelProto model = builder.ToModel();
  EXPECT_EQ(model.functions().size(), ExpectLayerContracts(model.graph()));
  for (const auto &function : model.functions())
    EXPECT_TRUE(function.name() == "Layer" || function.name().value().starts_with("Layer_"));
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
  CacheLayerBodies(builder);
  EXPECT_GE(ExpectLayerContracts(builder.Subgraph("then_branch").BuildGraph(), {"x", "w0", "w1"}),
            2u);
  EXPECT_EQ(builder.OpsetVersion("ai.rt"), 1);
  const auto optimized = builder.ToModel();
  const auto optimized_graph = builder.BuildGraph();
  EXPECT_NO_THROW(VerifyModel(optimized));
  EXPECT_NO_THROW(checker::check_model(optimized));
  bool recorded = false;
  for (const auto &opset : plan.required_opset_imports())
    recorded |= opset.domain() == "ai.rt";
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

TEST(RepeatedLayerPattern, LocalFunctionExtractionUsesSourceOpsetsAndPreservesNodeMetadata) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  auto &source = builder.MakeLocalFunction("Source", "source.local");
  source.SetOpsetVersion("", 23);
  source.SetOpsetVersion("ai.rt", 1);
  source.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 3; ++index) {
    const std::string suffix = std::to_string(index);
    source.MakeInput("w" + suffix, core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
    AddMatMulLayer(source, index == 0 ? "x" : "h" + std::to_string(index - 1), "w" + suffix,
                   "h" + suffix);
  }
  source.MakeOutput("h2");
  builder.SetOpsetVersion("source.local", 1);
  builder.SetOpsetVersion("ai.rt", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 3; ++index)
    builder.MakeInput("w" + std::to_string(index), core::symbolic::TensorType::kFloat,
                      LayerShape({4, 4}));
  builder.MakeNode("Source", {"x", "w0", "w1", "w2"}, {"y"}, "source.local");
  builder.MakeOutput("y");

  ModelProto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1u);
  auto *source_matmul = model.mutable_functions(0)->mutable_node(0);
  source_matmul->set_doc_string("source node documentation");
  auto *metadata = source_matmul->add_metadata_props();
  metadata->set_key("source");
  metadata->set_value("preserved");
  auto *device = source_matmul->add_device_configurations();
  device->set_configuration_id("sharded");
  device->set_pipeline_stage(2);

  core::builder::GraphBuilder imported(model, LayerSchemaLookup());
  auto &imported_source = imported.LocalFunction("Source");
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(onnx_patterns::CreatePattern("RepeatedLayer"));
  core::builder::GraphGraph optimizer(imported_source, std::move(patterns));
  ASSERT_FALSE(imported_source.Nodes().empty());
  const auto match = optimizer.Patterns().front()->Match(optimizer, imported_source.Nodes()[0]);
  ASSERT_NE(match.pattern, nullptr) << match.ToString();
  ASSERT_FALSE(optimizer.Optimize(1).rewrites().empty());
  const ModelProto extracted = imported.ToModel();
  EXPECT_NO_THROW(VerifyModel(extracted));
  EXPECT_NO_THROW({
    core::builder::GraphBuilder reimported(extracted, LayerSchemaLookup());
    EXPECT_TRUE(reimported.GetShape("y").Shape() == LayerShape({2, 4}));
  });
  const auto source_function =
      std::find_if(extracted.functions().begin(), extracted.functions().end(),
                   [](const FunctionProto &function) { return function.name() == "Source"; });
  ASSERT_NE(source_function, extracted.functions().end());
  EXPECT_TRUE(std::any_of(
      source_function->opset_import().begin(), source_function->opset_import().end(),
      [](const auto &opset) { return opset.domain().empty() && opset.version() == 23; }));
  EXPECT_TRUE(std::any_of(
      source_function->opset_import().begin(), source_function->opset_import().end(),
      [](const auto &opset) { return opset.domain() == "ai.rt" && opset.version() == 1; }));
  EXPECT_FALSE(std::any_of(source_function->opset_import().begin(),
                           source_function->opset_import().end(),
                           [](const auto &opset) { return opset.domain() == "ai.onnx"; }));
  const auto layer_function =
      std::find_if(extracted.functions().begin(), extracted.functions().end(),
                   [](const FunctionProto &function) { return function.name() == "Layer"; });
  ASSERT_NE(layer_function, extracted.functions().end());
  EXPECT_TRUE(std::any_of(
      layer_function->opset_import().begin(), layer_function->opset_import().end(),
      [](const auto &opset) { return opset.domain().empty() && opset.version() == 23; }));
  EXPECT_TRUE(std::any_of(
      layer_function->opset_import().begin(), layer_function->opset_import().end(),
      [](const auto &opset) { return opset.domain() == "ai.rt" && opset.version() == 1; }));
  for (const auto &source_opset : source_function->opset_import()) {
    EXPECT_TRUE(std::any_of(layer_function->opset_import().begin(),
                            layer_function->opset_import().end(), [&](const auto &layer_opset) {
                              return layer_opset.domain() == source_opset.domain() &&
                                     layer_opset.version() == source_opset.version();
                            }));
  }
  ASSERT_FALSE(layer_function->node().empty());
  EXPECT_EQ(layer_function->node(0).doc_string(), "source node documentation");
  EXPECT_TRUE(std::any_of(layer_function->node(0).metadata_props().begin(),
                          layer_function->node(0).metadata_props().end(), [](const auto &entry) {
                            return entry.key() == "source" && entry.value() == "preserved";
                          }));
  ASSERT_EQ(layer_function->node(0).device_configurations().size(), 1u);
  EXPECT_EQ(layer_function->node(0).device_configurations(0).configuration_id(), "sharded");
  EXPECT_EQ(layer_function->node(0).device_configurations(0).pipeline_stage(), 2);

  core::builder::GraphGraph root_optimizer(
      imported, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
  root_optimizer.Canonicalize();
  ModelProto before_root_optimization = imported.ToModel();
  core::builder::ConstantFoldingOptions folding;
  folding.enabled = false;
  const auto root_plan = root_optimizer.Optimize(1, nullptr, folding);
  auto restored = imported.UndoRewritings(root_plan);
  ModelProto restored_model = restored.ToModel();
  const auto order_opsets = [](auto &proto) {
    std::map<std::string, int64_t> imports;
    for (const auto &opset : proto.opset_import())
      imports.emplace(opset.domain().value(), opset.version());
    proto.clear_opset_import();
    for (const auto &[domain, version] : imports)
      proto.add_opset(domain, version);
  };
  for (ModelProto *snapshot : {&before_root_optimization, &restored_model}) {
    order_opsets(*snapshot);
    for (auto &function : snapshot->ref_functions())
      order_opsets(function);
  }
  std::string difference;
  EXPECT_TRUE(restored_model.Equals(before_root_optimization, &difference)) << difference;
  EXPECT_NO_THROW(VerifyModel(restored_model));
  EXPECT_TRUE(restored.GetShape("y").Shape() == LayerShape({2, 4}));
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

TEST(ComputeContext, IncludesLocalFunctionActivationsInExportedAndFullProfiles) {
  constexpr int64_t width = 16;
  constexpr int64_t expanded_width = 1024;
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  auto &function = builder.MakeLocalFunction("Projection", "ai.rt", builder.OpsetVersions());
  function.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1, width}));
  function.MakeInput("w1", core::symbolic::TensorType::kFloat, LayerShape({width, expanded_width}));
  function.MakeInput("w2", core::symbolic::TensorType::kFloat, LayerShape({expanded_width, width}));
  function.MakeNode("MatMul", {"x", "w1"}, {"hidden"});
  function.MakeNode("MatMul", {"hidden", "w2"}, {"function_result"});
  function.MakeOutput("function_result");
  builder.MakeInput("caller_x", core::symbolic::TensorType::kFloat, LayerShape({1, width}));
  builder.MakeInput("caller_w1", core::symbolic::TensorType::kFloat,
                    LayerShape({width, expanded_width}));
  builder.MakeInput("caller_w2", core::symbolic::TensorType::kFloat,
                    LayerShape({expanded_width, width}));
  builder.MakeNode("Projection", {"caller_x", "caller_w1", "caller_w2"}, {"result"}, "ai.rt");
  builder.MakeOutput("result");
  EXPECT_TRUE(builder.Compute().Memory().empty());
  const ModelProto model = builder.ToModel();

  const auto internal_bytes = [](const core::compute::ComputeContext &context) {
    return std::get<int64_t>(std::get<core::expressions::DimType>(
        context.NodeMemory(0).at(core::compute::kNodeMemoryAlreadyAllocatedBytesKey)));
  };
  EXPECT_EQ(internal_bytes(builder.Compute()), 135232);

  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  const auto tags = context.ComputeValueAndNodeTags(model.graph());
  context.ComputeInPlaceReuseGraph(model.graph(), context.Shapes(), false, tags.first);
  EXPECT_EQ(internal_bytes(context), 135232);
}

TEST(ComputeContext, KeepsFunctionActivationsLiveThroughNestedBranches) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  auto &function = builder.MakeLocalFunction("BranchFunction", "ai.rt", builder.OpsetVersions());
  function.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1024}));
  function.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
  function.MakeNode("Neg", {"x"}, {"a"});
  function.MakeNode("Abs", {"x"}, {"b"});
  utils::RepeatedProtoField<AttributeProto> branches;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &attribute = branches.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::GRAPH);
    auto *graph = attribute.mutable_g();
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
  function.MakeNode("If", {"condition"}, {"function_result"}, "", "", branches);
  function.MakeOutput("function_result");
  builder.MakeInput("caller_x", core::symbolic::TensorType::kFloat, LayerShape({1024}));
  builder.MakeInput("caller_condition", core::symbolic::TensorType::kBool, LayerShape({}));
  builder.MakeNode("BranchFunction", {"caller_x", "caller_condition"}, {"result"}, "ai.rt");
  builder.MakeOutput("result");
  EXPECT_TRUE(builder.Compute().Memory().empty());
  const ModelProto model = builder.ToModel();

  const auto internal_bytes = [](const core::compute::ComputeContext &context) {
    return std::get<int64_t>(
        std::get<core::compute::TaggedMemory>(
            context.NodeMemory(0).at(core::compute::kNodeMemoryIntermediatesKey))
            .at(""));
  };
  EXPECT_EQ(internal_bytes(builder.Compute()), 8192);

  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  const auto tags = context.ComputeValueAndNodeTags(model.graph());
  context.ComputeInPlaceReuseGraph(model.graph(), context.Shapes(), false, tags.first);
  EXPECT_EQ(internal_bytes(context), 8192);
}

TEST(ComputeContext, IncludesLocalFunctionWorkspaceInIncrementalAndFullPeakMemory) {
  onnx_shapes::RegisterShapeFunctions();
  onnx_shapes::RegisterPeakMemoryFunctions();
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {"ai.rt", 1}});
  auto &function = builder.MakeLocalFunction("AttentionFunction", "ai.rt", builder.OpsetVersions());
  const auto shape = LayerShape({1, 2, 4, 8});
  const auto key_value_shape = LayerShape({1, 2, 16, 8});
  function.MakeInput("q", core::symbolic::TensorType::kFloat, shape);
  function.MakeInput("k", core::symbolic::TensorType::kFloat, key_value_shape);
  function.MakeInput("v", core::symbolic::TensorType::kFloat, key_value_shape);
  function.MakeNode("Attention", {"q", "k", "v"}, {"attention_result"});
  function.MakeOutput("attention_result");
  builder.MakeInput("caller_q", core::symbolic::TensorType::kFloat, shape);
  builder.MakeInput("caller_k", core::symbolic::TensorType::kFloat, key_value_shape);
  builder.MakeInput("caller_v", core::symbolic::TensorType::kFloat, key_value_shape);
  builder.MakeNode("AttentionFunction", {"caller_q", "caller_k", "caller_v"}, {"result"}, "ai.rt");
  builder.MakeOutput("result");

  const ModelProto model = builder.ToModel();
  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  context.AppendNodePeakMemory(model.graph().node(0), 0, core::symbolic::Device::kCPU);
  EXPECT_EQ(context.NodePeakMemory(0), 1 * 2 * 4 * 16 * 4);
  EXPECT_EQ(context.ComputePeakMemory(model.graph(), core::symbolic::Device::kCPU),
            (std::vector<int64_t>{1 * 2 * 4 * 16 * 4}));
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
  const ModelProto model = builder.ToModel();
  EXPECT_EQ(model.functions().size(), 1u + ExpectLayerContracts(model.graph()));
}

TEST(RepeatedLayerPattern, RootTraversalSurvivesHelpersAppendedByMultipleFunctions) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"local.test", 1}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
  constexpr std::size_t count = 8;
  for (std::size_t index = 0; index < count; ++index) {
    const std::string name = "Source" + std::to_string(index);
    auto &source = builder.MakeLocalFunction(name, "local.test");
    source.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    source.MakeInput("w", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
    std::string input = "x";
    for (int layer = 0; layer < 3; ++layer) {
      const std::string output = "h" + std::to_string(layer);
      AddMatMulLayer(source, input, "w", output);
      input = output;
    }
    source.MakeOutput(input);
    const std::string output = "y" + std::to_string(index);
    builder.MakeNode(name, {"x", "w"}, {output}, "local.test");
    builder.MakeOutput(output);
  }
  const auto original = builder.ToModel();
  const auto plan = OptimizeLayers(builder);
  EXPECT_EQ(plan.replaced_function_ids().size(), count);
  EXPECT_GT(builder.LocalFunctions().size(), count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &source = builder.LocalFunction("Source" + std::to_string(index));
    ASSERT_EQ(source.Nodes().size(), 3u);
    for (const auto &node : source.Nodes())
      EXPECT_EQ(node.domain(), "ai.rt");
  }
  const auto optimized = builder.ToModel();
  EXPECT_NO_THROW(VerifyModel(optimized));
  auto replayed = core::builder::GraphBuilder(original, LayerSchemaLookup()).ApplyRewritings(plan);
  EXPECT_TRUE(replayed.ToModel().Equals(optimized));
  const auto restored = builder.UndoRewritings(plan);
  for (std::size_t index = 0; index < count; ++index)
    EXPECT_EQ(restored.LocalFunction("Source" + std::to_string(index)).Nodes().size(), 6u);
}

TEST(RepeatedLayerPattern, ExtractedFunctionInheritsSourceFunctionOpsets) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"local.test", 1}});
  auto &source = builder.MakeLocalFunction("Source", "local.test");
  source.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  source.MakeInput("w", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
  std::string input = "x";
  for (int index = 0; index < 3; ++index) {
    const std::string prefix = "h" + std::to_string(index);
    AddMatMulLayer(source, input, "w", prefix);
    input = prefix;
  }
  source.MakeOutput(input);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
  builder.MakeNode("Source", {"x", "w"}, {"y"}, "local.test");
  builder.MakeOutput("y");

  EXPECT_FALSE(OptimizeLayers(source).rewrites().empty());
  const ModelProto model = builder.ToModel();
  const FunctionProto *source_function = nullptr;
  const FunctionProto *extracted_function = nullptr;
  for (const FunctionProto &function : model.functions()) {
    if (function.name() == "Source")
      source_function = &function;
    if (function.domain() == "ai.rt")
      extracted_function = &function;
  }
  ASSERT_NE(source_function, nullptr);
  ASSERT_NE(extracted_function, nullptr);
  bool source_imports_call_domain = false;
  bool source_calls_extracted_function = false;
  for (const auto &opset : source_function->opset_import())
    source_imports_call_domain |= opset.domain() == "ai.rt" && opset.version() == 1;
  for (const NodeProto &node : source_function->node())
    source_calls_extracted_function |= node.domain() == "ai.rt";
  EXPECT_TRUE(source_imports_call_domain);
  EXPECT_TRUE(source_calls_extracted_function);
  for (const auto &[domain, version] : source.OpsetVersions()) {
    const std::string exported_domain = domain == "ai.onnx" ? "" : domain;
    const auto found =
        std::find_if(extracted_function->opset_import().begin(),
                     extracted_function->opset_import().end(), [&](const auto &opset) {
                       return opset.domain() == exported_domain && opset.version() == version;
                     });
    EXPECT_NE(found, extracted_function->opset_import().end()) << domain;
  }
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(RepeatedLayerPattern, DefaultPatternsDoNotWrapButExplicitSelectionDoes) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  AddLayerWeight(builder, "w", 4, 4);
  AddMatMulLayer(builder, "x", "w", "h0");
  AddMatMulLayer(builder, "h0", "w", "h1");
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[0].set_doc_string("preserve node documentation");
  nodes[0].add_metadata("annotation", "preserve node metadata");
  nodes[0].add_device_configurations()->set_configuration_id("preserve device config");
  builder.MakeOutput("h1");
  onnx_patterns::RegisterPatterns();
  core::builder::GraphGraph defaults(builder, core::builder::CreateRegisteredPatterns());
  defaults.Optimize();
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 0u);
  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_GE(ExpectLayerContracts(builder.BuildGraph()), 2u);
  const ModelProto model = builder.ToModel();
  bool preserved = false;
  for (const FunctionProto &function : model.functions()) {
    if (function.domain() != "ai.rt")
      continue;
    for (const NodeProto &node : function.node()) {
      if (node.name() == "h0_matmul") {
        const auto annotation =
            std::find_if(node.metadata_props().begin(), node.metadata_props().end(),
                         [](const auto &entry) { return entry.key() == "annotation"; });
        preserved = node.doc_string() == "preserve node documentation" &&
                    annotation != node.metadata_props().end() &&
                    annotation->value() == "preserve node metadata" &&
                    node.device_configurations().size() == 1u &&
                    node.device_configurations()[0].configuration_id() == "preserve device config";
      }
    }
  }
  EXPECT_TRUE(preserved);
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

TEST(RepeatedLayerPattern, ExtractsStandardAndMicrosoftGeluWithoutMixingDomains) {
  for (const std::string domain : {"", "com.microsoft", "mixed"}) {
    SCOPED_TRACE(domain);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {"com.microsoft", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    for (int index = 0; index < 2; ++index) {
      const std::string suffix = std::to_string(index);
      AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
      builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
      const std::string node_domain =
          domain == "mixed" ? (index == 0 ? "" : "com.microsoft") : domain;
      builder.MakeNode("Gelu", {"m" + suffix}, {"y" + suffix}, node_domain, "gelu" + suffix);
      builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    }
    const GraphProto before = builder.BuildGraph();
    const auto plan = OptimizeLayers(builder);
    if (domain == "mixed") {
      EXPECT_TRUE(plan.rewrites().empty());
      EXPECT_TRUE(before.Equals(builder.BuildGraph()));
      continue;
    }
    EXPECT_FALSE(plan.rewrites().empty());
    EXPECT_EQ(builder.Nodes().size(), 2u);
    EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 2u);
    ExpectOriginalOperators(before, builder.BuildGraph());
    ExpectCoherentLayerMetadata(builder);
    EXPECT_NO_THROW(VerifyModel(builder.ToModel()));
    EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  }
}

TEST(RepeatedLayerPattern, ExtractsMicrosoftActivationsAndNormalizationsWithSideOutputs) {
  for (const std::string op :
       {"BiasGelu", "FastGelu", "SkipLayerNormalization", "SkipSimplifiedLayerNormalization"}) {
    SCOPED_TRACE(op);
    const bool normalization =
        op == "SkipLayerNormalization" || op == "SkipSimplifiedLayerNormalization";
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {"com.microsoft", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    builder.MakeInput("scale", core::symbolic::TensorType::kFloat, LayerShape({4}));
    for (int index = 0; index < 2; ++index) {
      const std::string suffix = std::to_string(index);
      AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
      builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
      const std::vector<std::string> inputs =
          normalization ? std::vector<std::string>{"m" + suffix, "x", "scale"}
                        : std::vector<std::string>{"m" + suffix, "scale"};
      const std::vector<std::string> outputs =
          normalization ? std::vector<std::string>{"y" + suffix, "", "", "sum" + suffix}
                        : std::vector<std::string>{"y" + suffix};
      builder.MakeNode(op, inputs, outputs, "com.microsoft", "contrib" + suffix);
      builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
      if (normalization)
        builder.MakeOutput("sum" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    }
    ModelProto original = builder.ToModel();
    if (normalization) {
      for (auto &node : original.mutable_graph()->ref_node()) {
        if (node.domain() == "com.microsoft") {
          node.ref_output()[1] = "";
          node.ref_output()[2] = "";
        }
      }
    }
    core::builder::GraphBuilder imported(original, LayerSchemaLookup());
    const GraphProto before = imported.BuildGraph();
    EXPECT_FALSE(OptimizeLayers(imported).rewrites().empty());
    EXPECT_EQ(ExpectLayerContracts(imported.BuildGraph()), 2u);
    ExpectOriginalOperators(before, imported.BuildGraph());
    ExpectCoherentLayerMetadata(imported);
    const auto model = imported.ToModel();
    for (const auto &function : model.functions()) {
      if (function.domain() != "ai.rt")
        continue;
      ASSERT_EQ(function.node().size(), 2u);
      const auto &contrib = function.node(1);
      EXPECT_EQ(contrib.domain(), "com.microsoft");
      EXPECT_EQ(contrib.op_type(), op);
      if (normalization) {
        EXPECT_EQ(contrib.output_size(), 4);
        EXPECT_TRUE(contrib.output(1).empty());
        EXPECT_TRUE(contrib.output(2).empty());
      }
      EXPECT_TRUE(std::any_of(function.opset_import().begin(), function.opset_import().end(),
                              [](const auto &opset) {
                                return opset.domain() == "com.microsoft" && opset.version() == 1;
                              }));
    }
    EXPECT_NO_THROW(VerifyModel(model));
    EXPECT_NO_THROW({
      core::builder::GraphBuilder reimported(model, LayerSchemaLookup());
      EXPECT_EQ(reimported.GetShape("y0").Shape(), LayerShape({2, 4}));
    });
  }
}

TEST(RepeatedLayerPattern, ExtractsMicrosoftAttentionHelpersWithExplicitState) {
  for (const std::string op : {"CausalConvWithState", "LinearAttention", "RotaryEmbedding"}) {
    SCOPED_TRACE(op);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {"com.microsoft", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({1, 2, 4}));
    if (op == "CausalConvWithState") {
      builder.MakeInput("conv_weight", core::symbolic::TensorType::kFloat, LayerShape({2, 1, 3}));
      builder.MakeInput("past", core::symbolic::TensorType::kFloat, LayerShape({1, 2, 2}));
    } else if (op == "LinearAttention") {
      builder.MakeInput("past", core::symbolic::TensorType::kFloat, LayerShape({1, 1, 4, 4}));
    } else {
      builder.MakeInput("positions", core::symbolic::TensorType::kInt64, LayerShape({1, 2}));
      builder.MakeInput("cos", core::symbolic::TensorType::kFloat, LayerShape({2, 2}));
      builder.MakeInput("sin", core::symbolic::TensorType::kFloat, LayerShape({2, 2}));
    }
    for (int index = 0; index < 2; ++index) {
      const std::string suffix = std::to_string(index);
      AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
      builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
      std::vector<std::string> inputs;
      std::vector<std::string> outputs = {"y" + suffix};
      utils::RepeatedProtoField<AttributeProto> attributes;
      if (op == "CausalConvWithState") {
        inputs = {"m" + suffix, "conv_weight", "", "past"};
      } else if (op == "LinearAttention") {
        inputs = {"m" + suffix, "m" + suffix, "m" + suffix, "past"};
        attributes = LayerIntAttribute("q_num_heads", 1);
        attributes.push_back(LayerIntAttribute("kv_num_heads", 1)[0]);
      } else {
        inputs = {"m" + suffix, "positions", "cos", "sin"};
        attributes = LayerIntAttribute("num_heads", 1);
      }
      if (op != "RotaryEmbedding")
        outputs.push_back("state" + suffix);
      builder.MakeNode(op, inputs, outputs, "com.microsoft", "contrib" + suffix, attributes);
      builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({1, 2, 4}));
      if (op != "RotaryEmbedding")
        builder.MakeOutput("state" + suffix, core::symbolic::TensorType::kFloat,
                           op == "CausalConvWithState" ? LayerShape({1, 2, 2})
                                                       : LayerShape({1, 1, 4, 4}));
    }
    const GraphProto before = builder.BuildGraph();
    EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
    EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 2u);
    ExpectOriginalOperators(before, builder.BuildGraph());
    ExpectCoherentLayerMetadata(builder);
    EXPECT_NO_THROW(VerifyModel(builder.ToModel()));
  }
}

TEST(RepeatedLayerPattern, ExtractsArbitraryDomainsAndStochasticOperatorsWithoutReordering) {
  for (const auto &[domain, op] :
       std::vector<std::pair<std::string, std::string>>{{"other", "Gelu"},
                                                        {"com.microsoft", "Opaque"},
                                                        {"com.microsoft", "Dropout"},
                                                        {"", "Dropout"},
                                                        {"", "RandomNormalLike"}}) {
    SCOPED_TRACE(domain + "/" + op);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(),
                                        {{"", 24}, {"com.microsoft", 1}, {"other", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    for (int index = 0; index < 2; ++index) {
      const std::string suffix = std::to_string(index);
      AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
      builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
      builder.MakeNode(op, {"m" + suffix}, {"y" + suffix}, domain, "custom" + suffix);
      builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    }
    const GraphProto before = builder.BuildGraph();
    EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
    EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 2u);
    ExpectOriginalOperators(before, builder.BuildGraph());
    ExpectCoherentLayerMetadata(builder);
    EXPECT_NO_THROW(VerifyModel(builder.ToModel()));
    EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  }
}

TEST(RepeatedLayerPattern, UsesBuilderLocalShapeInferenceRegardlessOfDomain) {
  const std::string domain = "test.layer.custom";
  int calls = 0;
  core::builder::OperatorRegistry registry;
  registry.RegisterShapeInference(
      domain, "CustomProjection",
      [&calls](core::shapes::ShapesContext &ctx, const NodeProto &node) {
        ++calls;
        ctx.CopyValueFrom(node.output(0), ctx, node.input(0));
      });
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {domain, 1}}, registry);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 2; ++index) {
    const std::string suffix = std::to_string(index);
    AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
    builder.MakeNode("CustomProjection", {"m" + suffix}, {"custom" + suffix}, domain,
                     "custom_node" + suffix);
    ASSERT_TRUE(builder.HasShape("custom" + suffix));
    builder.MakeNode("Relu", {"custom" + suffix}, {"y" + suffix}, "", "relu" + suffix);
    builder.MakeOutput("y" + suffix);
  }
  const GraphProto before = builder.BuildGraph();
  const auto rewrites = OptimizeLayers(builder);
  ASSERT_FALSE(rewrites.rewrites().empty());
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 2u);
  ExpectOriginalOperators(before, builder.BuildGraph());
  ExpectCoherentLayerMetadata(builder);
  EXPECT_NO_THROW(VerifyModel(builder.ToModel()));
  core::builder::GraphBuilder reimported(builder.ToModel(), LayerSchemaLookup(), registry);
  EXPECT_EQ(reimported.GetShape("y0").Shape(), LayerShape({2, 4}));
  const int before_undo = calls;
  auto restored = builder.UndoRewritings(rewrites);
  EXPECT_GT(calls, before_undo);
  EXPECT_EQ(restored.GetShape("custom0").Shape(), LayerShape({2, 4}));
  const int before_replay = calls;
  auto replayed = restored.ApplyRewritings(rewrites);
  EXPECT_GT(calls, before_replay);
  EXPECT_EQ(replayed.GetShape("y0").Shape(), LayerShape({2, 4}));
  ExpectCoherentLayerMetadata(replayed);
  core::builder::GraphBuilder isolated("isolated", LayerSchemaLookup(), {{domain, 1}});
  isolated.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  isolated.MakeNode("CustomProjection", {"x"}, {"y"}, domain);
  EXPECT_FALSE(isolated.HasShape("y"));
}

TEST(RepeatedLayerPattern, RejectsAnnotationsUnsupportedByInferenceFallback) {
  for (bool map_output : {false, true}) {
    SCOPED_TRACE(map_output);
    const std::string domain = map_output ? "test.layer.opaque" : "ai.rt";
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {domain, 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    for (int index = 0; index < 2; ++index) {
      const std::string suffix = std::to_string(index);
      AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
      builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
      builder.MakeNode("Opaque", {"m" + suffix}, {"y" + suffix}, domain, "opaque" + suffix);
      if (map_output) {
        ValueInfoProto output;
        output.set_name("y" + suffix);
        auto *type = output.mutable_type()->mutable_map_type();
        type->set_key_type(TensorProto::INT64);
        *type->mutable_value_type() = builder.Inputs()[0].type();
        builder.MakeOutput(output);
      } else {
        builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
      }
    }
    core::builder::GraphGraph graph(builder);
    ASSERT_TRUE(graph.HasType("y0"));
    const GraphProto before = builder.BuildGraph();
    EXPECT_NO_THROW({
      EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
      EXPECT_TRUE(before.Equals(builder.BuildGraph()));
      EXPECT_TRUE(builder.LocalFunctions().empty());
    });
  }
}

TEST(RepeatedLayerPattern, RejectsContribWithoutInternalOutputShapes) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}, {"com.microsoft", 1}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 2; ++index) {
    const std::string suffix = std::to_string(index);
    AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
    builder.MakeNode("Gelu", {"m" + suffix}, {"g" + suffix}, "com.microsoft", "gelu" + suffix);
    builder.MakeNode("Relu", {"g" + suffix}, {"y" + suffix}, "", "relu" + suffix);
    builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  }
  const GraphProto before = builder.BuildGraph();
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
}

TEST(RepeatedLayerPattern, ExtractsOpaqueSideCallsWithoutChangingExecutionOrder) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(),
                                      {{"", 24}, {"com.microsoft", 1}, {"other", 1}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 2; ++index) {
    const std::string suffix = std::to_string(index);
    AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
    builder.MakeNode("Opaque", {"m" + suffix}, {"side" + suffix}, "other", "opaque" + suffix);
    builder.MakeNode("Gelu", {"m" + suffix}, {"y" + suffix}, "com.microsoft", "gelu" + suffix);
    builder.MakeOutput("y" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    builder.MakeOutput("side" + suffix, core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  }
  const GraphProto before = builder.BuildGraph();
  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 2u);
  ExpectOriginalOperators(before, builder.BuildGraph());
  EXPECT_EQ(builder.Nodes().size(), 2u);
}

TEST(RepeatedLayerPattern, RejectsContractionThatWouldReorderIndependentCalls) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 24}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 2; ++index) {
    const std::string suffix = std::to_string(index);
    AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {"x", "w" + suffix}, {"m" + suffix}, "", "matmul" + suffix);
    builder.MakeNode(index == 0 ? "Neg" : "Abs", {"x"}, {"side" + suffix}, "", "side" + suffix);
    builder.MakeNode("Relu", {"m" + suffix}, {"y" + suffix}, "", "relu" + suffix);
    builder.MakeOutput("y" + suffix);
    builder.MakeOutput("side" + suffix);
  }
  const GraphProto before = builder.BuildGraph();
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(before.Equals(builder.BuildGraph()));
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

TEST(RepeatedLayerPattern, PreservesAttributesBoundByRetainedFunctionCalls) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(),
                                      {{"", 28}, {"activation.local", 1}});
  FunctionProto activation;
  activation.set_name("Activation");
  activation.set_domain("activation.local");
  activation.add_opset("", 28);
  activation.add_input("value");
  activation.add_output("result");
  activation.add_attribute("alpha");
  auto leaky_relu = MakeNode("LeakyRelu", {"value"}, {"result"});
  auto &reference = *leaky_relu.add_attribute();
  reference.set_name("alpha");
  reference.set_ref_attr_name("alpha");
  reference.set_type(AttributeProto::FLOAT);
  activation.add_node(std::move(leaky_relu));
  builder.MakeLocalFunction(activation);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  for (int index = 0; index < 2; ++index) {
    const std::string suffix = std::to_string(index);
    AddLayerWeight(builder, "w" + suffix, 4, 4, static_cast<float>(index + 1));
    builder.MakeNode("MatMul", {index == 0 ? "x" : "h0", "w" + suffix}, {"m" + suffix});
    utils::RepeatedProtoField<AttributeProto> arguments;
    auto &alpha = arguments.add();
    alpha.set_name("alpha");
    alpha.set_type(AttributeProto::FLOAT);
    alpha.set_f(0.3f);
    builder.MakeNode("Activation", {"m" + suffix}, {"h" + suffix}, "activation.local", "",
                     arguments);
  }
  builder.MakeOutput("h1");
  EXPECT_FALSE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_EQ(ExpectLayerContracts(builder.BuildGraph()), 2u);
  EXPECT_NO_THROW(VerifyModel(builder.ToModel()));
}

TEST(RepeatedLayerPattern, RejectsEnclosingAttributeReferencesIncludingNestedGraphs) {
  for (bool nested : {false, true}) {
    SCOPED_TRACE(nested);
    core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}, {"source.local", 1}});
    auto &source = builder.MakeLocalFunction("Source", "source.local");
    source.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    source.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, LayerShape({}));
    for (int index = 0; index < 2; ++index) {
      const std::string suffix = std::to_string(index);
      source.MakeInput("w" + suffix, core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
      builder.MakeInput("w" + suffix, core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
      const std::string input = index == 0 ? "x" : "h0";
      source.MakeNode("MatMul", {input, "w" + suffix}, {"m" + suffix});
      utils::RepeatedProtoField<AttributeProto> attributes;
      auto &alpha = attributes.add();
      alpha.set_name("alpha");
      alpha.set_type(AttributeProto::FLOAT);
      alpha.set_f(0.3f);
      if (nested) {
        GraphProto branch;
        branch.set_name("branch");
        NodeProto activation = MakeNode("LeakyRelu", {"m" + suffix}, {"result"});
        activation.ref_attribute() = attributes;
        branch.add_node(std::move(activation));
        branch.add_output()->set_name("result");
        source.MakeNode("If", {"condition"}, {"h" + suffix}, "", "", LayerBranches(branch));
      } else {
        source.MakeNode("LeakyRelu", {"m" + suffix}, {"h" + suffix}, "", "", attributes);
      }
    }
    source.MakeOutput("h1");
    utils::RepeatedProtoField<AttributeProto> arguments;
    auto &slope = arguments.add();
    slope.set_name("slope");
    slope.set_type(AttributeProto::FLOAT);
    slope.set_f(0.3f);
    builder.MakeNode("Source", {"x", "condition", "w0", "w1"}, {"y"}, "source.local");
    builder.MakeOutput("y");
    ModelProto model = builder.ToModel();
    auto &function = model.ref_functions()[0];
    function.add_attribute("slope");
    const auto reference_alpha = [&](const auto &self, NodeProto &node) -> void {
      for (auto &attribute : node.ref_attribute()) {
        if (node.op_type() == "LeakyRelu" && attribute.name() == "alpha") {
          attribute.clear_f();
          attribute.set_ref_attr_name("slope");
        }
        if (attribute.has_g())
          for (auto &child : attribute.mutable_g()->ref_node())
            self(self, child);
        for (auto &graph : attribute.ref_graphs())
          for (auto &child : graph.ref_node())
            self(self, child);
      }
    };
    for (auto &node : function.ref_node())
      reference_alpha(reference_alpha, node);
    model.mutable_graph()->mutable_node(0)->ref_attribute() = arguments;
    core::builder::GraphBuilder imported(model, LayerSchemaLookup());
    auto &imported_source = imported.LocalFunction("Source");
    const GraphProto before = imported_source.BuildGraph();
    EXPECT_TRUE(OptimizeLayers(imported_source).rewrites().empty());
    EXPECT_TRUE(before.Equals(imported_source.BuildGraph()));
  }
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

TEST(RepeatedLayerPattern, LeavesPreexistingLayerFunctionCallsUnwrapped) {
  core::builder::GraphBuilder builder("g", LayerSchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  AddLayerWeight(builder, "w", 4, 4);
  builder.SetOpsetVersion("ai.rt", 1);
  auto &function = builder.MakeLocalFunction("Layer", "ai.rt");
  function.MakeInput("activation", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  function.MakeInput("weight", core::symbolic::TensorType::kFloat, LayerShape({4, 4}));
  AddMatMulLayer(function, "activation", "weight", "h0");
  AddMatMulLayer(function, "h0", "weight", "h1");
  AddMatMulLayer(function, "h1", "weight", "h2");
  function.MakeOutput("h2", core::symbolic::TensorType::kFloat, LayerShape({2, 4}));
  builder.MakeNode("Layer", {"x", "w"}, {"result"}, "ai.rt");
  builder.MakeOutput("result");
  const GraphProto before = builder.BuildGraph();

  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  const GraphProto after = builder.BuildGraph();
  EXPECT_EQ(ExpectLayerContracts(after), 1u);
  EXPECT_TRUE(before.Equals(after));
  EXPECT_TRUE(OptimizeLayers(builder).rewrites().empty());
  EXPECT_TRUE(after.Equals(builder.BuildGraph()));
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
