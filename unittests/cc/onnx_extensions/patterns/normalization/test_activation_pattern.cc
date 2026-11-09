// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/patterns/normalization/activation_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
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

core::symbolic::SymShape Shape(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

utils::RepeatedProtoField<AttributeProto>
IntAttributes(std::initializer_list<std::pair<const char *, int64_t>> values) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &[name, value] : values) {
    AttributeProto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::INT);
    attribute.set_i(value);
  }
  return attributes;
}

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType type) {
  builder.MakeNode("Cast", {input}, {output}, "", "",
                   IntAttributes({{"to", static_cast<int64_t>(type)}}));
}

void AddFloatInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                         const std::vector<int64_t> &dims, const std::vector<float> &values) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dims, values));
}

void AddDoubleInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                          double value) {
  builder.MakeInitializer(MakeInitializer<double>(name.c_str(), {1}, {value}));
}

void AddFloat16Initializer(core::builder::GraphBuilder &builder, const std::string &name,
                           float value) {
  builder.MakeInitializer(
      MakeInitializer<uint16_t>(name.c_str(), {1}, {core::runtime::FloatToFloat16Bits(value)}));
}

void AddSubgraphReference(core::builder::GraphBuilder &builder, const std::string &subgraph_name,
                          const std::string &output) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("body_ref");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s(subgraph_name);
  builder.MakeNode("SubgraphCarrier", {}, {output}, "", "", attributes);
}

void SetNodeDomain(core::builder::GraphBuilder &builder, std::size_t index,
                   const std::string &domain) {
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[index].set_domain(domain);
}

void SetValueType(core::builder::GraphBuilder &builder, const std::string &name,
                  core::symbolic::TensorType type, const core::symbolic::SymShape &shape) {
  auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  shapes.Set(name, core::symbolic::SymTensor(nullptr, type, shape));
}

void MakeExactGelu(core::builder::GraphBuilder &builder,
                   core::symbolic::TensorType type = core::symbolic::TensorType::kFloat,
                   int half_placement = 0, bool reciprocal = false, bool divide_half = false,
                   unsigned swapped = 0, const std::vector<int64_t> &scalar_shape = {1}) {
  builder.MakeInput("x", type, Shape({2, 3}));
  const auto dtype = static_cast<TensorProto::DataType>(core::symbolic::TensorTypeToDataType(type));
  for (const auto &[name, value] : std::vector<std::pair<std::string, double>>{
           {"sqrt_two", reciprocal ? 1.0 / std::sqrt(2.0) : std::sqrt(2.0)},
           {"one", 1.0},
           {"half", divide_half ? 2.0 : 0.5}})
    builder.MakeInitializer(
        onnx_patterns::collections::MakeScalarTensor(name, dtype, scalar_shape, value));
  const auto binary = [&](const char *op, std::string left, std::string right,
                          const std::string &output, unsigned flag) {
    if ((swapped & flag) != 0 && std::string(op) != "Div")
      std::swap(left, right);
    builder.MakeNode(op, {left, right}, {output}, "", output + "_node");
  };
  binary(reciprocal ? "Mul" : "Div", "x", "sqrt_two", "scaled", 1);
  builder.MakeNode("Erf", {"scaled"}, {"erf"});
  binary("Add", "erf", "one", "term", 2);
  if (half_placement == 2) {
    binary("Mul", "x", "term", "product", 4);
    binary(divide_half ? "Div" : "Mul", "product", "half", "y", 8);
  } else {
    binary(divide_half ? "Div" : "Mul", half_placement == 0 ? "x" : "term", "half", "halved", 8);
    binary("Mul", half_placement == 0 ? "halved" : "x", half_placement == 0 ? "term" : "halved",
           "y", 4);
  }
  builder.MakeOutput("y");
}

RewritingProto OptimizeGelu(core::builder::GraphBuilder &builder) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(onnx_patterns::CreatePattern("Gelu"));
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::ConstantFoldingOptions folding;
  folding.enabled = false;
  return graph.Optimize(4, nullptr, folding);
}

TEST(GeluPattern, OptimizesExactFormsAndCommutativeInputsInSupportedTypes) {
  for (const auto type : {core::symbolic::TensorType::kFloat, core::symbolic::TensorType::kFloat16,
                          core::symbolic::TensorType::kDouble}) {
    for (int placement : {0, 1}) {
      for (bool reciprocal : {false, true}) {
        for (bool divide_half : {false, true}) {
          for (unsigned swapped : {0u, 1u, 2u, 4u, 8u, 15u}) {
            SCOPED_TRACE(testing::Message() << static_cast<int>(type) << ":" << placement << ":"
                                            << reciprocal << ":" << divide_half << ":" << swapped);
            core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
            MakeExactGelu(builder, type, placement, reciprocal, divide_half, swapped,
                          swapped == 0 ? std::vector<int64_t>{} : std::vector<int64_t>{1});
            const auto plan = OptimizeGelu(builder);
            ASSERT_FALSE(plan.rewrites().empty());
            EXPECT_EQ(plan.rewrites(0).pattern_name(), "Gelu");
            ASSERT_EQ(builder.Nodes().size(), 1u);
            const auto &gelu = builder.Nodes()[0];
            EXPECT_EQ(gelu.op_type(), "Gelu");
            EXPECT_TRUE(gelu.domain().empty());
            EXPECT_EQ(gelu.input(0), "x");
            EXPECT_EQ(gelu.output(0), "y");
            EXPECT_EQ(GetAttributeOr<std::string>(gelu, "approximate", ""), "none");
            EXPECT_EQ(builder.GetShape("y").Dtype(), type);
            EXPECT_EQ(builder.GetShape("y").Shape(), Shape({2, 3}));
            EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
          }
        }
      }
    }
  }
}

TEST(GeluPattern, ExactRejectsFinalHalfScalingWithoutMutation) {
  for (const auto type : {core::symbolic::TensorType::kFloat, core::symbolic::TensorType::kFloat16,
                          core::symbolic::TensorType::kDouble}) {
    for (bool reciprocal : {false, true}) {
      for (bool divide_half : {false, true}) {
        for (unsigned swapped : {0u, 8u, 15u}) {
          SCOPED_TRACE(testing::Message() << static_cast<int>(type) << ":" << reciprocal << ":"
                                          << divide_half << ":" << swapped);
          core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
          MakeExactGelu(builder, type, 2, reciprocal, divide_half, swapped);
          const auto before = builder.ToModel();
          EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
          EXPECT_TRUE(builder.ToModel().Equals(before));
        }
      }
    }
  }
}

TEST(GeluPattern, ExactRequiresKnownOpset20EvenWithLowerConfiguredMinimum) {
  {
    core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
    MakeExactGelu(builder);
    builder.SetOpsetVersion("", 19);
    const auto before = builder.ToModel();
    core::builder::GraphGraph graph(builder);
    onnx_patterns::GeluPattern pattern(0, 1);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[4]).pattern, nullptr);
    EXPECT_TRUE(builder.ToModel().Equals(before));
  }
  core::builder::GraphBuilder unknown("unknown", SchemaLookup(), {{"", 20}});
  MakeExactGelu(unknown);
  const_cast<std::unordered_map<std::string, int> &>(unknown.OpsetVersions()).clear();
  EXPECT_EQ(unknown.OpsetVersion(""), core::shapes::kUnknownOpsetVersion);
  core::builder::GraphGraph unknown_graph(unknown);
  onnx_patterns::GeluPattern pattern(0, 1);
  EXPECT_EQ(pattern.Match(unknown_graph, unknown.Nodes()[4]).pattern, nullptr);
  core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
  MakeExactGelu(builder);
  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern contrib(0, 20, "com.microsoft");
  EXPECT_EQ(contrib.Match(graph, builder.Nodes()[4]).pattern, nullptr);
}

TEST(GeluPattern, ExactRejectsIncorrectConstantsOperandsAndTypesWithoutMutation) {
  for (const std::string variant :
       {"sqrt", "one", "half", "reciprocal", "division_order", "half_division_order", "other_input",
        "missing_type", "missing_shape", "wrong_type", "bfloat16"}) {
    SCOPED_TRACE(variant);
    core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
    MakeExactGelu(builder,
                  variant == "bfloat16" ? core::symbolic::TensorType::kBfloat16
                                        : core::symbolic::TensorType::kFloat,
                  0, variant == "reciprocal", variant == "half_division_order");
    auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
    if (variant == "sqrt" || variant == "one" || variant == "half" || variant == "reciprocal") {
      const std::string name = variant == "sqrt" || variant == "reciprocal" ? "sqrt_two" : variant;
      auto &initializers =
          const_cast<utils::RepeatedProtoField<TensorProto> &>(builder.Initializers());
      for (auto &initializer : initializers)
        if (initializer.name().value() == name)
          initializer =
              MakeInitializer<float>(name.c_str(), {1}, {variant == "half" ? 0.5001f : 1.001f});
    } else if (variant == "division_order" || variant == "half_division_order") {
      const int index = variant == "division_order" ? 0 : 3;
      const std::string left = nodes[index].input(0).value();
      *nodes[index].mutable_input(0) = nodes[index].input(1).value();
      *nodes[index].mutable_input(1) = left;
    } else if (variant == "other_input") {
      builder.MakeInput("other", core::symbolic::TensorType::kFloat, Shape({2, 3}));
      *nodes[3].mutable_input(0) = "other";
    } else if (variant == "wrong_type") {
      SetValueType(builder, "one", core::symbolic::TensorType::kDouble, Shape({1}));
    } else if (variant == "missing_type" || variant == "missing_shape") {
      auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
      if (variant == "missing_shape")
        shapes.Remove("x");
      else
        shapes.Set("x", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kUndefined,
                                                  Shape({2, 3})));
    }
    const auto before = builder.ToModel();
    EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
    EXPECT_TRUE(builder.ToModel().Equals(before));
  }
}

TEST(GeluPattern, ExactRejectsRankExpandingScalarBroadcastWithoutMutation) {
  core::builder::GraphBuilder builder("scalar", SchemaLookup(), {{"", 20}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({}));
  AddFloatInitializer(builder, "sqrt_two", {1}, {std::sqrt(2.0f)});
  AddFloatInitializer(builder, "one", {1}, {1.0f});
  AddFloatInitializer(builder, "half", {1}, {0.5f});
  builder.MakeNode("Div", {"x", "sqrt_two"}, {"scaled"});
  builder.MakeNode("Erf", {"scaled"}, {"erf"});
  builder.MakeNode("Add", {"erf", "one"}, {"term"});
  builder.MakeNode("Mul", {"x", "half"}, {"halved"});
  builder.MakeNode("Mul", {"halved", "term"}, {"y"});
  builder.MakeOutput("y");
  const auto before = builder.ToModel();
  EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
  EXPECT_TRUE(builder.ToModel().Equals(before));
}

TEST(GeluPattern, ExactRejectsObservableIntermediatesAndUnsupportedNodeMetadata) {
  for (int index = 0; index < 5; ++index) {
    for (const std::string variant : {"domain", "overload", "attribute", "device", "graph_output",
                                      "shared", "capture", "metadata"}) {
      if (index == 4 && (variant == "graph_output" || variant == "shared" || variant == "capture" ||
                         variant == "metadata"))
        continue;
      SCOPED_TRACE(std::to_string(index) + ":" + variant);
      core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
      MakeExactGelu(builder);
      auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
      const std::string output = nodes[index].output(0).value();
      if (variant == "domain") {
        nodes[index].set_domain("custom");
      } else if (variant == "overload") {
        nodes[index].set_overload("custom");
      } else if (variant == "attribute") {
        nodes[index].ref_attribute() = IntAttributes({{"unknown", 1}});
      } else if (variant == "device") {
        nodes[index].add_device_configurations();
      } else if (variant == "metadata") {
        nodes[index].add_metadata("annotation", "value");
      } else if (variant == "graph_output") {
        builder.MakeOutput(output);
      } else if (variant == "shared") {
        builder.MakeNode("Identity", {output}, {"other"});
        builder.MakeOutput("other");
      } else {
        auto &body = builder.MakeSubgraph("body");
        body.MakeNode("Identity", {output}, {"captured"});
        body.MakeOutput("captured");
        AddSubgraphReference(builder, "body", "carrier");
      }
      const auto before = builder.ToModel();
      core::builder::GraphGraph graph(builder);
      onnx_patterns::GeluPattern pattern;
      EXPECT_EQ(pattern.Match(graph, nodes[4]).pattern, nullptr);
      EXPECT_THROW(pattern.Apply(graph, {&nodes[0], &nodes[1], &nodes[2], &nodes[3], &nodes[4]}),
                   core::builder::BuilderError);
      EXPECT_TRUE(builder.ToModel().Equals(before));
    }
  }
}

TEST(GeluPattern, ExactRefusesOverridableConstantsAndInconsistentDirectCalls) {
  for (const char *name : {"sqrt_two", "one", "half"}) {
    core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
    MakeExactGelu(builder);
    ModelProto model = builder.ToModel();
    auto &input = *model.mutable_graph()->add_input();
    input.set_name(name);
    auto &tensor = *input.mutable_type()->mutable_tensor_type();
    tensor.set_elem_type(TensorProto::FLOAT);
    tensor.mutable_shape()->add_dim()->set_dim_value(1);
    core::builder::GraphBuilder overridable(model, SchemaLookup());
    const auto before = overridable.ToModel();
    EXPECT_TRUE(OptimizeGelu(overridable).rewrites().empty());
    EXPECT_TRUE(overridable.ToModel().Equals(before));
  }
  core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
  MakeExactGelu(builder);
  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  auto match = pattern.Match(graph, builder.Nodes()[4]);
  ASSERT_EQ(match.pattern, &pattern);
  std::swap(match.nodes[0], match.nodes[1]);
  EXPECT_THROW(pattern.Apply(graph, match.nodes), core::builder::BuilderError);
  match.nodes[0] = nullptr;
  EXPECT_THROW(pattern.Apply(graph, match.nodes), core::builder::BuilderError);
}

TEST(GeluPattern, ExactRejectsMalformedScalarPayloadsWithoutMutation) {
  for (const auto type : {core::symbolic::TensorType::kFloat, core::symbolic::TensorType::kFloat16,
                          core::symbolic::TensorType::kDouble}) {
    for (bool truncated : {false, true}) {
      core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
      MakeExactGelu(builder, type);
      auto &initializers =
          const_cast<utils::RepeatedProtoField<TensorProto> &>(builder.Initializers());
      auto &bytes = initializers[0].ref_raw_data();
      bytes.resize(truncated ? bytes.size() - 1 : bytes.size() + 1);
      const auto before = builder.ToModel();
      EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
      EXPECT_TRUE(builder.ToModel().Equals(before));
    }
  }
  core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
  MakeExactGelu(builder);
  auto &initializers = const_cast<utils::RepeatedProtoField<TensorProto> &>(builder.Initializers());
  initializers[0] = MakeInitializer<float>("sqrt_two", {1}, {std::sqrt(2.0f), std::sqrt(2.0f)});
  const auto before = builder.ToModel();
  EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
  EXPECT_TRUE(builder.ToModel().Equals(before));
}

TEST(GeluPattern, ExactAcceptsTensorConstantsAndOrtHistoricalSqrtTwo) {
  for (const auto type : {core::symbolic::TensorType::kFloat, core::symbolic::TensorType::kFloat16,
                          core::symbolic::TensorType::kDouble}) {
    core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
    MakeExactGelu(builder, type);
    ModelProto model = builder.ToModel();
    auto &graph = *model.mutable_graph();
    const auto original_nodes = graph.node();
    graph.clr_node();
    for (const auto &initializer : graph.initializer()) {
      auto &node = *graph.add_node();
      node.set_op_type("Constant");
      node.add_output(initializer.name());
      auto &attribute = *node.add_attribute();
      attribute.set_name("value");
      attribute.set_type(AttributeProto::TENSOR);
      *attribute.mutable_t() = initializer;
    }
    for (const auto &node : original_nodes) {
      auto &added = *graph.add_node();
      added = node;
      added.ref_metadata_props().clear();
    }
    graph.clr_initializer();
    core::builder::GraphBuilder constants(model, SchemaLookup());
    EXPECT_FALSE(OptimizeGelu(constants).rewrites().empty());
    ASSERT_EQ(constants.Nodes().size(), 1u);
    EXPECT_EQ(constants.Nodes()[0].op_type(), "Gelu");
    EXPECT_EQ(GetAttributeOr<std::string>(constants.Nodes()[0], "approximate", ""), "none");
  }
  for (float value : {std::sqrt(2.0f), 1.4142099618911743f}) {
    core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
    MakeExactGelu(builder);
    auto &initializers =
        const_cast<utils::RepeatedProtoField<TensorProto> &>(builder.Initializers());
    initializers[0] = MakeInitializer<float>("sqrt_two", {1}, {value});
    EXPECT_FALSE(OptimizeGelu(builder).rewrites().empty());
    ASSERT_EQ(builder.Nodes().size(), 1u);
    EXPECT_EQ(builder.Nodes()[0].op_type(), "Gelu");
  }
}

TEST(GeluPattern, ExactRejectsPromotedFloatConstantsInDoubleGraphs) {
  for (const double value : {static_cast<double>(std::sqrt(2.0f)), 1.4142099618911743,
                             static_cast<double>(1.0f / std::sqrt(2.0f))}) {
    SCOPED_TRACE(value);
    const bool reciprocal = value < 1.0;
    core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
    MakeExactGelu(builder, core::symbolic::TensorType::kDouble, 0, reciprocal);
    auto &initializers =
        const_cast<utils::RepeatedProtoField<TensorProto> &>(builder.Initializers());
    initializers[0] = MakeInitializer<double>("sqrt_two", {1}, {value});
    const auto before = builder.ToModel();
    EXPECT_TRUE(OptimizeGelu(builder).rewrites().empty());
    EXPECT_TRUE(builder.ToModel().Equals(before));
  }
}

TEST(GeluPattern, ExactPreservesFinalMetadataSharedOutputsAndSymbolicShapes) {
  core::builder::GraphBuilder builder("exact", SchemaLookup(), {{"", 20}});
  MakeExactGelu(builder);
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[4].add_metadata("annotation", "value");
  nodes[4].set_doc_string("exact activation");
  core::symbolic::SymShape symbolic;
  symbolic.PushBack(core::symbolic::SymDim("batch"));
  symbolic.PushBack(core::symbolic::SymDim("features"));
  for (const auto *name : {"x", "scaled", "erf", "term", "halved", "y"})
    SetValueType(builder, name, core::symbolic::TensorType::kFloat, symbolic);
  builder.MakeNode("Identity", {"y"}, {"other"});
  builder.MakeOutput("other");
  builder.MakeOutput("x");
  const auto plan = OptimizeGelu(builder);
  ASSERT_FALSE(plan.rewrites().empty());
  EXPECT_EQ(plan.rewrites(0).pattern_name(), "Gelu");
  ASSERT_EQ(builder.Nodes().size(), 2u);
  const auto &gelu = builder.Nodes()[0];
  EXPECT_EQ(gelu.op_type(), "Gelu");
  EXPECT_EQ(gelu.name(), "GeluPattern--y_node");
  EXPECT_EQ(gelu.doc_string(), "exact activation");
  ASSERT_EQ(gelu.metadata_props().size(), 1u);
  EXPECT_EQ(gelu.metadata_props()[0].key(), "annotation");
  EXPECT_EQ(gelu.metadata_props()[0].value(), "value");
  EXPECT_EQ(builder.GetShape("y").Shape(), symbolic);
  EXPECT_EQ(builder.Nodes()[1].input(0), "y");
}

TEST(GeluPattern, FusesExactTanhDecomposition) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.7978515625);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"}, "", "gelu_decomposition");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern(0, 20, "com.microsoft");
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 8u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[7], &builder.Nodes()[7]);
  EXPECT_EQ(match.insert_at, nullptr);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gelu");
  EXPECT_EQ(replacements[0].domain().value(), "com.microsoft");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<std::string>(replacements[0], "approximate", ""), "tanh");
}

TEST(GeluPattern, RejectsWrongTanhScale) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.8);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "one Gelu scalar constant has an unexpected value");
}

TEST(GeluPattern, RejectsIntermediateGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.7978515625);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"});
  builder.MakeOutput("x3");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an intermediate Gelu result is externally used");
}

TEST(GeluPattern, RejectsNonOnnxIntermediateDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.SetOpsetVersion("custom", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape({2, 3}));
  AddDoubleInitializer(builder, "three", 3.0);
  AddDoubleInitializer(builder, "cubic_scale", 0.044715);
  AddDoubleInitializer(builder, "sqrt_two_over_pi", 0.7978515625);
  AddDoubleInitializer(builder, "one", 1.0);
  AddDoubleInitializer(builder, "half", 0.5);
  builder.MakeNode("Pow", {"x", "three"}, {"x3"});
  builder.MakeNode("Mul", {"x3", "cubic_scale"}, {"scaled_x3"});
  builder.MakeNode("Add", {"x", "scaled_x3"}, {"polynomial"});
  builder.MakeNode("Mul", {"polynomial", "sqrt_two_over_pi"}, {"scaled_polynomial"});
  builder.MakeNode("Tanh", {"scaled_polynomial"}, {"tanh"});
  builder.MakeNode("Add", {"tanh", "one"}, {"tanh_one"});
  builder.MakeNode("Mul", {"x", "half"}, {"x_half"});
  builder.MakeNode("Mul", {"x_half", "tanh_one"}, {"y"});
  builder.MakeOutput("y");
  SetNodeDomain(builder, 4, "custom");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::GeluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the tanh-based Gelu topology is incomplete");
}

TEST(LeakyReluPattern, FusesWhereDecomposition) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 3}));
  AddFloatInitializer(builder, "zero", {1}, {0.0F});
  AddFloatInitializer(builder, "slope", {1}, {-0.33F});
  builder.MakeNode("Greater", {"x", "zero"}, {"positive"});
  builder.MakeNode("Mul", {"x", "slope"}, {"negative"});
  builder.MakeNode("Where", {"positive", "x", "negative"}, {"y"}, "", "where");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LeakyReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 3u);
  EXPECT_EQ(match.insert_at, nullptr);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "LeakyRelu");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[0], "alpha", 0.0F), -0.33F);
}

TEST(LeakyReluPattern, RejectsNonzeroThreshold) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 3}));
  AddFloatInitializer(builder, "zero", {1}, {1.0F});
  AddFloatInitializer(builder, "slope", {1}, {-0.33F});
  builder.MakeNode("Greater", {"x", "zero"}, {"positive"});
  builder.MakeNode("Mul", {"x", "slope"}, {"negative"});
  builder.MakeNode("Where", {"positive", "x", "negative"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LeakyReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the threshold is not zero or the slope is not scalar");
}

TEST(LeakyReluPattern, RejectsSubgraphCapture) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 3}));
  AddFloatInitializer(builder, "zero", {1}, {0.0F});
  AddFloatInitializer(builder, "slope", {1}, {-0.33F});
  builder.MakeNode("Greater", {"x", "zero"}, {"positive"});
  builder.MakeNode("Mul", {"x", "slope"}, {"negative"});
  builder.MakeNode("Where", {"positive", "x", "negative"}, {"y"});
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"positive"}, {"captured"});
  body.MakeOutput("captured");
  AddSubgraphReference(builder, "body", "body_result");
  builder.MakeOutput("y");
  builder.MakeOutput("body_result");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LeakyReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "a LeakyRelu intermediate result is externally used");
}

void BuildSoftmaxCrossEntropyGraph(
    core::builder::GraphBuilder &builder, int64_t squeeze_axis,
    core::symbolic::TensorType score_type = core::symbolic::TensorType::kFloat16,
    int64_t ignored_index_branch = 0, float ignored_loss_branch = 0.0F) {
  builder.SetOpsetVersion("", 14);
  builder.MakeInput("x", score_type, Shape({3, 5}));
  builder.MakeInput("indices", core::symbolic::TensorType::kInt64, Shape({3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("ignore", {1}, {-100}));
  builder.MakeInitializer(MakeInitializer<int64_t>("zeroi", {1}, {ignored_index_branch}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axis", {1}, {squeeze_axis}));
  if (score_type == core::symbolic::TensorType::kFloat16) {
    AddFloat16Initializer(builder, "zerof", ignored_loss_branch);
  } else {
    AddFloatInitializer(builder, "zerof", {1}, {ignored_loss_branch});
  }
  builder.MakeNode("Equal", {"indices", "ignore"}, {"ignored"});
  builder.MakeNode("Not", {"ignored"}, {"kept"});
  builder.MakeNode("Where", {"kept", "indices", "zeroi"}, {"safe_indices"});
  builder.MakeNode("Unsqueeze", {"safe_indices", "axis"}, {"indices_2d"});
  builder.MakeNode("LogSoftmax", {"x"}, {"log_probabilities"}, "", "",
                   IntAttributes({{"axis", 1}}));
  builder.MakeNode("GatherElements", {"log_probabilities", "indices_2d"}, {"selected"}, "", "",
                   IntAttributes({{"axis", 1}}));
  builder.MakeNode("Squeeze", {"selected", "axis"}, {"selected_1d"});
  builder.MakeNode("Neg", {"selected_1d"}, {"losses"});
  builder.MakeNode("Where", {"kept", "losses", "zerof"}, {"masked_losses"});
  AddCast(builder, "kept", "kept_float", TensorProto::DataType::FLOAT);
  builder.MakeNode("ReduceSum", {"kept_float"}, {"denominator_float"}, "", "",
                   IntAttributes({{"keepdims", 0}}));
  AddCast(builder, "denominator_float", "denominator", TensorProto::DataType::FLOAT16);
  AddCast(builder, "masked_losses", "losses_float", TensorProto::DataType::FLOAT);
  builder.MakeNode("ReduceSum", {"losses_float"}, {"numerator_float"}, "", "",
                   IntAttributes({{"keepdims", 0}}));
  AddCast(builder, "numerator_float", "numerator", TensorProto::DataType::FLOAT16);
  builder.MakeNode("Div", {"numerator", "denominator"}, {"y"}, "", "loss");
  builder.MakeOutput("y");
}

TEST(SoftmaxCrossEntropyLossCastPattern, FusesExactMaskedMeanLoss) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  ASSERT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 16u);
  EXPECT_EQ(match.nodes[0], &builder.Nodes()[0]);
  EXPECT_EQ(match.nodes[15], &builder.Nodes()[15]);
  EXPECT_EQ(match.insert_at, nullptr);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "SoftmaxCrossEntropyLoss");
  EXPECT_EQ(replacements[0].domain().value(), "");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "indices");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "ignore_index", 0), -100);
  EXPECT_EQ(GetAttributeOr<std::string>(replacements[0], "reduction", ""), "mean");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsWrongSqueezeAxis) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 0);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the ignore index or squeeze axes constants differ");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonzeroIgnoredIndexBranch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1, core::symbolic::TensorType::kFloat16, 1);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an ignored-label Where branch is not exactly zero");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonzeroIgnoredLossBranch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1, core::symbolic::TensorType::kFloat16, 0, 0.5F);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an ignored-label Where branch is not exactly zero");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonFloat16Scores) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1, core::symbolic::TensorType::kFloat);

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the score, label, zero-branch, loss, or output types differ");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonFloat16Output) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);
  SetValueType(builder, "y", core::symbolic::TensorType::kFloat, Shape({}));

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the score, label, zero-branch, loss, or output types differ");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsNonOnnxInternalDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);
  SetNodeDomain(builder, 4, "custom");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the masked GatherElements loss topology is incomplete");
}

TEST(SoftmaxCrossEntropyLossCastPattern, RejectsIntermediateGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  BuildSoftmaxCrossEntropyGraph(builder, 1);
  builder.MakeOutput("selected");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SoftmaxCrossEntropyLossCastPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[15]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "an intermediate loss result is externally used");
}

TEST(MaxReluPattern, ReplacesSymmetricZeroMaximum) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInput("x", core::symbolic::TensorType::kInt32, Shape({2, 3}));
  builder.MakeInitializer(MakeInitializer<int32_t>("zero", {1}, {0}));
  builder.MakeNode("Max", {"zero", "x"}, {"y"}, "", "maximum");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MaxReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);

  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Relu");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(replacements[0].name().value(), "MaxReluPattern--maximum");
}

TEST(MaxReluPattern, RejectsTwoZeroConstants) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 20);
  builder.MakeInitializer(MakeInitializer<int32_t>("zero0", {1}, {0}));
  builder.MakeInitializer(MakeInitializer<int32_t>("zero1", {1}, {0}));
  builder.MakeNode("Max", {"zero0", "zero1"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MaxReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "Max does not have exactly one scalar zero input");
}

TEST(MaxReluPattern, RejectsIntegerReluBeforeOpsetFourteen) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.MakeInput("x", core::symbolic::TensorType::kInt32, Shape({2, 3}));
  builder.MakeInitializer(MakeInitializer<int32_t>("zero", {1}, {0}));
  builder.MakeNode("Max", {"zero", "x"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MaxReluPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "integer Relu requires default-domain opset 14 or newer");
}

} // namespace
} // namespace Test
