// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/attention/attention_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
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

core::symbolic::SymShape Shape(std::initializer_list<int64_t> dimensions) {
  std::vector<core::symbolic::SymDim> symbolic;
  for (int64_t dimension : dimensions) {
    symbolic.emplace_back(dimension);
  }
  return core::symbolic::SymShape(symbolic);
}

utils::RepeatedProtoField<AttributeProto> IntAttr(const char *name, int64_t value) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(value);
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> PermAttr(const std::vector<int64_t> &values) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("perm");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : values) {
    attribute.ints().push_back(value);
  }
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> SplitAttrs(int64_t axis) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &num_outputs = attributes.add();
  num_outputs.set_name("num_outputs");
  num_outputs.set_type(AttributeProto::AttributeType::INT);
  num_outputs.set_i(2);
  AttributeProto &axis_attribute = attributes.add();
  axis_attribute.set_name("axis");
  axis_attribute.set_type(AttributeProto::AttributeType::INT);
  axis_attribute.set_i(axis);
  return attributes;
}

void AddInt64(core::builder::GraphBuilder &builder, const std::string &name,
              const std::vector<int64_t> &dimensions, const std::vector<int64_t> &values) {
  builder.MakeInitializer(MakeInitializer<int64_t>(name.c_str(), dimensions, values));
}

void AddFloat(core::builder::GraphBuilder &builder, const std::string &name,
              const std::vector<int64_t> &dimensions, const std::vector<float> &values) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dimensions, values));
}

void SeedShape(core::builder::GraphBuilder &builder, const std::string &name,
               core::symbolic::TensorType type, const core::symbolic::SymShape &shape) {
  auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  shapes.Set(name, core::symbolic::SymTensor(nullptr, type, shape));
}

bool HasFunction(const core::builder::GraphBuilder &builder, const std::string &name) {
  return builder.HasLocalFunction(name);
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

std::size_t FindNode(const core::builder::GraphBuilder &builder, const std::string &op_type) {
  for (std::size_t index = 0; index < builder.Nodes().size(); ++index) {
    if (builder.Nodes()[index].op_type().value() == op_type) {
      return index;
    }
  }
  return builder.Nodes().size();
}

void ExpectTopologicalGraph(const core::builder::GraphBuilder &builder) {
  const GraphProto graph = builder.BuildGraph();
  EXPECT_NO_THROW(VerifyGraph(graph));
  std::unordered_set<std::string> available;
  for (const ValueInfoProto &input : graph.input()) {
    available.insert(input.name().value());
  }
  for (const TensorProto &initializer : graph.initializer()) {
    available.insert(initializer.name().value());
  }
  for (const NodeProto &node : graph.node()) {
    for (const utils::String &input : node.input()) {
      if (!input.value().empty()) {
        EXPECT_NE(available.count(input.value()), 0u)
            << node.op_type().value() << " reads " << input.value() << " before its producer";
      }
    }
    for (const utils::String &output : node.output()) {
      if (!output.value().empty()) {
        available.insert(output.value());
      }
    }
  }
  for (const ValueInfoProto &output : graph.output()) {
    EXPECT_NE(available.count(output.name().value()), 0u);
  }
}

template <typename Pattern> void OptimizeAndVerify(core::builder::GraphBuilder &builder) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<Pattern>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  EXPECT_FALSE(optimizer.Optimize().empty());
  ExpectTopologicalGraph(builder);
}

core::builder::GraphBuilder
MakeCosSinBuilder(TensorProto::DataType position_cast = TensorProto::DataType::FLOAT,
                  core::symbolic::TensorType weight_type = core::symbolic::TensorType::kFloat,
                  bool late_weight = false, bool shared_reshape = false) {
  core::builder::GraphBuilder builder("cos_sin", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("A", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("B", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput(late_weight ? "weight_source" : "weights", weight_type, Shape({1, 1, 8}));
  AddInt64(builder, "one", {}, {1});
  AddInt64(builder, "axes", {2}, {0, 1});
  AddInt64(builder, "reshape_shape", {3}, {0, -1, 1});
  builder.MakeNode("Squeeze", {"A"}, {"sA"});
  builder.MakeNode("Squeeze", {"B"}, {"sB"});
  builder.MakeNode("Range", {"sA", "sB", "one"}, {"positions"});
  builder.MakeNode("Unsqueeze", {"positions", "axes"}, {"unsqueezed"});
  builder.MakeNode("Cast", {"unsqueezed"}, {"cast"}, "", "", IntAttr("to", position_cast));
  builder.MakeNode("Reshape", {"cast", "reshape_shape"}, {"reshaped"});
  if (shared_reshape) {
    builder.MakeNode("Identity", {"reshaped"}, {"earlier_use"});
  }
  if (late_weight) {
    builder.MakeNode("Identity", {"weight_source"}, {"weights"});
  }
  builder.MakeNode("Mul", {"weights", "reshaped"}, {"weighted"});
  builder.MakeNode("Cos", {"weighted"}, {"cos"});
  builder.MakeNode("Cast", {"cos"}, {"cos_cache"}, "", "",
                   IntAttr("to", TensorProto::DataType::FLOAT16));
  builder.MakeNode("Sin", {"weighted"}, {"sin"});
  builder.MakeNode("Cast", {"sin"}, {"sin_cache"}, "", "",
                   IntAttr("to", TensorProto::DataType::FLOAT16));
  builder.MakeOutput("cos_cache", core::symbolic::TensorType::kFloat16, Shape({1, 3, 8}));
  builder.MakeOutput("sin_cache", core::symbolic::TensorType::kFloat16, Shape({1, 3, 8}));
  return builder;
}

core::builder::GraphBuilder
MakeAttentionBuilder(float infinity = -std::numeric_limits<float>::infinity(), bool fused = false,
                     bool fused_trans_b = true, float fused_alpha = 1.0F, bool late_value = false,
                     bool shared_bias = false) {
  core::builder::GraphBuilder builder("attention", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.SetOpsetVersion("com.microsoft", 1);
  builder.MakeInput("query", core::symbolic::TensorType::kFloat, Shape({2, 4, 3, 8}));
  builder.MakeInput("keys", core::symbolic::TensorType::kFloat, Shape({2, 4, 5, 8}));
  builder.MakeInput("values", core::symbolic::TensorType::kFloat, Shape({2, 4, 5, 8}));
  builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 3, 5}));
  AddFloat(builder, "zero", {1}, {0.0F});
  AddFloat(builder, "infinity", {1}, {infinity});
  AddFloat(builder, "scale", {1}, {0.5F});
  builder.MakeNode("Mul", {"query", "scale"}, {"scaled_query"});
  builder.MakeNode("Mul", {"keys", "scale"}, {"scaled_keys"});
  if (fused) {
    NodeProto node =
        MakeNode("FusedMatMul", {"scaled_query", "scaled_keys"}, {"scores"}, "com.microsoft");
    if (fused_trans_b) {
      AttributeProto *trans_b = node.add_attribute();
      trans_b->set_name("transB");
      trans_b->set_type(AttributeProto::AttributeType::INT);
      trans_b->set_i(1);
    }
    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::AttributeType::FLOAT);
    alpha->set_f(fused_alpha);
    builder.ReserveName("scores");
    auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
    nodes.push_back(std::move(node));
    SeedShape(builder, "scores", core::symbolic::TensorType::kFloat, Shape({2, 4, 3, 5}));
  } else {
    builder.MakeNode("Transpose", {"scaled_keys"}, {"keys_t"}, "", "", PermAttr({0, 1, 3, 2}));
    builder.MakeNode("MatMul", {"scaled_query", "keys_t"}, {"scores"});
  }
  builder.MakeNode("Where", {"mask", "zero", "infinity"}, {"bias"});
  builder.MakeNode("Add", {"scores", "bias"}, {"masked"});
  if (shared_bias) {
    builder.MakeNode("Identity", {"bias"}, {"earlier_bias_use"});
  }
  builder.MakeNode("Softmax", {"masked"}, {"probabilities"}, "", "softmax", IntAttr("axis", -1));
  builder.MakeNode("IsNaN", {"probabilities"}, {"nan"});
  builder.MakeNode("Where", {"nan", "zero", "probabilities"}, {"filtered"});
  if (late_value) {
    builder.MakeNode("Identity", {"values"}, {"late_values"});
  }
  builder.MakeNode("MatMul", {"filtered", late_value ? "late_values" : "values"}, {"Y"});
  builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({2, 4, 3, 8}));
  return builder;
}

core::builder::GraphBuilder MakeFunctionGqaBuilder(bool late_value = false) {
  core::builder::GraphBuilder builder("gqa_function", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.SetOpsetVersion("intermediate", 1);
  builder.MakeInput("query", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 1}));
  builder.MakeInput("keys", core::symbolic::TensorType::kFloat, Shape({1, 1, 5, 1}));
  builder.MakeInput(late_value ? "values_source" : "values", core::symbolic::TensorType::kFloat,
                    Shape({1, 1, 5, 1}));
  builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 3, 5}));
  AddFloat(builder, "scale", {1}, {0.5F});
  AddInt64(builder, "two", {1}, {2});
  AddInt64(builder, "expand_shape", {5}, {1, 1, 2, 1, 1});
  AddInt64(builder, "reshape_shape", {4}, {0, 2, -1, 1});
  builder.MakeNode("Unsqueeze", {"keys", "two"}, {"keys_u"});
  builder.MakeNode("Expand", {"keys_u", "expand_shape"}, {"keys_e"});
  builder.MakeNode("Reshape", {"keys_e", "reshape_shape"}, {"keys_r"});
  if (late_value) {
    builder.MakeNode("Identity", {"values_source"}, {"values"});
  }
  builder.MakeNode("Unsqueeze", {"values", "two"}, {"values_u"});
  builder.MakeNode("Expand", {"values_u", "expand_shape"}, {"values_e"});
  builder.MakeNode("Reshape", {"values_e", "reshape_shape"}, {"values_r"});
  builder.MakeNode("LocalAttentionSW_to1", {"query", "keys_r", "values_r", "mask", "scale"}, {"Y"},
                   "intermediate", "attention");
  builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 1}));
  return builder;
}

core::builder::GraphBuilder MakeAttentionGqaBuilder(int opset = 23, int64_t cache_axis = 2,
                                                    bool active_optional = false,
                                                    bool softcap = false,
                                                    bool earlier_consumer = false) {
  core::builder::GraphBuilder builder("attention_gqa", SchemaLookup());
  builder.SetOpsetVersion("", opset);
  builder.MakeInput("query", core::symbolic::TensorType::kFloat, Shape({1, 2, 1, 1}));
  builder.MakeInput("key", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
  builder.MakeInput("value", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
  builder.MakeInput("past_key", core::symbolic::TensorType::kFloat, Shape({1, 1, 0, 1}));
  builder.MakeInput("past_value", core::symbolic::TensorType::kFloat, Shape({1, 1, 0, 1}));
  builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 1, 1}));
  AddInt64(builder, "two", {1}, {2});
  AddInt64(builder, "one", {1}, {1});
  AddInt64(builder, "expand_shape", {5}, {1, 1, 2, 1, 1});
  builder.MakeNode("Concat", {"past_key", "key"}, {"present_key"}, "", "key_concat",
                   IntAttr("axis", 2));
  builder.MakeNode("Concat", {"past_value", "value"}, {"present_value"}, "", "value_concat",
                   IntAttr("axis", 2));
  if (earlier_consumer) {
    builder.MakeNode("Identity", {"present_key"}, {"earlier_present_key"});
  }
  builder.MakeNode("Unsqueeze", {"present_key", "two"}, {"keys_u"});
  builder.MakeNode("Expand", {"keys_u", "expand_shape"}, {"keys_e"});
  builder.MakeNode("Squeeze", {"keys_e", "one"}, {"keys_r"});
  builder.MakeNode("Unsqueeze", {"present_value", "two"}, {"values_u"});
  builder.MakeNode("Expand", {"values_u", "expand_shape"}, {"values_e"});
  builder.MakeNode("Squeeze", {"values_e", "one"}, {"values_r"});
  std::vector<std::string> inputs = {"query", "keys_r", "values_r", "mask"};
  if (active_optional) {
    inputs.push_back("past_key");
  }
  NodeProto attention = MakeNode("Attention", inputs, {"Y"}, "", "attention");
  AttributeProto *scale = attention.add_attribute();
  scale->set_name("scale");
  scale->set_type(AttributeProto::AttributeType::FLOAT);
  scale->set_f(0.11F);
  if (softcap) {
    AttributeProto *attribute = attention.add_attribute();
    attribute->set_name("softcap");
    attribute->set_type(AttributeProto::AttributeType::FLOAT);
    attribute->set_f(10.0F);
  }
  builder.ReserveName("Y");
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes.push_back(std::move(attention));
  if (cache_axis != 2) {
    for (std::size_t index : {std::size_t{0}, std::size_t{1}}) {
      for (AttributeProto &attribute : nodes[index].ref_attribute()) {
        if (attribute.name().value() == "axis") {
          attribute.set_i(cache_axis);
        }
      }
    }
  }
  SeedShape(builder, "Y", core::symbolic::TensorType::kFloat, Shape({1, 2, 1, 1}));
  builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({1, 2, 1, 1}));
  builder.MakeOutput("present_key", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
  builder.MakeOutput("present_value", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
  return builder;
}

TEST(RotaryConcatPartPattern, RewritesSplitFormAndRejectsWrongZeroShape) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({3, 16}));
  AddInt64(builder, "split", {2}, {8, 8});
  AddInt64(builder, "shape", {2}, {3, 8});
  builder.MakeNode("Split", {"X", "split"}, {"x1", "x2"}, "", "split", IntAttr("axis", 1));
  builder.MakeNode("Neg", {"x1"}, {"nx1"});
  builder.MakeNode("ConstantOfShape", {"shape"}, {"zero"});
  builder.MakeNode("Concat", {"nx1", "zero"}, {"left"}, "", "", IntAttr("axis", 1));
  builder.MakeNode("Concat", {"zero", "x2"}, {"right"}, "", "", IntAttr("axis", 1));
  builder.MakeNode("Add", {"left", "right"}, {"Y"}, "", "add");
  builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({3, 16}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::RotaryConcatPartPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[5]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 4u);
  EXPECT_EQ(replacement[0].op_type().value(), "ConstantOfShape");
  EXPECT_EQ(replacement[1].op_type().value(), "Split");
  EXPECT_EQ(replacement[2].op_type().value(), "Neg");
  EXPECT_EQ(replacement[3].op_type().value(), "Concat");
  EXPECT_EQ(replacement[3].output()[0].value(), "Y");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.SetOpsetVersion("", 18);
  rejected.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({3, 16}));
  AddInt64(rejected, "split", {2}, {8, 8});
  AddInt64(rejected, "shape", {2}, {3, 16});
  rejected.MakeNode("Split", {"X", "split"}, {"x1", "x2"}, "", "", IntAttr("axis", 1));
  rejected.MakeNode("Neg", {"x1"}, {"nx1"});
  rejected.MakeNode("ConstantOfShape", {"shape"}, {"zero"});
  rejected.MakeNode("Concat", {"nx1", "zero"}, {"left"}, "", "", IntAttr("axis", 1));
  rejected.MakeNode("Concat", {"zero", "x2"}, {"right"}, "", "", IntAttr("axis", 1));
  rejected.MakeNode("Add", {"left", "right"}, {"Y"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[5]).pattern, nullptr);
}

TEST(FunctionHalfRotaryEmbeddingPattern, CreatesFunctionAndRejectsWrongAxis) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
  builder.MakeInput("cos", core::symbolic::TensorType::kFloat, Shape({6, 8}));
  builder.MakeInput("sin", core::symbolic::TensorType::kFloat, Shape({6, 8}));
  builder.MakeNode("Split", {"X"}, {"x1", "x2"}, "", "split", SplitAttrs(-1));
  builder.MakeNode("Neg", {"x2"}, {"nx2"});
  builder.MakeNode("Concat", {"nx2", "x1"}, {"rotated"}, "", "", IntAttr("axis", -1));
  builder.MakeNode("Mul", {"rotated", "sin"}, {"scaled_rotated"});
  builder.MakeNode("Mul", {"X", "cos"}, {"scaled_x"});
  builder.MakeNode("Add", {"scaled_rotated", "scaled_x"}, {"Y"});
  builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::FunctionHalfRotaryEmbeddingPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  EXPECT_EQ(match.insert_at, &builder.Nodes()[5]);
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "HalfRotaryEmbedding");
  EXPECT_EQ(replacement[0].domain().value(), "intermediate");
  EXPECT_EQ(replacement[0].input()[1].value(), "cos");
  EXPECT_EQ(replacement[0].input()[2].value(), "sin");
  ASSERT_TRUE(HasFunction(builder, "HalfRotaryEmbedding"));
  EXPECT_EQ(builder.LocalFunction("HalfRotaryEmbedding").Nodes().size(), 6u);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.SetOpsetVersion("", 18);
  rejected.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
  rejected.MakeNode("Split", {"X"}, {"x1", "x2"}, "", "", SplitAttrs(2));
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(RotaryEmbeddingPattern, EmitsOnnxRotaryEmbeddingAndRejectsOldOpset) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  builder.SetOpsetVersion("intermediate", 1);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 2, 3, 8}));
  builder.MakeInput("cos_half", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, 4}));
  builder.MakeInput("sin_half", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, 4}));
  builder.MakeNode("Concat", {"cos_half", "cos_half"}, {"cos"}, "", "", IntAttr("axis", -1));
  builder.MakeNode("Concat", {"sin_half", "sin_half"}, {"sin"}, "", "", IntAttr("axis", -1));
  builder.MakeNode("HalfRotaryEmbedding", {"X", "cos", "sin"}, {"Y"}, "intermediate", "half");
  SeedShape(builder, "Y", core::symbolic::TensorType::kFloat, Shape({2, 2, 3, 8}));
  builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({2, 2, 3, 8}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::RotaryEmbeddingPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 7u);
  const NodeProto &rotary = replacement[replacement.size() - 1];
  EXPECT_EQ(rotary.op_type().value(), "RotaryEmbedding");
  EXPECT_EQ(GetAttributeOr<int64_t>(rotary, "num_heads", 0), 2);
  EXPECT_EQ(FindAttribute(rotary, "rotary_embedding_dim"), nullptr);
  EXPECT_EQ(rotary.output()[0].value(), "Y");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.SetOpsetVersion("", 22);
  rejected.SetOpsetVersion("intermediate", 1);
  rejected.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 2, 3, 8}));
  rejected.MakeInput("cos_half", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, 4}));
  rejected.MakeInput("sin_half", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, 4}));
  rejected.MakeNode("Concat", {"cos_half", "cos_half"}, {"cos"}, "", "", IntAttr("axis", -1));
  rejected.MakeNode("Concat", {"sin_half", "sin_half"}, {"sin"}, "", "", IntAttr("axis", -1));
  rejected.MakeNode("HalfRotaryEmbedding", {"X", "cos", "sin"}, {"Y"}, "intermediate");
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[2]).pattern, nullptr);
}

TEST(FunctionCausalMaskPattern, CreatesShiftedFunctionAndRejectsWrongAxes) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("A", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("B", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("shift", core::symbolic::TensorType::kInt64, Shape({1}));
  AddInt64(builder, "zero", {}, {0});
  AddInt64(builder, "one", {}, {1});
  AddInt64(builder, "axes1", {3}, {0, 1, 2});
  AddInt64(builder, "axes2", {3}, {0, 1, 3});
  builder.MakeNode("Squeeze", {"A"}, {"sA"});
  builder.MakeNode("Squeeze", {"B"}, {"sB"});
  builder.MakeNode("Range", {"zero", "sB", "one"}, {"r1"});
  builder.MakeNode("Range", {"sA", "sB", "one"}, {"r2"});
  builder.MakeNode("Unsqueeze", {"r1", "axes1"}, {"u1"});
  builder.MakeNode("Unsqueeze", {"r2", "axes2"}, {"u2"});
  builder.MakeNode("Sub", {"u2", "shift"}, {"shifted"});
  builder.MakeNode("Greater", {"u1", "shifted"}, {"mask"}, "", "greater");
  builder.MakeOutput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 2, 3}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::FunctionCausalMaskPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "ShiftedCausalMask");
  EXPECT_EQ(replacement[0].input()[2].value(), "shift");
  EXPECT_TRUE(HasFunction(builder, "ShiftedCausalMask"));

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.SetOpsetVersion("", 18);
  rejected.MakeInput("A", core::symbolic::TensorType::kInt64, Shape({1}));
  rejected.MakeInput("B", core::symbolic::TensorType::kInt64, Shape({1}));
  AddInt64(rejected, "zero", {}, {0});
  AddInt64(rejected, "one", {}, {1});
  AddInt64(rejected, "axes1", {3}, {0, 1, 2});
  AddInt64(rejected, "axes2", {3}, {0, 1, 2});
  rejected.MakeNode("Squeeze", {"A"}, {"sA"});
  rejected.MakeNode("Squeeze", {"B"}, {"sB"});
  rejected.MakeNode("Range", {"zero", "sB", "one"}, {"r1"});
  rejected.MakeNode("Range", {"sA", "sB", "one"}, {"r2"});
  rejected.MakeNode("Unsqueeze", {"r1", "axes1"}, {"u1"});
  rejected.MakeNode("Unsqueeze", {"r2", "axes2"}, {"u2"});
  rejected.MakeNode("LessOrEqual", {"u1", "u2"}, {"mask"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[6]).pattern, nullptr);
}

TEST(FunctionCausalMaskMulAddPattern, CreatesFunctionAndRejectsNonUnitStep) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("A", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("B", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("N", core::symbolic::TensorType::kInt64, Shape({1}));
  AddInt64(builder, "zero", {}, {0});
  AddInt64(builder, "one", {}, {1});
  AddInt64(builder, "axes1", {3}, {0, 1, 2});
  AddInt64(builder, "axes2", {3}, {1, 2, 3});
  builder.MakeNode("Squeeze", {"A"}, {"sA"});
  builder.MakeNode("Squeeze", {"B"}, {"sB"});
  builder.MakeNode("Range", {"zero", "sA", "one"}, {"r1"});
  builder.MakeNode("Range", {"zero", "sB", "one"}, {"r2"});
  builder.MakeNode("Unsqueeze", {"r1", "axes1"}, {"u1"});
  builder.MakeNode("Unsqueeze", {"r2", "axes2"}, {"u2"});
  builder.MakeNode("Mul", {"u2", "N"}, {"scaled"});
  builder.MakeNode("Add", {"u1", "scaled"}, {"mask"}, "", "add");
  builder.MakeOutput("mask", core::symbolic::TensorType::kInt64, Shape({2, 1, 1, 3}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::FunctionCausalMaskMulAddPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "CausalMaskMulAdd");
  EXPECT_EQ(replacement[0].input()[2].value(), "N");
  EXPECT_TRUE(HasFunction(builder, "CausalMaskMulAdd"));

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.SetOpsetVersion("", 18);
  rejected.MakeInput("A", core::symbolic::TensorType::kInt64, Shape({1}));
  rejected.MakeInput("B", core::symbolic::TensorType::kInt64, Shape({1}));
  rejected.MakeInput("N", core::symbolic::TensorType::kInt64, Shape({1}));
  AddInt64(rejected, "zero", {}, {0});
  AddInt64(rejected, "one", {}, {1});
  AddInt64(rejected, "two", {}, {2});
  AddInt64(rejected, "axes1", {3}, {0, 1, 2});
  AddInt64(rejected, "axes2", {3}, {1, 2, 3});
  rejected.MakeNode("Squeeze", {"A"}, {"sA"});
  rejected.MakeNode("Squeeze", {"B"}, {"sB"});
  rejected.MakeNode("Range", {"zero", "sA", "two"}, {"r1"});
  rejected.MakeNode("Range", {"zero", "sB", "one"}, {"r2"});
  rejected.MakeNode("Unsqueeze", {"r1", "axes1"}, {"u1"});
  rejected.MakeNode("Unsqueeze", {"r2", "axes2"}, {"u2"});
  rejected.MakeNode("Mul", {"u2", "N"}, {"scaled"});
  rejected.MakeNode("Add", {"u1", "scaled"}, {"mask"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[7]).pattern, nullptr);
}

TEST(FunctionCosSinCachePattern, CreatesTypedFunctionAndRejectsMismatchedCasts) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("A", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("B", core::symbolic::TensorType::kInt64, Shape({1}));
  builder.MakeInput("weights", core::symbolic::TensorType::kFloat, Shape({1, 1, 8}));
  AddInt64(builder, "one", {}, {1});
  AddInt64(builder, "axes", {2}, {0, 1});
  AddInt64(builder, "reshape_shape", {3}, {0, -1, 1});
  builder.MakeNode("Squeeze", {"A"}, {"sA"});
  builder.MakeNode("Squeeze", {"B"}, {"sB"});
  builder.MakeNode("Range", {"sA", "sB", "one"}, {"positions"});
  builder.MakeNode("Unsqueeze", {"positions", "axes"}, {"unsqueezed"});
  builder.MakeNode("Cast", {"unsqueezed"}, {"cast"}, "", "",
                   IntAttr("to", TensorProto::DataType::FLOAT));
  builder.MakeNode("Reshape", {"cast", "reshape_shape"}, {"reshaped"});
  builder.MakeNode("Mul", {"weights", "reshaped"}, {"weighted"});
  builder.MakeNode("Cos", {"weighted"}, {"cos"});
  builder.MakeNode("Cast", {"cos"}, {"cos_cache"}, "", "",
                   IntAttr("to", TensorProto::DataType::FLOAT16));
  builder.MakeNode("Sin", {"weighted"}, {"sin"});
  builder.MakeNode("Cast", {"sin"}, {"sin_cache"}, "", "",
                   IntAttr("to", TensorProto::DataType::FLOAT16));
  builder.MakeOutput("cos_cache", core::symbolic::TensorType::kFloat16, Shape({1, 3, 8}));
  builder.MakeOutput("sin_cache", core::symbolic::TensorType::kFloat16, Shape({1, 3, 8}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::FunctionCosSinCachePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[7]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "CosSinCacheWithRange_to10");
  EXPECT_EQ(replacement[0].output()[0].value(), "cos_cache");
  EXPECT_EQ(replacement[0].output()[1].value(), "sin_cache");
  EXPECT_TRUE(HasFunction(builder, "CosSinCacheWithRange_to10"));

  core::builder::GraphBuilder rejected = std::move(builder);
  auto &rejected_nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(rejected.Nodes());
  for (AttributeProto &attribute : rejected_nodes[10].ref_attribute()) {
    if (attribute.name().value() == "to") {
      attribute.set_i(TensorProto::DataType::FLOAT);
    }
  }
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[7]).pattern, nullptr);
}

TEST(FunctionAttentionPattern, CreatesLocalAttentionAndRejectsPositiveInfinity) {
  auto make_builder = [](float infinity) {
    core::builder::GraphBuilder builder("attention", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    builder.MakeInput("query", core::symbolic::TensorType::kFloat, Shape({2, 4, 3, 8}));
    builder.MakeInput("keys", core::symbolic::TensorType::kFloat, Shape({2, 4, 5, 8}));
    builder.MakeInput("values", core::symbolic::TensorType::kFloat, Shape({2, 4, 5, 8}));
    builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 3, 5}));
    AddFloat(builder, "zero", {1}, {0.0F});
    AddFloat(builder, "infinity", {1}, {infinity});
    AddFloat(builder, "scale", {1}, {0.5F});
    builder.MakeNode("Mul", {"query", "scale"}, {"scaled_query"});
    builder.MakeNode("Mul", {"keys", "scale"}, {"scaled_keys"});
    builder.MakeNode("Transpose", {"scaled_keys"}, {"keys_t"}, "", "", PermAttr({0, 1, 3, 2}));
    builder.MakeNode("MatMul", {"scaled_query", "keys_t"}, {"scores"});
    builder.MakeNode("Where", {"mask", "zero", "infinity"}, {"bias"});
    builder.MakeNode("Add", {"scores", "bias"}, {"masked"});
    builder.MakeNode("Softmax", {"masked"}, {"probabilities"}, "", "softmax", IntAttr("axis", -1));
    builder.MakeNode("IsNaN", {"probabilities"}, {"nan"});
    builder.MakeNode("Where", {"nan", "zero", "probabilities"}, {"filtered"});
    builder.MakeNode("MatMul", {"filtered", "values"}, {"Y"});
    builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({2, 4, 3, 8}));
    return builder;
  };

  core::builder::GraphBuilder builder = make_builder(-std::numeric_limits<float>::infinity());
  core::builder::GraphGraph graph(builder);
  onnx_patterns::FunctionAttentionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[6]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "LocalAttention_to1");
  EXPECT_EQ(replacement[0].input()[0].value(), "query");
  EXPECT_EQ(replacement[0].input()[1].value(), "keys");
  EXPECT_EQ(replacement[0].input()[2].value(), "values");
  EXPECT_TRUE(HasFunction(builder, "LocalAttention_to1"));

  core::builder::GraphBuilder rejected = make_builder(std::numeric_limits<float>::infinity());
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[6]).pattern, nullptr);
}

TEST(FunctionAttentionGQAPattern, FusesRepeatInterleaveAndRejectsShapeMismatch) {
  auto make_builder = [](int64_t value_repeat) {
    core::builder::GraphBuilder builder("gqa_function", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    builder.SetOpsetVersion("intermediate", 1);
    builder.MakeInput("query", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 1}));
    builder.MakeInput("keys", core::symbolic::TensorType::kFloat, Shape({1, 1, 5, 1}));
    builder.MakeInput("values", core::symbolic::TensorType::kFloat, Shape({1, 1, 5, 1}));
    builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 3, 5}));
    AddFloat(builder, "scale", {1}, {0.5F});
    AddInt64(builder, "two", {1}, {2});
    AddInt64(builder, "key_expand_shape", {5}, {1, 1, 2, 1, 1});
    AddInt64(builder, "value_expand_shape", {5}, {1, 1, value_repeat, 1, 1});
    AddInt64(builder, "key_reshape_shape", {4}, {0, 2, -1, 1});
    AddInt64(builder, "value_reshape_shape", {4}, {0, value_repeat, -1, 1});
    builder.MakeNode("Unsqueeze", {"keys", "two"}, {"keys_u"});
    builder.MakeNode("Expand", {"keys_u", "key_expand_shape"}, {"keys_e"});
    builder.MakeNode("Reshape", {"keys_e", "key_reshape_shape"}, {"keys_r"});
    builder.MakeNode("Unsqueeze", {"values", "two"}, {"values_u"});
    builder.MakeNode("Expand", {"values_u", "value_expand_shape"}, {"values_e"});
    builder.MakeNode("Reshape", {"values_e", "value_reshape_shape"}, {"values_r"});
    builder.MakeNode("LocalAttentionSW_to1", {"query", "keys_r", "values_r", "mask", "scale"},
                     {"Y"}, "intermediate", "attention");
    builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 1}));
    return builder;
  };

  core::builder::GraphBuilder builder = make_builder(2);
  core::builder::GraphGraph graph(builder);
  onnx_patterns::FunctionAttentionGQAPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[6]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "LocalAttentionGQASW_to1");
  EXPECT_EQ(replacement[0].input()[1].value(), "keys");
  EXPECT_EQ(replacement[0].input()[2].value(), "values");
  EXPECT_TRUE(HasFunction(builder, "LocalAttentionGQASW_to1"));

  core::builder::GraphBuilder rejected = make_builder(3);
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[6]).pattern, nullptr);
}

TEST(AttentionGQAPattern, EmitsAttentionCacheOutputsAndRejectsOldOpset) {
  auto make_builder = [](int opset) {
    core::builder::GraphBuilder builder("attention_gqa", SchemaLookup());
    builder.SetOpsetVersion("", opset);
    builder.MakeInput("query", core::symbolic::TensorType::kFloat, Shape({1, 2, 1, 1}));
    builder.MakeInput("key", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
    builder.MakeInput("value", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
    builder.MakeInput("past_key", core::symbolic::TensorType::kFloat, Shape({1, 1, 0, 1}));
    builder.MakeInput("past_value", core::symbolic::TensorType::kFloat, Shape({1, 1, 0, 1}));
    builder.MakeInput("mask", core::symbolic::TensorType::kBool, Shape({1, 1, 1, 1}));
    AddInt64(builder, "two", {1}, {2});
    AddInt64(builder, "one", {1}, {1});
    AddInt64(builder, "expand_shape", {5}, {1, 1, 2, 1, 1});
    builder.MakeNode("Concat", {"past_key", "key"}, {"present_key"}, "", "key_concat",
                     IntAttr("axis", 2));
    builder.MakeNode("Concat", {"past_value", "value"}, {"present_value"}, "", "value_concat",
                     IntAttr("axis", 2));
    builder.MakeNode("Unsqueeze", {"present_key", "two"}, {"keys_u"});
    builder.MakeNode("Expand", {"keys_u", "expand_shape"}, {"keys_e"});
    builder.MakeNode("Squeeze", {"keys_e", "one"}, {"keys_r"});
    builder.MakeNode("Unsqueeze", {"present_value", "two"}, {"values_u"});
    builder.MakeNode("Expand", {"values_u", "expand_shape"}, {"values_e"});
    builder.MakeNode("Squeeze", {"values_e", "one"}, {"values_r"});
    NodeProto attention =
        MakeNode("Attention", {"query", "keys_r", "values_r", "mask"}, {"Y"}, "", "attention");
    AttributeProto *scale = attention.add_attribute();
    scale->set_name("scale");
    scale->set_type(AttributeProto::AttributeType::FLOAT);
    scale->set_f(0.11F);
    builder.ReserveName("Y");
    auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
    nodes.push_back(std::move(attention));
    SeedShape(builder, "Y", core::symbolic::TensorType::kFloat, Shape({1, 2, 1, 1}));
    builder.MakeOutput("Y", core::symbolic::TensorType::kFloat, Shape({1, 2, 1, 1}));
    builder.MakeOutput("present_key", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
    builder.MakeOutput("present_value", core::symbolic::TensorType::kFloat, Shape({1, 1, 1, 1}));
    return builder;
  };

  core::builder::GraphBuilder builder = make_builder(23);
  core::builder::GraphGraph graph(builder);
  onnx_patterns::AttentionGQAPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[8]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  EXPECT_EQ(match.insert_at, &builder.Nodes()[8]);
  const auto replacement = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0].op_type().value(), "Attention");
  ASSERT_EQ(replacement[0].output_size(), 3);
  EXPECT_EQ(replacement[0].output()[0].value(), "Y");
  EXPECT_EQ(replacement[0].output()[1].value(), "present_key");
  EXPECT_EQ(replacement[0].output()[2].value(), "present_value");
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacement[0], "scale", 0.0F), 0.11F);

  core::builder::GraphBuilder rejected = make_builder(22);
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[8]).pattern, nullptr);
}

TEST(RotaryConcatPartPattern, RejectsNonZeroPaddingAndNonContiguousSlices) {
  auto make_split = [](bool nonzero) {
    core::builder::GraphBuilder builder("rotary_split_guard", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({3, 16}));
    AddInt64(builder, "split", {2}, {8, 8});
    AddInt64(builder, "shape", {2}, {3, 8});
    builder.MakeNode("Split", {"X", "split"}, {"x1", "x2"}, "", "", IntAttr("axis", 1));
    builder.MakeNode("Neg", {"x1"}, {"nx1"});
    builder.MakeNode("ConstantOfShape", {"shape"}, {"zero"});
    if (nonzero) {
      auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
      AttributeProto *value = nodes[2].add_attribute();
      value->set_name("value");
      value->set_type(AttributeProto::AttributeType::TENSOR);
      *value->mutable_t() = MakeInitializer<float>("value", {1}, {1.0F});
    }
    builder.MakeNode("Concat", {"nx1", "zero"}, {"left"}, "", "", IntAttr("axis", 1));
    builder.MakeNode("Concat", {"zero", "x2"}, {"right"}, "", "", IntAttr("axis", 1));
    builder.MakeNode("Add", {"left", "right"}, {"Y"});
    return builder;
  };
  onnx_patterns::RotaryConcatPartPattern pattern;
  core::builder::GraphBuilder nonzero = make_split(true);
  core::builder::GraphGraph nonzero_graph(nonzero);
  EXPECT_EQ(pattern.Match(nonzero_graph, nonzero.Nodes()[5]).pattern, nullptr);

  auto make_slices = [](int64_t right_start, int64_t right_end) {
    const int64_t right_length = right_end - right_start;
    core::builder::GraphBuilder builder("rotary_slice_guard", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 16}));
    AddInt64(builder, "zero", {1}, {0});
    AddInt64(builder, "eight", {1}, {8});
    AddInt64(builder, "right_start", {1}, {right_start});
    AddInt64(builder, "right_end", {1}, {right_end});
    AddInt64(builder, "axis_negative", {1}, {-1});
    AddInt64(builder, "axis_positive", {1}, {1});
    AddInt64(builder, "left_zero_shape", {2}, {2, right_length});
    AddInt64(builder, "right_zero_shape", {2}, {2, 8});
    builder.MakeNode("Slice", {"X", "zero", "eight", "axis_negative"}, {"left_slice"});
    builder.MakeNode("Neg", {"left_slice"}, {"negated"});
    builder.MakeNode("ConstantOfShape", {"left_zero_shape"}, {"left_zero"});
    builder.MakeNode("Concat", {"negated", "left_zero"}, {"left"}, "", "", IntAttr("axis", -1));
    builder.MakeNode("Slice", {"X", "right_start", "right_end", "axis_positive"}, {"right_slice"});
    builder.MakeNode("ConstantOfShape", {"right_zero_shape"}, {"right_zero"});
    builder.MakeNode("Concat", {"right_zero", "right_slice"}, {"right"}, "", "",
                     IntAttr("axis", 1));
    builder.MakeNode("Add", {"left", "right"}, {"Y"});
    return builder;
  };
  core::builder::GraphBuilder normalized = make_slices(8, 16);
  core::builder::GraphGraph normalized_graph(normalized);
  EXPECT_EQ(pattern.Match(normalized_graph, normalized.Nodes()[7]).pattern, &pattern);
  core::builder::GraphBuilder gap = make_slices(9, 16);
  core::builder::GraphGraph gap_graph(gap);
  EXPECT_EQ(pattern.Match(gap_graph, gap.Nodes()[7]).pattern, nullptr);
}

TEST(FunctionHalfRotaryEmbeddingPattern, RejectsReversedConcatAndOpset17) {
  auto make_builder = [](int opset, bool reversed) {
    core::builder::GraphBuilder builder("half_rotary_guard", SchemaLookup());
    builder.SetOpsetVersion("", opset);
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 8}));
    builder.MakeInput("cos", core::symbolic::TensorType::kFloat, Shape({3, 8}));
    builder.MakeInput("sin", core::symbolic::TensorType::kFloat, Shape({3, 8}));
    builder.MakeNode("Split", {"X"}, {"x1", "x2"}, "", "", SplitAttrs(-1));
    builder.MakeNode("Neg", {"x2"}, {"negated"});
    builder.MakeNode("Concat",
                     reversed ? std::vector<std::string>{"x1", "negated"}
                              : std::vector<std::string>{"negated", "x1"},
                     {"rotated"}, "", "", IntAttr("axis", -1));
    builder.MakeNode("Mul", {"rotated", "sin"}, {"rotated_sin"});
    builder.MakeNode("Mul", {"X", "cos"}, {"x_cos"});
    builder.MakeNode("Add", {"rotated_sin", "x_cos"}, {"Y"});
    return builder;
  };
  onnx_patterns::FunctionHalfRotaryEmbeddingPattern pattern;
  core::builder::GraphBuilder reversed = make_builder(18, true);
  core::builder::GraphGraph reversed_graph(reversed);
  EXPECT_EQ(pattern.Match(reversed_graph, reversed.Nodes()[0]).pattern, nullptr);
  core::builder::GraphBuilder old_opset = make_builder(17, false);
  core::builder::GraphGraph old_opset_graph(old_opset);
  EXPECT_EQ(pattern.Match(old_opset_graph, old_opset.Nodes()[0]).pattern, nullptr);
}

TEST(RotaryEmbeddingPattern, ValidatesPartialWiringAndDimensions) {
  auto make_builder = [](bool second_half, int64_t cache_half) {
    core::builder::GraphBuilder builder("partial_rotary", SchemaLookup());
    builder.SetOpsetVersion("", 23);
    builder.SetOpsetVersion("intermediate", 1);
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 8}));
    builder.MakeInput("cos_half", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, cache_half}));
    builder.MakeInput("sin_half", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, cache_half}));
    AddInt64(builder, "split", {2}, {4, 4});
    builder.MakeNode("Concat", {"cos_half", "cos_half"}, {"cos"}, "", "", IntAttr("axis", -1));
    builder.MakeNode("Concat", {"sin_half", "sin_half"}, {"sin"}, "", "", IntAttr("axis", 3));
    builder.MakeNode("Split", {"X", "split"}, {"rotary", "tail"}, "", "", IntAttr("axis", -1));
    builder.MakeNode("HalfRotaryEmbedding", {second_half ? "tail" : "rotary", "cos", "sin"},
                     {"rotated"}, "intermediate");
    SeedShape(builder, "rotated", core::symbolic::TensorType::kFloat, Shape({1, 2, 3, 4}));
    builder.MakeNode("Concat", {"rotated", second_half ? "rotary" : "tail"}, {"Y"}, "", "",
                     IntAttr("axis", -1));
    return builder;
  };
  onnx_patterns::RotaryEmbeddingPattern pattern;
  core::builder::GraphBuilder positive = make_builder(false, 2);
  core::builder::GraphGraph positive_graph(positive);
  EXPECT_EQ(pattern.Match(positive_graph, positive.Nodes()[3]).pattern, &pattern);
  core::builder::GraphBuilder wrong_output = make_builder(true, 2);
  core::builder::GraphGraph wrong_output_graph(wrong_output);
  EXPECT_EQ(pattern.Match(wrong_output_graph, wrong_output.Nodes()[3]).pattern, nullptr);
  core::builder::GraphBuilder wrong_cache = make_builder(false, 1);
  core::builder::GraphGraph wrong_cache_graph(wrong_cache);
  EXPECT_EQ(pattern.Match(wrong_cache_graph, wrong_cache.Nodes()[3]).pattern, nullptr);
}

TEST(FunctionCosSinCachePattern, RejectsWrongPositionCastWeightsAndSharedIntermediate) {
  onnx_patterns::FunctionCosSinCachePattern pattern;
  core::builder::GraphBuilder wrong_cast = MakeCosSinBuilder(TensorProto::DataType::DOUBLE);
  core::builder::GraphGraph wrong_cast_graph(wrong_cast);
  EXPECT_EQ(
      pattern.Match(wrong_cast_graph, wrong_cast.Nodes()[FindNode(wrong_cast, "Cos")]).pattern,
      nullptr);
  core::builder::GraphBuilder wrong_weights =
      MakeCosSinBuilder(TensorProto::DataType::FLOAT, core::symbolic::TensorType::kFloat16);
  core::builder::GraphGraph wrong_weights_graph(wrong_weights);
  EXPECT_EQ(
      pattern.Match(wrong_weights_graph, wrong_weights.Nodes()[FindNode(wrong_weights, "Cos")])
          .pattern,
      nullptr);
  core::builder::GraphBuilder shared = MakeCosSinBuilder(
      TensorProto::DataType::FLOAT, core::symbolic::TensorType::kFloat, false, true);
  core::builder::GraphGraph shared_graph(shared);
  EXPECT_EQ(pattern.Match(shared_graph, shared.Nodes()[FindNode(shared, "Cos")]).pattern, nullptr);
}

TEST(FunctionAttentionPattern, RejectsFiniteMinimumAndInvalidFusedMatMulAttributes) {
  onnx_patterns::FunctionAttentionPattern pattern;
  core::builder::GraphBuilder finite = MakeAttentionBuilder(std::numeric_limits<float>::lowest());
  core::builder::GraphGraph finite_graph(finite);
  EXPECT_EQ(pattern.Match(finite_graph, finite.Nodes()[FindNode(finite, "Softmax")]).pattern,
            nullptr);

  core::builder::GraphBuilder valid_fused =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity(), true, true, 1.0F);
  core::builder::GraphGraph valid_fused_graph(valid_fused);
  const auto valid_match =
      pattern.Match(valid_fused_graph, valid_fused.Nodes()[FindNode(valid_fused, "Softmax")]);
  ASSERT_EQ(valid_match.pattern, &pattern);
  const auto valid_replacement = pattern.Apply(valid_fused_graph, valid_match.nodes);
  ASSERT_EQ(valid_replacement.size(), 1u);
  EXPECT_EQ(valid_replacement[0].op_type().value(), "LocalAttentionNoT_to1");
  core::builder::GraphBuilder missing_trans_b =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity(), true, false, 1.0F);
  core::builder::GraphGraph missing_trans_b_graph(missing_trans_b);
  EXPECT_EQ(pattern
                .Match(missing_trans_b_graph,
                       missing_trans_b.Nodes()[FindNode(missing_trans_b, "Softmax")])
                .pattern,
            nullptr);
  core::builder::GraphBuilder alpha_two =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity(), true, true, 2.0F);
  core::builder::GraphGraph alpha_two_graph(alpha_two);
  EXPECT_EQ(
      pattern.Match(alpha_two_graph, alpha_two.Nodes()[FindNode(alpha_two, "Softmax")]).pattern,
      nullptr);
}

TEST(AttentionPatternsTopology, OptimizeProducesTopologicalGraphs) {
  core::builder::GraphBuilder cos_sin =
      MakeCosSinBuilder(TensorProto::DataType::FLOAT, core::symbolic::TensorType::kFloat, true);
  OptimizeAndVerify<onnx_patterns::FunctionCosSinCachePattern>(cos_sin);
  ASSERT_EQ(cos_sin.Nodes().size(), 1u);
  EXPECT_EQ(cos_sin.Nodes()[0].op_type().value(), "CosSinCacheWithRange_to10");
  EXPECT_EQ(cos_sin.Nodes()[0].input()[2].value(), "weight_source");

  core::builder::GraphBuilder attention =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity(), false, true, 1.0F, true);
  OptimizeAndVerify<onnx_patterns::FunctionAttentionPattern>(attention);
  ASSERT_EQ(attention.Nodes().size(), 1u);
  EXPECT_EQ(attention.Nodes()[0].op_type().value(), "LocalAttention_to1");
  EXPECT_EQ(attention.Nodes()[0].input()[2].value(), "values");

  core::builder::GraphBuilder function_gqa = MakeFunctionGqaBuilder(true);
  OptimizeAndVerify<onnx_patterns::FunctionAttentionGQAPattern>(function_gqa);
  ASSERT_EQ(function_gqa.Nodes().size(), 1u);
  EXPECT_EQ(function_gqa.Nodes()[0].op_type().value(), "LocalAttentionGQASW_to1");
  EXPECT_EQ(function_gqa.Nodes()[0].input()[2].value(), "values_source");

  core::builder::GraphBuilder attention_gqa = MakeAttentionGqaBuilder();
  OptimizeAndVerify<onnx_patterns::AttentionGQAPattern>(attention_gqa);
  ASSERT_EQ(attention_gqa.Nodes().size(), 1u);
  EXPECT_EQ(attention_gqa.Nodes()[0].op_type().value(), "Attention");
}

TEST(AttentionGQAPattern, RejectsCacheAxisOptionalInputsAttributesAndEarlierConsumer) {
  onnx_patterns::AttentionGQAPattern pattern;
  const auto rejects = [&](core::builder::GraphBuilder builder) {
    core::builder::GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[FindNode(builder, "Attention")]).pattern,
              nullptr);
  };
  rejects(MakeAttentionGqaBuilder(23, 1));
  rejects(MakeAttentionGqaBuilder(23, 2, true));
  rejects(MakeAttentionGqaBuilder(23, 2, false, true));
  rejects(MakeAttentionGqaBuilder(23, 2, false, false, true));
}

TEST(AttentionPatternsSafety, RejectsEarlierExternalConsumers) {
  onnx_patterns::FunctionAttentionPattern attention_pattern;
  core::builder::GraphBuilder attention =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity(), false, true, 1.0F, false, true);
  core::builder::GraphGraph attention_graph(attention);
  EXPECT_EQ(
      attention_pattern.Match(attention_graph, attention.Nodes()[FindNode(attention, "Softmax")])
          .pattern,
      nullptr);

  onnx_patterns::AttentionGQAPattern gqa_pattern;
  core::builder::GraphBuilder gqa = MakeAttentionGqaBuilder(23, 2, false, false, true);
  core::builder::GraphGraph gqa_graph(gqa);
  EXPECT_EQ(gqa_pattern.Match(gqa_graph, gqa.Nodes()[FindNode(gqa, "Attention")]).pattern, nullptr);

  core::builder::GraphBuilder graph_output =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity());
  graph_output.MakeOutput("bias", core::symbolic::TensorType::kFloat, Shape({1, 1, 3, 5}));
  core::builder::GraphGraph graph_output_graph(graph_output);
  EXPECT_EQ(attention_pattern
                .Match(graph_output_graph, graph_output.Nodes()[FindNode(graph_output, "Softmax")])
                .pattern,
            nullptr);

  core::builder::GraphBuilder captured =
      MakeAttentionBuilder(-std::numeric_limits<float>::infinity());
  core::builder::GraphBuilder &body = captured.MakeSubgraph("body");
  body.MakeNode("Identity", {"bias"}, {"captured_bias"});
  body.MakeOutput("captured_bias");
  AddSubgraphReference(captured, "body", "body_result");
  captured.MakeOutput("body_result");
  core::builder::GraphGraph captured_graph(captured);
  EXPECT_TRUE(captured_graph.IsUsedBySubgraph("bias"));
  EXPECT_EQ(attention_pattern.Match(captured_graph, captured.Nodes()[FindNode(captured, "Softmax")])
                .pattern,
            nullptr);
}

} // namespace
} // namespace Test
