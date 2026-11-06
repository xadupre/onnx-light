// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"
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

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape4D(int64_t n, int64_t c, int64_t h, int64_t w) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(n));
  shape.PushBack(core::symbolic::SymDim(c));
  shape.PushBack(core::symbolic::SymDim(h));
  shape.PushBack(core::symbolic::SymDim(w));
  return shape;
}

const AttributeProto *FindAttr(const NodeProto &node, const std::string &name) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name().value() == name) {
      return &attribute;
    }
  }
  return nullptr;
}

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const TensorProto &initializer : builder.Initializers()) {
    if (initializer.name().value() == name) {
      return &initializer;
    }
  }
  return nullptr;
}

template <typename Pattern> void OptimizeWith(core::builder::GraphBuilder &builder) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<Pattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();
}

TEST(ConvBiasNullPattern, RemovesZeroBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {0.0f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBiasNullPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Conv");
  EXPECT_EQ(replacements[0].input_size(), 2);
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "w");
}

TEST(ConvBiasNullPattern, RejectsNonZeroBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {0.5f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBiasNullPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Conv bias is not a known all-zero constant");
}

TEST(ConvBiasNullPattern, RemovesShapeExpandedZeroBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape4D(2, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {0.0f, 0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  utils::RepeatedProtoField<AttributeProto> shape_attributes;
  AttributeProto &start = shape_attributes.add();
  start.set_name("start");
  start.set_type(AttributeProto::AttributeType::INT);
  start.set_i(0);
  AttributeProto &end = shape_attributes.add();
  end.set_name("end");
  end.set_type(AttributeProto::AttributeType::INT);
  end.set_i(1);
  builder.MakeNode("Shape", {"b"}, {"b_shape"}, "", "", shape_attributes);
  builder.MakeNode("Expand", {"zero", "b_shape"}, {"expanded_bias"});
  builder.MakeNode("Conv", {"x", "w", "expanded_bias"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::ConvBiasNullPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Conv");
  ASSERT_EQ(builder.Nodes()[0].input_size(), 2);
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].input()[1].value(), "w");
}

TEST(ConvBiasNullPattern, RejectsExpandedZeroWithDynamicShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape4D(2, 1, 2, 2));
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  builder.MakeInput("bias_shape", core::symbolic::TensorType::kInt64, shape);
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Expand", {"zero", "bias_shape"}, {"expanded_bias"});
  builder.MakeNode("Conv", {"x", "w", "expanded_bias"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBiasNullPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Conv bias is not a constant");
}

TEST(PadConvPattern, FoldsSpatialPadding) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<int64_t>("pads", {8}, {0, 0, 1, 1, 0, 0, 1, 1}));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeNode("Pad", {"x", "pads"}, {"padded"});
  builder.MakeNode("Conv", {"padded", "w"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadConvPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Conv");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  const AttributeProto *pads = FindAttr(replacements[0], "pads");
  ASSERT_NE(pads, nullptr);
  ASSERT_EQ(pads->ints().size(), 4);
  EXPECT_EQ(pads->ints()[0], 1);
  EXPECT_EQ(pads->ints()[1], 1);
  EXPECT_EQ(pads->ints()[2], 1);
  EXPECT_EQ(pads->ints()[3], 1);
}

TEST(PadConvPattern, RejectsChannelPadding) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 4));
  builder.MakeInitializer(MakeInitializer<int64_t>("pads", {8}, {0, 1, 1, 1, 0, 1, 1, 1}));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  builder.MakeNode("Pad", {"x", "pads"}, {"padded"});
  builder.MakeNode("Conv", {"padded", "w"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadConvPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the Pad node pads the batch or channel dimension");
}

void MakePadPool(core::builder::GraphBuilder &builder, const std::string &op = "AveragePool",
                 const std::vector<int64_t> &pads = {0, 0, 1, 0, 0, 0, 0, 1},
                 const NodeProto &pool_attributes = NodeProto{}, bool axes = false) {
  builder.MakeInput("x",
                    op == "MaxPool" ? core::symbolic::TensorType::kUint8
                                    : core::symbolic::TensorType::kFloat,
                    Shape4D(1, 1, 4, 5));
  builder.MakeInitializer(
      MakeInitializer<int64_t>("pads", {static_cast<int64_t>(pads.size())}, pads));
  std::vector<std::string> inputs{"x", "pads"};
  if (axes) {
    builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {-1}));
    inputs.insert(inputs.end(), {"", "axes"});
  }
  builder.MakeNode("Pad", inputs, {"padded"});
  NodeProto pool = pool_attributes;
  AddAttribute<std::vector<int64_t>>(pool, "kernel_shape", {3, 3});
  builder.MakeNode(op, {"padded"}, {"y"}, "", "pool", pool.attribute());
  builder.MakeOutput("y");
}

TEST(PadPoolPattern, FoldsAsymmetricPaddingAndPreservesPoolingAttributes) {
  for (const auto *op : {"AveragePool", "MaxPool"}) {
    for (bool include : {false, true}) {
      SCOPED_TRACE(std::string(op) + ":" + std::to_string(include));
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 19}});
      NodeProto attributes;
      AddAttribute<std::vector<int64_t>>(attributes, "strides", {2, 1});
      AddAttribute<std::vector<int64_t>>(attributes, "dilations", {2, 1});
      if (include) {
        AddAttribute<std::vector<int64_t>>(attributes, "pads", {0, 1, 1, 0});
        if (std::string(op) == "AveragePool")
          AddAttribute<int64_t>(attributes, "count_include_pad", 1);
      }
      MakePadPool(builder, op, {0, 0, 1, 0, 0, 0, 0, 1}, attributes);
      const auto output_shape = builder.GetShape("y").Shape();
      std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
      patterns.push_back(onnx_patterns::CreatePattern("PadPool"));
      core::builder::GraphGraph graph(builder, std::move(patterns));
      core::builder::ConstantFoldingOptions folding;
      folding.enabled = false;
      ASSERT_FALSE(graph.Optimize(2, nullptr, folding).rewrites().empty());
      ASSERT_EQ(builder.Nodes().size(), 1u);
      const auto &pool = builder.Nodes()[0];
      EXPECT_EQ(pool.op_type(), op);
      EXPECT_EQ(pool.input(0), "x");
      EXPECT_EQ(pool.output(0), "y");
      EXPECT_EQ(pool.name(), "PadPoolPattern--pool");
      std::vector<int64_t> actual;
      ASSERT_TRUE(GetAttributeInts(pool, "pads", actual));
      EXPECT_EQ(actual,
                (include ? std::vector<int64_t>{1, 1, 1, 1} : std::vector<int64_t>{1, 0, 0, 1}));
      actual.clear();
      ASSERT_TRUE(GetAttributeInts(pool, "strides", actual));
      EXPECT_EQ(actual, (std::vector<int64_t>{2, 1}));
      actual.clear();
      ASSERT_TRUE(GetAttributeInts(pool, "dilations", actual));
      EXPECT_EQ(actual, (std::vector<int64_t>{2, 1}));
      if (std::string(op) == "AveragePool") {
        EXPECT_EQ(GetAttributeOr<int64_t>(pool, "count_include_pad", -1), 1);
      }
      EXPECT_EQ(builder.GetShape("y").Shape(), output_shape);
      EXPECT_TRUE(graph.Optimize(2, nullptr, folding).rewrites().empty());
    }
  }
  const auto l1 = core::builder::RegisteredPatternSets().at("L1");
  EXPECT_NE(std::find(l1.begin(), l1.end(), "PadPool"), l1.end());
}

TEST(PadPoolPattern, SupportsConstantAxesAndSafeCeilMode) {
  for (const auto *op : {"AveragePool", "MaxPool"}) {
    for (bool axes : {false, true}) {
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
      NodeProto attributes;
      AddAttribute<int64_t>(attributes, "ceil_mode", 1);
      if (std::string(op) == "AveragePool")
        AddAttribute<int64_t>(attributes, "count_include_pad", 1);
      MakePadPool(builder, op,
                  axes ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{0, 0, 0, 1, 0, 0, 0, 0},
                  attributes, axes);
      core::builder::GraphGraph graph(builder);
      onnx_patterns::PadPoolPattern pattern;
      const auto match = pattern.Match(graph, builder.Nodes()[1]);
      ASSERT_EQ(match.pattern, &pattern) << match.ToString();
      const auto replacements = pattern.Apply(graph, match.nodes);
      ASSERT_EQ(replacements.size(), 1u);
      std::vector<int64_t> pads;
      ASSERT_TRUE(GetAttributeInts(replacements[0], "pads", pads));
      EXPECT_EQ(pads, (std::vector<int64_t>{0, 1, 0, 0}));
      EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "ceil_mode", 0), 1);
    }
  }
}

TEST(PadPoolPattern, ChecksTypeAndOpsetBoundariesWithSymbolicSpatialDimensions) {
  using core::symbolic::TensorType;
  for (const auto type : {TensorType::kFloat, TensorType::kFloat16, TensorType::kDouble,
                          TensorType::kBfloat16, TensorType::kUint8, TensorType::kInt8}) {
    for (int opset : {11, 12, 22}) {
      SCOPED_TRACE(std::to_string(static_cast<int>(type)) + ":" + std::to_string(opset));
      const bool maximum = type == TensorType::kUint8 || type == TensorType::kInt8;
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", opset}});
      core::symbolic::SymShape shape;
      shape.PushBack(core::symbolic::SymDim(1));
      shape.PushBack(core::symbolic::SymDim(2));
      shape.PushBack(core::symbolic::SymDim("width"));
      builder.MakeInput("x", type, shape);
      builder.MakeInitializer(MakeInitializer<int64_t>("pads", {6}, {0, 0, 1, 0, 0, 0}));
      builder.MakeNode("Pad", {"x", "pads"}, {"padded"});
      NodeProto attributes;
      AddAttribute<std::vector<int64_t>>(attributes, "kernel_shape", {3});
      builder.MakeNode(maximum ? "MaxPool" : "AveragePool", {"padded"}, {"y"}, "", "",
                       attributes.attribute());
      builder.MakeOutput("y");
      core::builder::GraphGraph graph(builder);
      onnx_patterns::PadPoolPattern pattern;
      const bool supported = maximum ? type == TensorType::kUint8 && opset >= 12
                                     : type != TensorType::kBfloat16 || opset >= 22;
      const auto match = pattern.Match(graph, builder.Nodes()[1]);
      EXPECT_EQ(match.pattern != nullptr, supported) << match.ToString();
      if (supported) {
        const auto replacement = pattern.Apply(graph, match.nodes);
        ASSERT_EQ(replacement.size(), 1u);
        EXPECT_EQ(replacement[0].input(0), "x");
        EXPECT_EQ(replacement[0].output(0), "y");
      }
    }
  }
}

TEST(PadPoolPattern, SupportsExplicit16BitZeroScalarsOnly) {
  using core::symbolic::TensorType;
  for (const auto type : {TensorType::kFloat16, TensorType::kBfloat16}) {
    for (bool raw : {false, true}) {
      for (int32_t bits : {0, 0x8000, 1, 0x3c00, 0x7c00, 0x7fff}) {
        SCOPED_TRACE(testing::Message() << static_cast<int>(type) << ":" << raw << ":" << bits);
        core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 22}});
        builder.MakeInput("x", type, Shape4D(1, 1, 4, 5));
        builder.MakeInitializer(MakeInitializer<int64_t>("pads", {8}, {0, 0, 1, 0, 0, 0, 0, 1}));
        TensorProto value;
        value.set_name("value");
        value.set_data_type(core::symbolic::TensorTypeToDataType(type));
        if (raw) {
          const uint8_t bytes[] = {static_cast<uint8_t>(bits), static_cast<uint8_t>(bits >> 8)};
          value.set_raw_data(bytes, sizeof(bytes));
        } else {
          value.add_int32_data(bits);
        }
        builder.MakeInitializer(value);
        builder.MakeNode("Pad", {"x", "pads", "value"}, {"padded"});
        NodeProto attributes;
        AddAttribute<std::vector<int64_t>>(attributes, "kernel_shape", {3, 3});
        builder.MakeNode("AveragePool", {"padded"}, {"y"}, "", "", attributes.attribute());
        builder.MakeOutput("y");
        const auto before = builder.ToModel();
        core::builder::GraphGraph graph(builder);
        onnx_patterns::PadPoolPattern pattern;
        const auto match = pattern.Match(graph, builder.Nodes()[1]);
        if (bits == 0 || bits == 0x8000) {
          ASSERT_EQ(match.pattern, &pattern);
          const auto replacements = pattern.Apply(graph, match.nodes);
          ASSERT_EQ(replacements.size(), 1u);
          EXPECT_EQ(replacements[0].input(0), "x");
          EXPECT_EQ(replacements[0].output(0), "y");
          std::vector<int64_t> pads;
          ASSERT_TRUE(GetAttributeInts(replacements[0], "pads", pads));
          EXPECT_EQ(pads, (std::vector<int64_t>{1, 0, 0, 1}));
        } else {
          EXPECT_EQ(match.pattern, nullptr);
          EXPECT_THROW(pattern.Apply(graph, {&builder.Nodes()[0], &builder.Nodes()[1]}),
                       core::builder::BuilderError);
        }
        EXPECT_TRUE(builder.ToModel().Equals(before));
      }
    }
  }
}

TEST(PadPoolPattern, RejectsAmbiguousAndMalformedConstantsInSharedPadReader) {
  for (bool convolution : {false, true}) {
    for (const auto *variant : {"mixed_pads", "mixed_axes", "mixed_value", "mixed_typed_value",
                                "extra_typed_value", "extra_raw_value", "truncated_raw_value"}) {
      SCOPED_TRACE(testing::Message() << convolution << ":" << variant);
      const std::string test = variant;
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
      builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 5));
      auto pads = MakeInitializer<int64_t>("pads", {2}, {1, 1});
      auto axes = MakeInitializer<int64_t>("axes", {1}, {-1});
      auto value = MakeInitializer<float>("value", {}, {0});
      if (test == "mixed_pads") {
        const uint8_t raw[] = {2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
        pads.set_raw_data(raw, sizeof(raw));
      } else if (test == "mixed_axes") {
        const uint8_t raw[] = {2, 0, 0, 0, 0, 0, 0, 0};
        axes.set_raw_data(raw, sizeof(raw));
      } else if (test == "mixed_value") {
        const uint8_t raw[] = {0, 0, 0, 0x40}; // FLOAT 2.
        value.set_raw_data(raw, sizeof(raw));
      } else if (test == "mixed_typed_value") {
        value.add_int32_data(2);
      } else if (test == "extra_typed_value") {
        value.add_float_data(2);
      } else {
        value.clr_float_data();
        const uint8_t raw[8] = {};
        value.set_raw_data(raw, test == "extra_raw_value" ? 8 : 3);
      }
      builder.MakeInitializer(pads);
      builder.MakeInitializer(axes);
      builder.MakeInitializer(value);
      builder.MakeNode("Pad", {"x", "pads", "value", "axes"}, {"padded"});
      if (convolution) {
        builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 1, 3}, {1, 1, 1}));
        builder.MakeNode("Conv", {"padded", "w"}, {"y"});
      } else {
        NodeProto attributes;
        AddAttribute<std::vector<int64_t>>(attributes, "kernel_shape", {1, 3});
        builder.MakeNode("AveragePool", {"padded"}, {"y"}, "", "", attributes.attribute());
      }
      builder.MakeOutput("y");
      const auto before = builder.ToModel();
      core::builder::GraphGraph graph(builder);
      onnx_patterns::PadPoolPattern pool;
      onnx_patterns::PadConvPattern conv;
      const core::builder::PatternOptimization &pattern =
          convolution ? static_cast<const core::builder::PatternOptimization &>(conv)
                      : static_cast<const core::builder::PatternOptimization &>(pool);
      EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
      EXPECT_THROW(pattern.Apply(graph, {&builder.Nodes()[0], &builder.Nodes()[1]}),
                   core::builder::BuilderError);
      EXPECT_TRUE(builder.ToModel().Equals(before));
    }
  }
}

TEST(PadPoolPattern, PreservesMetadataAndRejectsConflictingMetadataAndInvalidAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  MakePadPool(builder, "AveragePool", {1, 0}, NodeProto{}, true);
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[1].set_doc_string("pool documentation");
  for (std::size_t i = 0; i < 2; ++i) {
    auto *entry = nodes[i].add_metadata_props();
    entry->set_key(i == 0 ? "pad" : "pool");
    entry->set_value("retained");
  }
  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadPoolPattern pattern;
  const auto match = pattern.Match(graph, nodes[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].doc_string(), "pool documentation");
  ASSERT_EQ(replacements[0].metadata_props().size(), 2u);
  nodes[1].ref_metadata_props()[0].set_key("pad");
  nodes[1].ref_metadata_props()[0].set_value("conflict");
  EXPECT_EQ(pattern.Match(graph, nodes[1]).pattern, nullptr);
  nodes[1].ref_metadata_props()[0].set_key("pool");
  builder.MakeInitializer(MakeInitializer<int64_t>("duplicate_axes", {2}, {-1, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("bad_axis", {1}, {4}));
  builder.MakeInitializer(MakeInitializer<int64_t>("two_pads", {4}, {1, 1, 0, 0}));
  for (const auto *axes : {"duplicate_axes", "bad_axis", "two_pads"}) {
    SCOPED_TRACE(axes);
    nodes[0].ref_input()[3] = axes;
    nodes[0].ref_input()[1] = std::string(axes) == "duplicate_axes" ? "two_pads" : "pads";
    EXPECT_EQ(pattern.Match(graph, nodes[1]).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {&nodes[0], &nodes[1]}), core::builder::BuilderError);
  }
}

TEST(PadConvPattern, ReusesFullRankPaddingFromCompactAxes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 5));
  builder.MakeInitializer(MakeInitializer<int64_t>("pads", {2}, {1, 2}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {-1}));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 2, 2}, {1, 1, 1, 1}));
  builder.MakeNode("Pad", {"x", "pads", "", "axes"}, {"padded"});
  builder.MakeNode("Conv", {"padded", "w"}, {"y"});
  builder.MakeOutput("y");
  OptimizeWith<onnx_patterns::PadConvPattern>(builder);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Conv");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  std::vector<int64_t> pads;
  ASSERT_TRUE(GetAttributeInts(builder.Nodes()[0], "pads", pads));
  EXPECT_EQ(pads, (std::vector<int64_t>{0, 1, 0, 2}));
}

TEST(PadPoolPattern, RejectsChangedValuesDivisorsWindowsAndIndicesWithoutMutation) {
  for (const auto *variant :
       {"negative_pad", "channel_pad", "nonzero_value", "float_max", "indices", "ceil_tail",
        "ceil_divisor", "existing_pad_default_divisor", "existing_pad_zero_divisor", "oversize_pad",
        "overflow_pad", "auto_pad", "old_opset", "mode", "dynamic_pads"}) {
    SCOPED_TRACE(variant);
    const std::string test = variant;
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 19}});
    MakePadPool(builder, test == "indices" || test == "float_max" ? "MaxPool" : "AveragePool");
    auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
    if (test == "negative_pad" || test == "channel_pad" || test == "oversize_pad" ||
        test == "overflow_pad") {
      auto pads = std::vector<int64_t>{0, 0, 1, 0, 0, 0, 0, 1};
      if (test == "negative_pad")
        pads[2] = -1;
      else if (test == "channel_pad")
        pads[1] = 1;
      else
        pads[2] = test == "oversize_pad" ? 3 : std::numeric_limits<int64_t>::max();
      builder.MakeInitializer(MakeInitializer<int64_t>("bad_pads", {8}, pads));
      nodes[0].ref_input()[1] = "bad_pads";
    } else if (test == "nonzero_value") {
      builder.MakeInitializer(MakeInitializer<float>("value", {}, {1}));
      nodes[0].add_input("value");
    } else if (test == "float_max") {
      builder.MakeInput("float_x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 4, 5));
      nodes[0].ref_input()[0] = "float_x";
    } else if (test == "indices") {
      nodes[1].add_output("indices");
    } else if (test == "ceil_tail" || test == "ceil_divisor") {
      AddAttribute<int64_t>(nodes[1], "ceil_mode", 1);
      if (test == "ceil_tail")
        AddAttribute<int64_t>(nodes[1], "count_include_pad", 1);
    } else if (test == "existing_pad_default_divisor" || test == "existing_pad_zero_divisor") {
      AddAttribute<std::vector<int64_t>>(nodes[1], "pads", {1, 0, 0, 0});
      if (test == "existing_pad_zero_divisor")
        AddAttribute<int64_t>(nodes[1], "count_include_pad", 0);
    } else if (test == "auto_pad") {
      AddAttribute<std::string>(nodes[1], "auto_pad", "SAME_UPPER");
    } else if (test == "old_opset") {
      builder.SetOpsetVersion("", 10);
    } else if (test == "mode") {
      AddAttribute<std::string>(nodes[0], "mode", "edge");
    } else {
      core::symbolic::SymShape shape;
      shape.PushBack(core::symbolic::SymDim(8));
      builder.MakeInput("dynamic_pads", core::symbolic::TensorType::kInt64, shape);
      nodes[0].ref_input()[1] = "dynamic_pads";
    }
    const auto before = builder.ToModel();
    core::builder::GraphGraph graph(builder);
    onnx_patterns::PadPoolPattern pattern;
    EXPECT_EQ(pattern.Match(graph, nodes[1]).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {&nodes[0], &nodes[1]}), core::builder::BuilderError);
    EXPECT_TRUE(builder.ToModel().Equals(before));
  }
}

TEST(PadPoolPattern, RejectsObservablePadOutputsWithoutMutation) {
  for (int usage = 0; usage < 3; ++usage) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 19}});
    MakePadPool(builder);
    if (usage == 0) {
      builder.MakeNode("Identity", {"padded"}, {"other"});
      builder.MakeOutput("other");
    } else if (usage == 1) {
      builder.MakeOutput("padded");
    } else {
      auto &subgraph = builder.MakeSubgraph("body");
      subgraph.MakeNode("Identity", {"padded"}, {"captured"});
      subgraph.MakeOutput("captured");
      NodeProto attributes;
      AddAttribute<std::string>(attributes, "body_ref", "body");
      builder.MakeNode("SubgraphCarrier", {}, {"carrier"}, "", "", attributes.attribute());
    }
    const auto before = builder.ToModel();
    core::builder::GraphGraph graph(builder);
    onnx_patterns::PadPoolPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {&builder.Nodes()[0], &builder.Nodes()[1]}),
                 core::builder::BuilderError);
    EXPECT_TRUE(builder.ToModel().Equals(before));
  }
}

TEST(PadPoolPattern, RejectsMalformedFormsAndInconsistentApply) {
  for (const auto *variant :
       {"pool_domain", "pad_domain", "overload", "device", "wrong_pads_rank", "wrong_kernel_rank",
        "zero_stride", "zero_dilation", "duplicate", "reference", "wrong_attribute_type",
        "bad_flag", "unknown_attribute", "old_dilation", "missing_shape"}) {
    SCOPED_TRACE(variant);
    const std::string test = variant;
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 19}});
    MakePadPool(builder);
    auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
    if (test == "pool_domain" || test == "pad_domain")
      nodes[test == "pad_domain" ? 0 : 1].set_domain("foreign");
    else if (test == "overload")
      nodes[0].set_overload("foreign");
    else if (test == "device")
      nodes[1].add_device_configurations();
    else if (test == "wrong_pads_rank")
      AddAttribute<std::vector<int64_t>>(nodes[1], "pads", {1, 0});
    else if (test == "wrong_kernel_rank")
      nodes[1].ref_attribute()[0].ref_ints().resize(1);
    else if (test == "zero_stride" || test == "zero_dilation")
      AddAttribute<std::vector<int64_t>>(nodes[1], test == "zero_stride" ? "strides" : "dilations",
                                         {0, 1});
    else if (test == "duplicate")
      *nodes[1].add_attribute() = nodes[1].attribute(0);
    else if (test == "reference")
      nodes[1].ref_attribute()[0].set_ref_attr_name("kernel");
    else if (test == "wrong_attribute_type")
      nodes[1].ref_attribute()[0].set_type(AttributeProto::INT);
    else if (test == "bad_flag")
      AddAttribute<int64_t>(nodes[1], "count_include_pad", 2);
    else if (test == "unknown_attribute")
      AddAttribute<int64_t>(nodes[0], "unknown", 1);
    else if (test == "old_dilation") {
      builder.SetOpsetVersion("", 18);
      AddAttribute<std::vector<int64_t>>(nodes[1], "dilations", {1, 1});
    } else {
      auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
      shapes.Erase("x");
    }
    const auto before = builder.ToModel();
    core::builder::GraphGraph graph(builder);
    onnx_patterns::PadPoolPattern pattern;
    EXPECT_EQ(pattern.Match(graph, nodes[1]).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {&nodes[0], &nodes[1]}), core::builder::BuilderError);
    EXPECT_TRUE(builder.ToModel().Equals(before));
  }
  core::builder::GraphBuilder builder("valid", SchemaLookup(), {{"", 19}});
  MakePadPool(builder);
  core::builder::GraphGraph graph(builder);
  onnx_patterns::PadPoolPattern pattern;
  EXPECT_THROW(pattern.Apply(graph, {}), core::builder::BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {nullptr, &builder.Nodes()[1]}), core::builder::BuilderError);
  EXPECT_THROW(pattern.Apply(graph, {&builder.Nodes()[1], &builder.Nodes()[0]}),
               core::builder::BuilderError);
}

TEST(ConvAddFusionPattern, FoldsBiasAndLeavesActivation) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {0.5f, -0.5f}));
  builder.MakeInitializer(MakeInitializer<float>("add", {1, 2, 1, 1}, {1.0f, 2.0f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"conv"});
  builder.MakeNode("Add", {"add", "conv"}, {"added"});
  builder.MakeNode("Relu", {"added"}, {"y"});
  builder.MakeOutput("y");

  OptimizeWith<onnx_patterns::ConvAddFusionPattern>(builder);

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Conv");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "added");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Relu");
  const TensorProto *folded_bias = FindInitializer(builder, builder.Nodes()[0].input()[2].value());
  ASSERT_NE(folded_bias, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*folded_bias, values));
  EXPECT_EQ(values, (std::vector<double>{1.5, 1.5}));
}

TEST(ConvAddFusionPattern, RejectsSharedConvOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("add", {2, 1, 1}, {1.0f, 2.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("Add", {"conv", "add"}, {"y"});
  builder.MakeNode("Identity", {"conv"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvAddFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvAddFusionPattern, RejectsNonChannelBroadcast) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("add", {2}, {1.0f, 2.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("Add", {"conv", "add"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvAddFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvMulFusionPattern, FoldsScalarIntoDoubleWeightsAndBias) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<double>("w", {2, 1, 1, 1}, {2.0, 3.0}));
  builder.MakeInitializer(MakeInitializer<double>("b", {2}, {0.5, -0.5}));
  builder.MakeInitializer(MakeInitializer<double>("scale", {}, {4.0}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"conv"});
  builder.MakeNode("Mul", {"conv", "scale"}, {"y"});
  builder.MakeOutput("y");

  OptimizeWith<onnx_patterns::ConvMulFusionPattern>(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &conv = builder.Nodes()[0];
  const TensorProto *folded_weights = FindInitializer(builder, conv.input()[1].value());
  const TensorProto *folded_bias = FindInitializer(builder, conv.input()[2].value());
  ASSERT_NE(folded_weights, nullptr);
  ASSERT_NE(folded_bias, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*folded_weights, values));
  EXPECT_EQ(values, (std::vector<double>{8.0, 12.0}));
  ASSERT_TRUE(ReadFloatingValues(*folded_bias, values));
  EXPECT_EQ(values, (std::vector<double>{2.0, -2.0}));
}

TEST(ConvMulFusionPattern, RejectsMismatchedDtype) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<double>("scale", {}, {4.0}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("Mul", {"conv", "scale"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvMulFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvBatchNormalizationFusionPattern, FoldsParameters) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 14);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 1, 1, 1}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {0.5f, -0.5f}));
  builder.MakeInitializer(MakeInitializer<float>("scale", {2}, {2.0f, 3.0f}));
  builder.MakeInitializer(MakeInitializer<float>("bn_b", {2}, {1.0f, -1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("mean", {2}, {0.5f, 1.5f}));
  builder.MakeInitializer(MakeInitializer<float>("var", {2}, {4.0f, 9.0f}));
  builder.MakeNode("Conv", {"x", "w", "b"}, {"conv"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &epsilon = attributes.add();
  epsilon.set_name("epsilon");
  epsilon.set_type(AttributeProto::AttributeType::FLOAT);
  epsilon.set_f(0.0f);
  builder.MakeNode("BatchNormalization", {"conv", "scale", "bn_b", "mean", "var"}, {"y", "", ""},
                   "", "", attributes);
  builder.MakeOutput("y");

  {
    core::builder::GraphGraph graph(builder);
    onnx_patterns::ConvBatchNormalizationFusionPattern pattern;
    const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
    ASSERT_NE(match.pattern, nullptr)
        << (match.no_match.has_value() ? match.no_match->reason : "no diagnostic");
  }
  OptimizeWith<onnx_patterns::ConvBatchNormalizationFusionPattern>(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &conv = builder.Nodes()[0];
  const TensorProto *folded_weights = FindInitializer(builder, conv.input()[1].value());
  const TensorProto *folded_bias = FindInitializer(builder, conv.input()[2].value());
  ASSERT_NE(folded_weights, nullptr);
  ASSERT_NE(folded_bias, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*folded_weights, values));
  EXPECT_EQ(values, (std::vector<double>{2.0, 3.0}));
  ASSERT_TRUE(ReadFloatingValues(*folded_bias, values));
  EXPECT_EQ(values, (std::vector<double>{1.0, -3.0}));
}

TEST(ConvBatchNormalizationFusionPattern, RejectsOptionalOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 14);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 1, 1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("scale", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("bn_b", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("mean", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("var", {1}, {1.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("BatchNormalization", {"conv", "scale", "bn_b", "mean", "var"},
                   {"y", "saved_mean", ""});
  builder.MakeOutput("y");
  builder.MakeOutput("saved_mean");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBatchNormalizationFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ConvBatchNormalizationFusionPattern, RejectsTrainingOutputsBeforeOpset14) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape4D(1, 1, 2, 2));
  builder.MakeInitializer(MakeInitializer<float>("w", {1, 1, 1, 1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("scale", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("bn_b", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("mean", {1}, {0.0f}));
  builder.MakeInitializer(MakeInitializer<float>("var", {1}, {1.0f}));
  builder.MakeNode("Conv", {"x", "w"}, {"conv"});
  builder.MakeNode("BatchNormalization", {"conv", "scale", "bn_b", "mean", "var"},
                   {"y", "running_mean", "running_var", "saved_mean", "saved_var"});

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConvBatchNormalizationFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

} // namespace
} // namespace Test
