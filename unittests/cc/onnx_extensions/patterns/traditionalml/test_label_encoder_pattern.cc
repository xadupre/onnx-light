// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/traditionalml/label_encoder_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape1D(int64_t size) {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(size));
  return shape;
}

void AddInts(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
             const std::vector<int64_t> &values) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : values) {
    attribute.add_ints(value);
  }
}

void AddStrings(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
                const std::vector<std::string> &values) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::STRINGS);
  for (const std::string &value : values) {
    attribute.add_strings(value);
  }
}

void AddInt(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
            int64_t value) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(value);
}

void AddString(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
               const std::string &value) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s(value);
}

TEST(LabelEncoderFusionPattern, ComposesMappingsAndDefaults) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("ai.onnx.ml", 4);
  builder.MakeInput("x", core::symbolic::TensorType::kString, Shape1D(3));
  utils::RepeatedProtoField<AttributeProto> first;
  AddStrings(first, "keys_strings", {"a", "b"});
  AddInts(first, "values_int64s", {10, 20});
  AddInt(first, "default_int64", 30);
  utils::RepeatedProtoField<AttributeProto> second;
  AddInts(second, "keys_int64s", {10, 30});
  AddStrings(second, "values_strings", {"ten", "other"});
  AddString(second, "default_string", "missing");
  builder.MakeNode("LabelEncoder", {"x"}, {"mid"}, "ai.onnx.ml", "first", first);
  builder.MakeNode("LabelEncoder", {"mid"}, {"y"}, "ai.onnx.ml", "second", second);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LabelEncoderFusionPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  const AttributeProto *values = FindAttribute(replacements[0], "values_strings");
  ASSERT_NE(values, nullptr);
  ASSERT_EQ(values->strings_size(), 2);
  EXPECT_EQ(values->strings()[0].value(), "ten");
  EXPECT_EQ(values->strings()[1].value(), "missing");
  EXPECT_EQ(GetAttributeOr<std::string>(replacements[0], "default_string", ""), "other");
}

TEST(LabelEncoderFusionPattern, RejectsSharedIntermediate) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("ai.onnx.ml", 4);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape1D(3));
  utils::RepeatedProtoField<AttributeProto> attributes;
  AddInts(attributes, "keys_int64s", {1});
  AddInts(attributes, "values_int64s", {2});
  AddInt(attributes, "default_int64", 0);
  builder.MakeNode("LabelEncoder", {"x"}, {"mid"}, "ai.onnx.ml", "first", attributes);
  builder.MakeNode("LabelEncoder", {"mid"}, {"y"}, "ai.onnx.ml", "second", attributes);
  builder.MakeNode("Identity", {"mid"}, {"extra"});
  builder.MakeOutput("y");
  builder.MakeOutput("extra");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LabelEncoderFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(LabelEncoderFusionPattern, UsesSchemaDefaultsWhenOmitted) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("ai.onnx.ml", 4);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape1D(3));
  utils::RepeatedProtoField<AttributeProto> first;
  AddInts(first, "keys_int64s", {1});
  AddInts(first, "values_int64s", {10});
  utils::RepeatedProtoField<AttributeProto> second;
  AddInts(second, "keys_int64s", {-1});
  AddStrings(second, "values_strings", {"first-default"});
  builder.MakeNode("LabelEncoder", {"x"}, {"mid"}, "ai.onnx.ml", "first", first);
  builder.MakeNode("LabelEncoder", {"mid"}, {"y"}, "ai.onnx.ml", "second", second);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LabelEncoderFusionPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  EXPECT_EQ(GetAttributeOr<std::string>(replacements[0], "default_string", ""), "first-default");
  const AttributeProto *values = FindAttribute(replacements[0], "values_strings");
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values->strings()[0].value(), "_Unused");
}

TEST(LabelEncoderFusionPattern, RejectsIncompatibleMiddleType) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("ai.onnx.ml", 4);
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape1D(3));
  utils::RepeatedProtoField<AttributeProto> first;
  AddInts(first, "keys_int64s", {1});
  AddStrings(first, "values_strings", {"one"});
  AddString(first, "default_string", "other");
  utils::RepeatedProtoField<AttributeProto> second;
  AddInts(second, "keys_int64s", {1});
  AddInts(second, "values_int64s", {2});
  AddInt(second, "default_int64", 0);
  builder.MakeNode("LabelEncoder", {"x"}, {"mid"}, "ai.onnx.ml", "first", first);
  builder.MakeNode("LabelEncoder", {"mid"}, {"y"}, "ai.onnx.ml", "second", second);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::LabelEncoderFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

} // namespace
} // namespace Test
