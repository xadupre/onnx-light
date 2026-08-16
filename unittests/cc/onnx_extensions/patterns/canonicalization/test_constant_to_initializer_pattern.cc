// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/constant_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

void AddConstant(core::builder::GraphBuilder &builder, const std::string &output) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("value");
  attribute.set_type(AttributeProto::AttributeType::TENSOR);
  *attribute.mutable_t() = MakeInitializer<float>("", {2}, {1.0f, 2.0f});
  builder.MakeNode("Constant", {}, {output}, "", "", attributes);
}

TEST(ConstantToInitializerPattern, MatchesMaterialisableConstant) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  AddConstant(builder, "cst");
  builder.MakeNode("Identity", {"cst"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConstantToInitializerPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 1u);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].output()[0].value(), "cst");
}

TEST(ConstantToInitializerPattern, ProducesInitializerAndIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  AddConstant(builder, "cst");
  builder.MakeNode("Identity", {"cst"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::ConstantToInitializerPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  const auto &initializers = builder.Initializers();
  ASSERT_EQ(initializers.size(), 1u);
  EXPECT_EQ(initializers[0].data_type(), TensorProto::DataType::FLOAT);
  bool has_constant = false;
  for (const auto &node : builder.Nodes()) {
    if (node.op_type().value() == "Constant") {
      has_constant = true;
    }
  }
  EXPECT_FALSE(has_constant);
}

TEST(ConstantToInitializerPattern, RejectsNonConstantNode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, []() {
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(2));
    return shape;
  }());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ConstantToInitializerPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "candidate is not a default-domain Constant");
}

} // namespace
} // namespace Test
