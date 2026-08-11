// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, /*init_doc=*/false);
  };
}

core::symbolic::SymShape MakeShape(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t d : dims) {
    shape.PushBack(core::symbolic::SymDim(d));
  }
  return shape;
}

} // namespace

TEST(GraphGraph, IndexesPredecessorsAndSuccessors) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> z = builder.MakeNode("Add", {"x", "y"}, {"z"});
  const std::vector<std::string> out1 = builder.MakeNode("Mul", {"z", "x"}, {"out1"});
  const std::vector<std::string> out2 = builder.MakeNode("Sub", {"z", "y"}, {"out2"});
  builder.MakeOutput("out1");
  builder.MakeOutput("out2");

  core::builder::GraphGraph graph(builder);

  // The producer of a node output is the node itself; graph inputs have none.
  ASSERT_NE(graph.NodeBefore("z"), nullptr);
  EXPECT_EQ(graph.NodeBefore("z")->op_type().value(), "Add");
  EXPECT_EQ(graph.NodeBefore("x"), nullptr);
  EXPECT_EQ(graph.NodeBefore("missing"), nullptr);

  // "z" is consumed by both Mul and Sub.
  const std::vector<const NodeProto *> &z_consumers = graph.NextNodes("z");
  ASSERT_EQ(z_consumers.size(), 2u);
  EXPECT_EQ(z_consumers[0]->op_type().value(), "Mul");
  EXPECT_EQ(z_consumers[1]->op_type().value(), "Sub");
  EXPECT_TRUE(graph.NextNodes("missing").empty());

  // Positions follow the insertion order of the nodes.
  EXPECT_EQ(graph.Position(*graph.NodeBefore("z")), 0u);
  EXPECT_EQ(graph.Position(*graph.NodeBefore("out2")), 2u);
}

TEST(GraphGraph, UsageQueries) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Add", {"x", "y"}, {"z"});
  builder.MakeNode("Mul", {"z", "x"}, {"out1"});
  builder.MakeNode("Sub", {"z", "y"}, {"out2"});
  builder.MakeOutput("out1");
  builder.MakeOutput("out2");

  core::builder::GraphGraph graph(builder);

  EXPECT_TRUE(graph.IsOutput("out1"));
  EXPECT_FALSE(graph.IsOutput("z"));

  // "x" feeds Add and Mul, "z" feeds Mul and Sub: both used more than once.
  EXPECT_TRUE(graph.IsUsed("x"));
  EXPECT_TRUE(graph.IsUsedMoreThanOnce("x"));
  EXPECT_TRUE(graph.IsUsedMoreThanOnce("z"));
  // "y" feeds Add and Sub.
  EXPECT_TRUE(graph.IsUsedMoreThanOnce("y"));

  // An output is considered used (and used more than once) even with a single
  // or no consumer.
  EXPECT_TRUE(graph.IsUsed("out1"));
  EXPECT_TRUE(graph.IsUsedMoreThanOnce("out1"));

  // Nothing is captured by a subgraph here.
  EXPECT_FALSE(graph.IsUsedBySubgraph("z"));

  // An unknown value is neither used nor an output.
  EXPECT_FALSE(graph.IsUsed("missing"));
  EXPECT_FALSE(graph.IsUsedMoreThanOnce("missing"));
}

TEST(GraphGraph, ShapeAndTypeQueries) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Relu", {"x"}, {"y"});

  core::builder::GraphGraph graph(builder);

  ASSERT_TRUE(graph.HasShape("x"));
  EXPECT_EQ(graph.GetShape("x").Shape().Rank(), 2u);
  EXPECT_TRUE(graph.HasType("x"));
  EXPECT_EQ(graph.GetType("x"), core::symbolic::TensorType::kFloat);
  EXPECT_FALSE(graph.HasShape("missing"));
  EXPECT_FALSE(graph.HasType("missing"));
}

TEST(GraphGraph, ConstantInitializerQueries) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  builder.MakeInitializer(MakeInitializer<float>("scalar", {1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("matrix", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));

  core::builder::GraphGraph graph(builder);

  EXPECT_TRUE(graph.IsConstant("scalar"));
  EXPECT_FALSE(graph.IsConstant("x"));

  // "scalar" is a shape-(1,) constant equal to 2.
  EXPECT_TRUE(graph.IsConstantScalar("scalar"));
  EXPECT_TRUE(graph.IsConstantScalar("scalar", 2.0, /*broadcast=*/false));
  EXPECT_FALSE(graph.IsConstantScalar("scalar", 3.0, /*broadcast=*/false));

  // A rank-2 constant is not a scalar, even with broadcasting.
  EXPECT_FALSE(graph.IsConstantScalar("matrix"));
  EXPECT_FALSE(graph.IsConstantScalar("matrix", /*broadcast=*/true));

  // GetComputedConstant returns the underlying initializer tensor.
  const TensorProto *tensor = graph.GetComputedConstant("scalar");
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->name().value(), "scalar");
  EXPECT_EQ(graph.GetComputedConstant("x"), nullptr);
}

TEST(GraphGraph, ConstantScalarBroadcast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("ones", {1, 1}, {5.0f}));

  core::builder::GraphGraph graph(builder);

  // Shape (1, 1): only a scalar under the broadcast rule.
  EXPECT_FALSE(graph.IsConstantScalar("ones", /*broadcast=*/false));
  EXPECT_TRUE(graph.IsConstantScalar("ones", /*broadcast=*/true));
  EXPECT_TRUE(graph.IsConstantScalar("ones", 5.0, /*broadcast=*/true));
  EXPECT_FALSE(graph.IsConstantScalar("ones", 6.0, /*broadcast=*/true));
}

TEST(GraphGraph, ConstantNodeValueFloat) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &value = attributes.add();
  value.set_name("value_float");
  value.set_type(AttributeProto::AttributeType::FLOAT);
  value.set_f(7.0f);
  const std::vector<std::string> outputs =
      builder.MakeNode("Constant", {}, {"c"}, "", "", attributes);
  ASSERT_EQ(outputs.size(), 1u);

  core::builder::GraphGraph graph(builder);

  EXPECT_TRUE(graph.IsConstant("c"));
  // A value_float Constant is a scalar with value 7; it has no TensorProto.
  EXPECT_TRUE(graph.IsConstantScalar("c"));
  EXPECT_TRUE(graph.IsConstantScalar("c", 7.0, /*broadcast=*/false));
  EXPECT_FALSE(graph.IsConstantScalar("c", 8.0, /*broadcast=*/false));
  EXPECT_EQ(graph.GetComputedConstant("c"), nullptr);
}

TEST(GraphGraph, ConstantNodeValueTensor) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &value = attributes.add();
  value.set_name("value");
  value.set_type(AttributeProto::AttributeType::TENSOR);
  *value.add_t() = MakeInitializer<int64_t>("", {1}, {42});
  const std::vector<std::string> outputs =
      builder.MakeNode("Constant", {}, {"c"}, "", "", attributes);
  ASSERT_EQ(outputs.size(), 1u);

  core::builder::GraphGraph graph(builder);

  EXPECT_TRUE(graph.IsConstant("c"));
  const TensorProto *tensor = graph.GetComputedConstant("c");
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->data_type(), TensorProto::DataType::INT64);
  EXPECT_TRUE(graph.IsConstantScalar("c", 42.0, /*broadcast=*/false));
}

TEST(GraphGraph, SetComputedConstantIsCached) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  const std::vector<std::string> y = builder.MakeNode("Neg", {"x"}, {"y"});

  core::builder::GraphGraph graph(builder);
  EXPECT_EQ(graph.GetComputedConstant("y"), nullptr);

  graph.SetComputedConstant("y", MakeInitializer<float>("y", {1}, {3.0f}));
  const TensorProto *tensor = graph.GetComputedConstant("y");
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->name().value(), "y");
}

} // namespace Test
