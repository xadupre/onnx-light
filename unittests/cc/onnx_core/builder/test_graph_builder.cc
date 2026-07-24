// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// Schema provider backed by the built-in ONNX operator schemas.
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

TEST(GraphBuilder, StartsEmpty) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  EXPECT_EQ(builder.Nodes().size(), 0u);
  EXPECT_EQ(builder.Inputs().size(), 0u);
  EXPECT_FALSE(builder.HasName("x"));
}

TEST(GraphBuilder, NamesAreNeverReused) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  EXPECT_TRUE(builder.HasName("x"));
  EXPECT_THROW(builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
               core::builder::BuilderError);
}

TEST(GraphBuilder, UniqueNameNeverCollides) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  const std::string a = builder.UniqueName("t");
  const std::string b = builder.UniqueName("t");
  EXPECT_NE(a, b);
  EXPECT_TRUE(builder.HasName(a));
  EXPECT_TRUE(builder.HasName(b));
}

TEST(GraphBuilder, MakeNodeResolvesOpsetAndInfersShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(builder.HasName(outputs[0]));

  // The default ONNX opset was resolved from the operator schemas.
  EXPECT_GT(builder.OpsetVersion(""), 0);

  // The output shape was inferred incrementally.
  ASSERT_TRUE(builder.HasShape(outputs[0]));
  const core::symbolic::SymTensor &z = builder.GetShape(outputs[0]);
  EXPECT_EQ(z.Shape().Rank(), 2u);
}

TEST(GraphBuilder, MakeNodeUsesProvidedOutputName) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"}, {"z"});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0], "z");
}

TEST(GraphBuilder, MakeNodeRejectsUnknownInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  EXPECT_THROW(builder.MakeNode("Add", {"missing", "y"}), core::builder::BuilderError);
}

TEST(GraphBuilder, ExternalInitializerIsRecorded) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeExternalInitializer("w", core::symbolic::TensorType::kFloat, {4, 4}, "weights.bin", 0,
                                  64);
  EXPECT_TRUE(builder.HasName("w"));
  ASSERT_EQ(builder.Initializers().Size(), 1u);
  const TensorProto &init = builder.Initializers().At("w");
  EXPECT_EQ(init.data_location(), TensorProto::DataLocation::EXTERNAL);
}

TEST(GraphBuilder, ToModelProducesGraphWithOpset) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"});
  builder.MakeOutput(outputs[0]);

  const ModelProto model = builder.ToModel();
  EXPECT_GT(model.ir_version(), 0);
  EXPECT_EQ(model.graph().node().size(), 1);
  ASSERT_GT(model.opset_import().size(), 0);
}

TEST(GraphBuilder, ToFunctionRejectsInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeExternalInitializer("w", core::symbolic::TensorType::kFloat, {2, 2}, "w.bin", 0, 16);
  EXPECT_THROW(builder.ToFunction("custom"), core::builder::BuilderError);
}

TEST(GraphBuilder, ExplicitOpsetIsPreserved) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 17);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Add", {"x", "y"});
  EXPECT_EQ(builder.OpsetVersion(""), 17);
}

TEST(GraphBuilder, InputOutputFromProto) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto vi;
  vi.set_name("x");
  core::symbolic::SymTensorToValueInfo(
      core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
      vi);
  builder.MakeInput(vi);
  EXPECT_TRUE(builder.HasName("x"));
  ASSERT_EQ(builder.Inputs().size(), 1u);
  EXPECT_EQ(builder.Inputs()[0].name().value(), "x");
  EXPECT_TRUE(builder.HasShape("x"));

  ValueInfoProto out;
  out.set_name("x");
  builder.MakeOutput(out);
  ASSERT_EQ(builder.Outputs().size(), 1u);
  EXPECT_EQ(builder.Outputs()[0].name().value(), "x");
}

TEST(GraphBuilder, ToStringDescribesContent) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"});
  builder.MakeOutput(outputs[0]);

  const std::string text = builder.ToString();
  EXPECT_NE(text.find("GraphBuilder(name=g)"), std::string::npos);
  EXPECT_NE(text.find("inputs (2)"), std::string::npos);
  EXPECT_NE(text.find("nodes (1)"), std::string::npos);
  EXPECT_NE(text.find("Add("), std::string::npos);
}

TEST(GraphBuilder, LocalFunctionIsANestedBuilder) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  fct.MakeInput("b", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = fct.MakeNode("Add", {"a", "b"});
  fct.MakeOutput(outputs[0]);

  EXPECT_TRUE(builder.HasLocalFunction("MyFct"));
  EXPECT_EQ(builder.LocalFunction("MyFct").Nodes().size(), 1u);
  EXPECT_THROW(builder.MakeLocalFunction("MyFct"), core::builder::BuilderError);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> top = builder.MakeNode("MyFct", {"x", "z"}, {}, "custom");
  builder.MakeOutput(top[0]);

  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1);
  EXPECT_EQ(model.functions(0).name().value(), "MyFct");
}

TEST(GraphBuilder, SubgraphIsANestedBuilder) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  EXPECT_TRUE(builder.HasSubgraph("body"));
  EXPECT_EQ(builder.Subgraph("body").Inputs().size(), 1u);
  EXPECT_THROW(builder.MakeSubgraph("body"), core::builder::BuilderError);
}

} // namespace Test
