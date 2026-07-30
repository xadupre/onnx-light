// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include "onnx_helper.h"
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

TEST(GraphBuilder, MakeNodeMaintainsTagsAndReuseIncrementally) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> shape_out = builder.MakeNode("Shape", {"x"});
  ASSERT_EQ(shape_out.size(), 1u);

  // The value / node tags ("shape_tag") are kept up to date as nodes are added:
  // the output of Shape carries the "shape" tag right after MakeNode.
  const auto &value_tags = builder.Compute().ValueTags();
  auto it = value_tags.find(shape_out[0]);
  ASSERT_NE(it, value_tags.end());
  EXPECT_EQ(it->second, "shape");

  // In-place reuse is also computed incrementally: one entry per node so far.
  EXPECT_EQ(builder.Compute().Size(), builder.Nodes().size());

  const std::vector<std::string> abs_out = builder.MakeNode("Abs", {"x"});
  ASSERT_EQ(abs_out.size(), 1u);
  EXPECT_EQ(builder.Compute().Size(), builder.Nodes().size());
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

TEST(GraphBuilder, MakeNodeAcceptsRepeatedProtoFieldAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &alpha = attributes.add();
  alpha.set_name("alpha");
  alpha.set_type(AttributeProto::AttributeType::FLOAT);
  alpha.set_f(0.25f);

  const std::vector<std::string> outputs =
      builder.MakeNode("LeakyRelu", {"x"}, {}, "", "", attributes);
  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &node = builder.Nodes()[0];
  ASSERT_EQ(node.attribute().size(), 1);
  EXPECT_EQ(node.attribute()[0].name().value(), "alpha");
  EXPECT_FLOAT_EQ(node.attribute()[0].f(), 0.25f);
}

TEST(GraphBuilder, ExternalInitializerIsRecorded) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeExternalInitializer("w", core::symbolic::TensorType::kFloat, {4, 4}, "weights.bin", 0,
                                  64);
  EXPECT_TRUE(builder.HasName("w"));
  ASSERT_EQ(builder.Initializers().size(), 1u);
  const TensorProto &init = builder.Initializers()[0];
  EXPECT_EQ(init.name().value(), "w");
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

TEST(GraphBuilder, RemoveUnusedNodesDropsDeadEnds) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> used = builder.MakeNode("Add", {"x", "y"});
  // Two chained nodes that no graph output depends on: both are dead ends.
  const std::vector<std::string> dead = builder.MakeNode("Mul", {"x", "y"});
  builder.MakeNode("Neg", {dead[0]});
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.RemoveUnusedNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Add");
  // A second pass has nothing left to remove.
  EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
}

TEST(GraphBuilder, RemoveUnusedNodesKeepsSharedProducer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A shared producer feeding one used and one unused consumer must be kept.
  const std::vector<std::string> shared = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> used = builder.MakeNode("Abs", {shared[0]});
  builder.MakeNode("Abs", {shared[0]}); // dead-end consumer of the shared value
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.RemoveUnusedNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Abs");
}

TEST(GraphBuilder, RemoveUnusedNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A nested subgraph with one live and one dead-end node.
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> body_used = body.MakeNode("Neg", {"a"});
  body.MakeNode("Abs", {"a"}); // dead end inside the subgraph
  body.MakeOutput(body_used[0]);

  const std::vector<std::string> used = builder.MakeNode("Identity", {"x"});
  builder.MakeOutput(used[0]);

  // The removal descends into the subgraph and prunes its dead-end node.
  EXPECT_EQ(builder.RemoveUnusedNodes(), 1u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Neg");
}

TEST(GraphBuilder, RemoveDuplicateInitializersCollapsesEqual) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  // Two initializers with byte-for-byte identical content.
  builder.MakeInitializer(MakeInitializer<float>("w1", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  builder.MakeInitializer(MakeInitializer<float>("w2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));

  const std::vector<std::string> a = builder.MakeNode("Add", {"x", "w1"});
  const std::vector<std::string> b = builder.MakeNode("Add", {"x", "w2"});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(b[0]);

  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].name().value(), "w1");
  // The reference to the dropped duplicate is rewritten to the surviving name.
  EXPECT_EQ(builder.Nodes()[1].input(1), "w1");
  // A second pass has nothing left to collapse.
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
}

TEST(GraphBuilder, RemoveDuplicateInitializersKeepsDistinctContent) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  // Same shape and type but different data: not duplicates.
  builder.MakeInitializer(MakeInitializer<float>("w1", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  builder.MakeInitializer(MakeInitializer<float>("w2", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}));

  const std::vector<std::string> a = builder.MakeNode("Add", {"x", "w1"});
  const std::vector<std::string> b = builder.MakeNode("Add", {"x", "w2"});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(b[0]);

  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
  EXPECT_EQ(builder.Initializers().size(), 2u);
}

TEST(GraphBuilder, RemoveDuplicateInitializersRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  body.MakeInitializer(MakeInitializer<float>("c1", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  body.MakeInitializer(MakeInitializer<float>("c2", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  const std::vector<std::string> s = body.MakeNode("Add", {"a", "c2"});
  body.MakeOutput(s[0]);

  // The dedup descends into the subgraph and collapses its duplicate.
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Initializers().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Initializers()[0].name().value(), "c1");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input(1), "c1");
}

TEST(GraphBuilder, RemoveDuplicateInitializersCollapsesAcrossScopes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  // An initializer in the enclosing graph.
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  const std::vector<std::string> a = builder.MakeNode("Add", {"x", "w"});
  builder.MakeOutput(a[0]);

  // The subgraph declares its own initializer with the same content. Because a
  // subgraph body sees the enclosing scope, the duplicate is collapsed onto the
  // enclosing initializer and its reference is rewritten to "w".
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("b", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  body.MakeInitializer(MakeInitializer<float>("c", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  const std::vector<std::string> s = body.MakeNode("Add", {"b", "c"});
  body.MakeOutput(s[0]);

  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].name().value(), "w");
  EXPECT_EQ(builder.Subgraph("body").Initializers().size(), 0u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input(1), "w");
}

} // namespace Test
