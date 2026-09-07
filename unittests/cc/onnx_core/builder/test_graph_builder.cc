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

TEST(GraphBuilder, MaintainsConstantInfoIncrementally) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  TensorProto initializer;
  initializer.set_name("weight");
  initializer.set_data_type(TensorProto::DataType::FLOAT);
  initializer.add_dims(1);
  initializer.add_float_data(1.0f);
  builder.MakeInitializer(initializer);
  EXPECT_TRUE(builder.Compute().IsConstantValue("weight"));

  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &value = attributes.add();
  value.set_name("value_float");
  value.set_type(AttributeProto::AttributeType::FLOAT);
  value.set_f(1.0f);
  const std::vector<std::string> outputs =
      builder.MakeNode("Constant", {}, {"constant"}, "", "", attributes);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(builder.Compute().IsConstantValue(outputs[0]));
  EXPECT_TRUE(builder.Compute().NodeConstant(0));
  EXPECT_EQ(builder.Compute().NodeConstant().at(0), core::compute::ConstantInfo::kConstant);
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

TEST(GraphBuilder, BuildsTranslatedLinearModel) {
  core::builder::GraphBuilder builder("linear", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  TensorProto weight;
  weight.set_name("W");
  weight.set_data_type(TensorProto::DataType::FLOAT);
  weight.add_dims(3);
  weight.add_float_data(1.0f);
  weight.add_float_data(2.0f);
  weight.add_float_data(3.0f);
  builder.MakeInitializer(weight);

  TensorProto bias;
  bias.set_name("B");
  bias.set_data_type(TensorProto::DataType::FLOAT);
  bias.add_dims(3);
  bias.add_float_data(0.5f);
  bias.add_float_data(0.5f);
  bias.add_float_data(0.5f);
  builder.MakeInitializer(bias);

  const std::vector<std::string> multiplied = builder.MakeNode("Mul", {"X", "W"}, {"XW"});
  const std::vector<std::string> added = builder.MakeNode("Add", {multiplied[0], "B"}, {"Y"});
  builder.MakeOutput(added[0], core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.graph().node().size(), 2);
  EXPECT_EQ(model.graph().node()[0].op_type().value(), "Mul");
  EXPECT_EQ(model.graph().node()[1].op_type().value(), "Add");
  EXPECT_EQ(model.graph().initializer().size(), 2);
  EXPECT_EQ(model.graph().output()[0].name().value(), "Y");
}

TEST(GraphBuilder, ToFunctionRejectsInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeExternalInitializer("w", core::symbolic::TensorType::kFloat, {2, 2}, "w.bin", 0, 16);
  EXPECT_THROW(builder.ToFunction("custom"), core::builder::BuilderError);
}

TEST(GraphBuilder, ToFunctionPreservesFinalizedShapesAndMetadata) {
  core::builder::GraphBuilder builder("F", SchemaLookup());
  ValueInfoProto input;
  input.set_name("X");
  input.set_doc_string("Input declaration");
  auto *input_type = input.mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(TensorProto::DataType::FLOAT);
  input_type->mutable_shape()->add_dim();
  input_type->mutable_shape()->add_dim()->set_dim_value(3);
  builder.MakeInput(input);
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("Neg", {"X"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"out"});
  builder.MakeNode("Shape", {"intermediate"}, {"shape"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"intermediate"}, {name + "_result"});
    branch.MakeOutput(name + "_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  for (const std::string name : {"out", "shape", "selected"}) {
    builder.MakeOutput(name);
  }

  const FunctionProto function = builder.ToFunction("custom");
  const GraphProto graph = builder.ToGraph();
  EXPECT_EQ(function.name(), "F");
  EXPECT_EQ(function.domain(), "custom");
  ASSERT_EQ(function.node().size(), graph.node().size());
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
    EXPECT_EQ(function.node(i).SerializeAsString(), graph.node(i).SerializeAsString());
  }
  EXPECT_EQ(function.node(1).op_type(), "Shape");
  ASSERT_EQ(function.value_info().size(),
            graph.input().size() + graph.output().size() + graph.value_info().size());
  const auto check_value_info = [&](const auto &values) {
    for (const ValueInfoProto &expected : values) {
      bool found = false;
      for (const ValueInfoProto &actual : function.value_info()) {
        if (actual.name() == expected.name()) {
          EXPECT_EQ(actual.SerializeAsString(), expected.SerializeAsString());
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << expected.name();
    }
  };
  check_value_info(graph.input());
  check_value_info(graph.output());
  check_value_info(graph.value_info());
  EXPECT_EQ(builder.ToFunction("custom").SerializeAsString(), function.SerializeAsString());
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

TEST(GraphBuilder, RemoveIdentityNodesRewiresConsumers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Identity forwards x; a downstream node consumes the forwarded value.
  const std::vector<std::string> ident = builder.MakeNode("Identity", {"x"});
  const std::vector<std::string> used = builder.MakeNode("Neg", {ident[0]});
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  // The consumer now reads directly from the identity's input.
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  // A second pass has nothing left to remove.
  EXPECT_EQ(builder.RemoveIdentityNodes(), 0u);
}

TEST(GraphBuilder, RemoveIdentityNodesCollapsesChains) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A chain of two identities feeding a live consumer must collapse onto x.
  const std::vector<std::string> i1 = builder.MakeNode("Identity", {"x"});
  const std::vector<std::string> i2 = builder.MakeNode("Identity", {i1[0]});
  const std::vector<std::string> used = builder.MakeNode("Neg", {i2[0]});
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
}

TEST(GraphBuilder, RemoveIdentityNodesKeepsGraphOutputIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // The identity output is a declared graph output, so the node is kept.
  const std::vector<std::string> ident = builder.MakeNode("Identity", {"x"});
  builder.MakeOutput(ident[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
}

TEST(GraphBuilder, RemoveIdentityNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A nested subgraph with an interior identity to collapse.
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> inner = body.MakeNode("Identity", {"a"});
  const std::vector<std::string> body_used = body.MakeNode("Neg", {inner[0]});
  body.MakeOutput(body_used[0]);

  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input(0), "a");
}

TEST(GraphBuilder, RemoveDuplicateNodesCollapsesEqual) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Two Neg nodes computing the same value feed a shared consumer.
  const std::vector<std::string> a = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> b = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> sum = builder.MakeNode("Add", {a[0], b[0]});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Add");
  // Both Add inputs now read from the surviving Neg.
  EXPECT_EQ(builder.Nodes()[1].input(0), a[0]);
  EXPECT_EQ(builder.Nodes()[1].input(1), a[0]);
  // A second pass has nothing left to collapse.
  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
}

TEST(GraphBuilder, RemoveDuplicateNodesCollapsesBranches) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Two identical branches: once the first level collapses, the second level
  // becomes duplicated too and collapses in the same pass.
  const std::vector<std::string> a1 = builder.MakeNode("Mul", {"x", "x"});
  const std::vector<std::string> b1 = builder.MakeNode("Mul", {"x", "x"});
  const std::vector<std::string> a2 = builder.MakeNode("Relu", {a1[0]});
  const std::vector<std::string> b2 = builder.MakeNode("Relu", {b1[0]});
  const std::vector<std::string> sum = builder.MakeNode("Add", {a2[0], b2[0]});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Mul");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Relu");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Add");
  EXPECT_EQ(builder.Nodes()[2].input(0), a2[0]);
  EXPECT_EQ(builder.Nodes()[2].input(1), a2[0]);
}

TEST(GraphBuilder, RemoveDuplicateNodesKeepsDistinctAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const auto leaky = [&](float alpha) {
    utils::RepeatedProtoField<AttributeProto> attributes;
    AttributeProto &attr = attributes.add();
    attr.set_name("alpha");
    attr.set_type(AttributeProto::AttributeType::FLOAT);
    attr.set_f(alpha);
    return builder.MakeNode("LeakyRelu", {"x"}, {}, "", "", attributes);
  };
  const std::vector<std::string> a = leaky(0.1f);
  const std::vector<std::string> b = leaky(0.2f);
  const std::vector<std::string> sum = builder.MakeNode("Add", {a[0], b[0]});
  builder.MakeOutput(sum[0]);

  // Different attribute values mean different computations: nothing collapses.
  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
}

TEST(GraphBuilder, RemoveDuplicateNodesKeepsGraphOutputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Both duplicates are declared graph outputs, so neither can be dropped.
  const std::vector<std::string> a = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> b = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(b[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
}

TEST(GraphBuilder, RemoveDuplicateNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> n1 = body.MakeNode("Neg", {"a"});
  const std::vector<std::string> n2 = body.MakeNode("Neg", {"a"});
  const std::vector<std::string> s = body.MakeNode("Add", {n1[0], n2[0]});
  body.MakeOutput(s[0]);

  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 2u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].op_type().value(), "Add");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].input(0), n1[0]);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].input(1), n1[0]);
}

TEST(GraphBuilder, MoveShapeAndSizeNodesHoistsToProducer) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> p = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> a = builder.MakeNode("Add", {p[0], p[0]});
  // Shape reads p but is inserted after an unrelated node; the chained Size
  // reads the Shape output, so both must move right after Neg in one pass.
  const std::vector<std::string> sh = builder.MakeNode("Shape", {p[0]});
  const std::vector<std::string> sz = builder.MakeNode("Size", {sh[0]});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(sz[0]);

  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 4u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Shape");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Size");
  EXPECT_EQ(builder.Nodes()[3].op_type().value(), "Add");
  // A second pass leaves the already-tightened graph untouched.
  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 0u);
}

TEST(GraphBuilder, MoveShapeAndSizeNodesKeepsGraphInputReader) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Shape reads a graph input (no producing node), so it stays in place.
  const std::vector<std::string> n = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> sh = builder.MakeNode("Shape", {"x"});
  builder.MakeOutput(n[0]);
  builder.MakeOutput(sh[0]);

  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Shape");
}

TEST(GraphBuilder, MoveShapeAndSizeNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> p = body.MakeNode("Neg", {"a"});
  const std::vector<std::string> u = body.MakeNode("Add", {p[0], p[0]});
  const std::vector<std::string> sh = body.MakeNode("Shape", {p[0]});
  body.MakeOutput(u[0]);
  body.MakeOutput(sh[0]);

  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 3u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].op_type().value(), "Shape");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[2].op_type().value(), "Add");
}

TEST(GraphBuilder, InlineLocalFunctionsExpandsCallSite) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  // A local function computing Neg(Add(a, b)).
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  fct.MakeInput("b", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> sum = fct.MakeNode("Add", {"a", "b"});
  const std::vector<std::string> neg = fct.MakeNode("Neg", {sum[0]});
  fct.MakeOutput(neg[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> call = builder.MakeNode("MyFct", {"x", "z"}, {}, "custom");
  builder.MakeOutput(call[0]);

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  // The call node is replaced by the two body nodes, and the fully inlined
  // function definition is dropped.
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Add");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Neg");
  // Formal inputs are rewired to the call inputs.
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[0].input(1), "z");
  // The formal output is rewired to the call output.
  EXPECT_EQ(builder.Nodes()[1].output(0), call[0]);
  EXPECT_FALSE(builder.HasLocalFunction("MyFct"));
  // A second pass has nothing left to inline.
  EXPECT_EQ(builder.InlineLocalFunctions(), 0u);
}

TEST(GraphBuilder, InlineLocalFunctionsRewiresIntermediates) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> inner = fct.MakeNode("Neg", {"a"});
  const std::vector<std::string> outer = fct.MakeNode("Neg", {inner[0]});
  fct.MakeOutput(outer[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> call = builder.MakeNode("MyFct", {"x"}, {}, "custom");
  builder.MakeOutput(call[0]);

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  // The intermediate value is renamed to a fresh, unused name that chains the
  // two body nodes together.
  const std::string intermediate = builder.Nodes()[0].output(0);
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[1].input(0), intermediate);
  EXPECT_NE(intermediate, "x");
  EXPECT_EQ(builder.Nodes()[1].output(0), call[0]);
}

TEST(GraphBuilder, InlineLocalFunctionsExpandsNestedCalls) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  // Inner function Neg(a).
  core::builder::GraphBuilder &inner = builder.MakeLocalFunction("Inner", "custom");
  inner.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> inner_out = inner.MakeNode("Neg", {"a"});
  inner.MakeOutput(inner_out[0]);

  // Outer function calls Inner and forwards its result.
  core::builder::GraphBuilder &outer = builder.MakeLocalFunction("Outer", "custom");
  outer.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> nested = outer.MakeNode("Inner", {"a"}, {}, "custom");
  outer.MakeOutput(nested[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> call = builder.MakeNode("Outer", {"x"}, {}, "custom");
  builder.MakeOutput(call[0]);

  // Both the outer call and the nested inner call are expanded.
  EXPECT_EQ(builder.InlineLocalFunctions(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[0].output(0), call[0]);
  EXPECT_FALSE(builder.HasLocalFunction("Outer"));
  EXPECT_FALSE(builder.HasLocalFunction("Inner"));
}

TEST(GraphBuilder, InlineLocalFunctionsKeepsUncalledFunction) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> out = fct.MakeNode("Neg", {"a"});
  fct.MakeOutput(out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  // No call site, so nothing is inlined and the definition is left in place.
  EXPECT_EQ(builder.InlineLocalFunctions(), 0u);
  EXPECT_TRUE(builder.HasLocalFunction("MyFct"));
}

TEST(GraphBuilder, InlineLocalFunctionsResolvesAttributeReferences) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  // A local function whose body forwards a function attribute to a body node.
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("Scaled", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto ref;
  ref.set_name("alpha");
  ref.set_ref_attr_name("alpha");
  ref.set_type(AttributeProto::AttributeType::FLOAT);
  utils::RepeatedProtoField<AttributeProto> body_attrs;
  body_attrs.push_back(ref);
  const std::vector<std::string> scaled = fct.MakeNode("LeakyRelu", {"a"}, {}, "", "", body_attrs);
  fct.MakeOutput(scaled[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto alpha;
  alpha.set_name("alpha");
  alpha.set_type(AttributeProto::AttributeType::FLOAT);
  alpha.set_f(0.25f);
  utils::RepeatedProtoField<AttributeProto> call_attrs;
  call_attrs.push_back(alpha);
  const std::vector<std::string> call =
      builder.MakeNode("Scaled", {"x"}, {}, "custom", "", call_attrs);
  builder.MakeOutput(call[0]);

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "LeakyRelu");
  ASSERT_EQ(builder.Nodes()[0].attribute().size(), 1);
  const AttributeProto &resolved = builder.Nodes()[0].attribute()[0];
  EXPECT_EQ(resolved.name().value(), "alpha");
  EXPECT_TRUE(resolved.ref_attr_name().empty());
  EXPECT_FLOAT_EQ(resolved.f(), 0.25f);
}

TEST(GraphBuilder, InlineLocalFunctionsIncludeSelectsFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  core::builder::GraphBuilder &keep = builder.MakeLocalFunction("Keep", "custom");
  keep.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_out = keep.MakeNode("Neg", {"a"});
  keep.MakeOutput(keep_out[0]);

  core::builder::GraphBuilder &drop = builder.MakeLocalFunction("Drop", "custom");
  drop.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> drop_out = drop.MakeNode("Neg", {"a"});
  drop.MakeOutput(drop_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_call = builder.MakeNode("Keep", {"x"}, {}, "custom");
  const std::vector<std::string> drop_call = builder.MakeNode("Drop", {keep_call[0]}, {}, "custom");
  builder.MakeOutput(drop_call[0]);

  // Only "Drop" is inlined; the "Keep" call and its definition are untouched.
  EXPECT_EQ(builder.InlineLocalFunctions({{"custom", "Drop"}}), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Keep"));
  EXPECT_FALSE(builder.HasLocalFunction("Drop"));
  bool has_keep_call = false;
  for (const auto &node : builder.Nodes()) {
    if (node.op_type().value() == "Keep") {
      has_keep_call = true;
    }
  }
  EXPECT_TRUE(has_keep_call);
}

TEST(GraphBuilder, InlineLocalFunctionsExcludeSkipsFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  core::builder::GraphBuilder &keep = builder.MakeLocalFunction("Keep", "custom");
  keep.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_out = keep.MakeNode("Neg", {"a"});
  keep.MakeOutput(keep_out[0]);

  core::builder::GraphBuilder &drop = builder.MakeLocalFunction("Drop", "custom");
  drop.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> drop_out = drop.MakeNode("Neg", {"a"});
  drop.MakeOutput(drop_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_call = builder.MakeNode("Keep", {"x"}, {}, "custom");
  const std::vector<std::string> drop_call = builder.MakeNode("Drop", {keep_call[0]}, {}, "custom");
  builder.MakeOutput(drop_call[0]);

  // Everything except "Keep" is inlined.
  EXPECT_EQ(builder.InlineLocalFunctions({}, {{"custom", "Keep"}}), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Keep"));
  EXPECT_FALSE(builder.HasLocalFunction("Drop"));
}

TEST(GraphBuilder, InlineLocalFunctionsIncludeWildcardsDomainByName) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  core::builder::GraphBuilder &keep = builder.MakeLocalFunction("Keep", "custom");
  keep.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_out = keep.MakeNode("Neg", {"a"});
  keep.MakeOutput(keep_out[0]);

  core::builder::GraphBuilder &drop = builder.MakeLocalFunction("Drop", "custom");
  drop.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> drop_out = drop.MakeNode("Neg", {"a"});
  drop.MakeOutput(drop_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_call = builder.MakeNode("Keep", {"x"}, {}, "custom");
  const std::vector<std::string> drop_call = builder.MakeNode("Drop", {keep_call[0]}, {}, "custom");
  builder.MakeOutput(drop_call[0]);

  // An empty domain matches the function by name regardless of its domain.
  EXPECT_EQ(builder.InlineLocalFunctions({{"", "Drop"}}), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Keep"));
  EXPECT_FALSE(builder.HasLocalFunction("Drop"));
}

TEST(GraphBuilder, InlineLocalFunctionsIncludeWildcardsNameByDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup());

  core::builder::GraphBuilder &fa = builder.MakeLocalFunction("First", "custom");
  fa.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> fa_out = fa.MakeNode("Neg", {"a"});
  fa.MakeOutput(fa_out[0]);

  core::builder::GraphBuilder &fb = builder.MakeLocalFunction("Second", "other");
  fb.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> fb_out = fb.MakeNode("Neg", {"a"});
  fb.MakeOutput(fb_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> ca = builder.MakeNode("First", {"x"}, {}, "custom");
  const std::vector<std::string> cb = builder.MakeNode("Second", {ca[0]}, {}, "other");
  builder.MakeOutput(cb[0]);

  // An empty name matches every function in the given domain only.
  EXPECT_EQ(builder.InlineLocalFunctions({{"custom", ""}}), 1u);
  EXPECT_FALSE(builder.HasLocalFunction("First"));
  EXPECT_TRUE(builder.HasLocalFunction("Second"));
}

TEST(GraphBuilder, InlineLocalFunctionsRejectsIncludeAndExclude) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  EXPECT_THROW(builder.InlineLocalFunctions({{"", "A"}}, {{"", "B"}}), core::builder::BuilderError);
}

TEST(GraphBuilder, NativeLocalFunctionInfersBeforeFollowingNode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"X"}, {"result"}, "local");
  ASSERT_TRUE(builder.HasShape("result"));
  EXPECT_EQ(builder.GetShape("result").Shape(), MakeShape({2, 3}));
  builder.MakeNode("Add", {"result", "X"}, {"Y"});
  builder.MakeOutput("Y");
  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1u);
  ASSERT_GE(model.functions(0).value_info().size(), 2u);
  EXPECT_EQ(model.functions(0).value_info(0).name(), "a");
  EXPECT_TRUE(model.functions(0).value_info(0).has_type());
  EXPECT_EQ(model.graph().output(0).type().tensor_type().shape().dim_size(), 2);
  core::builder::GraphBuilder imported(model, SchemaLookup());
  EXPECT_EQ(imported.LocalFunction("F").GetShape("a").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(imported.GetShape("Y").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeLocalFunctionDefinitionsSurviveMovesAndEdits) {
  core::builder::GraphBuilder original("g", SchemaLookup());
  auto &function = original.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  original.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  original.MakeNode("F", {"X"}, {"first"}, "local");
  core::shapes::ShapesContext retained = original.Shapes();
  core::builder::GraphBuilder moved(std::move(original));
  moved.LocalFunction("F").MakeNode("Shape", {"a"}, {"shape"});
  moved.LocalFunction("F").MakeOutput("shape");
  const auto outputs = moved.MakeNode("F", {"X"}, {}, "local");
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(moved.GetShape(outputs[0]).Shape(), MakeShape({2, 3}));
  EXPECT_EQ(moved.GetShape(outputs[1]).Shape(), MakeShape({2}));
  // The copied context still owns the original, one-output definition.
  retained.ComputeShapeNode(MakeNode("F", {"X"}, {"old"}, "local"));
  EXPECT_EQ(retained.Get("old").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeGenericFunctionImportsAndSpecializesNestedMatMul) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.SetOpsetVersion("local", 1);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  seed.MakeInput("W", core::symbolic::TensorType::kFloat, MakeShape({3, 4}));
  ModelProto model = seed.ToModel();
  FunctionProto outer;
  outer.set_name("Outer");
  outer.set_domain("local");
  outer.add_input("a");
  outer.add_input("b");
  outer.add_output("r");
  outer.add_opset("", 18);
  outer.add_opset("local", 1);
  outer.add_node(MakeNode("Inner", {"a", "b"}, {"r"}, "local"));
  model.add_function(outer);
  FunctionProto inner = outer;
  inner.set_name("Inner");
  inner.ref_node().clear();
  inner.add_node(MakeNode("MatMul", {"a", "b"}, {"r"}));
  model.add_function(inner);
  model.mutable_graph()->add_node(MakeNode("Outer", {"X", "W"}, {"Y"}, "local"));
  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_FALSE(builder.LocalFunction("Inner").HasShape("a"));
  ASSERT_TRUE(builder.HasShape("Y"));
  EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 4}));
  builder.MakeNode("Add", {"Y", "Y"}, {"Z"});
  EXPECT_EQ(builder.GetShape("Z").Shape(), MakeShape({2, 4}));
}

TEST(GraphBuilder, NativeFunctionBindsRequiredAndDefaultAttributes) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.SetOpsetVersion("local", 1);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  ModelProto model = seed.ToModel();
  FunctionProto function;
  function.set_name("Convert");
  function.set_domain("local");
  function.add_input("a");
  function.add_output("r");
  function.add_opset("", 18);
  AttributeProto default_type;
  default_type.set_name("dtype");
  default_type.set_type(AttributeProto::AttributeType::INT);
  default_type.set_i(TensorProto::DataType::DOUBLE);
  function.add_attribute_proto(default_type);
  NodeProto cast = MakeNode("Cast", {"a"}, {"r"});
  AttributeProto reference;
  reference.set_name("to");
  reference.set_type(AttributeProto::AttributeType::INT);
  reference.set_ref_attr_name("dtype");
  cast.add_attribute(reference);
  function.add_node(cast);
  model.add_function(function);
  FunctionProto required = function;
  required.set_name("Required");
  required.ref_attribute_proto().clear();
  required.add_attribute("dtype");
  model.add_function(required);
  core::builder::GraphBuilder builder(model, SchemaLookup());
  builder.MakeNode("Convert", {"X"}, {"default"}, "local");
  EXPECT_EQ(builder.GetShape("default").Dtype(), core::symbolic::TensorType::kDouble);
  utils::RepeatedProtoField<AttributeProto> attributes;
  default_type.set_i(TensorProto::DataType::INT64);
  attributes.push_back(default_type);
  builder.MakeNode("Convert", {"X"}, {"explicit"}, "local", "", attributes);
  EXPECT_EQ(builder.GetShape("explicit").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_THROW(builder.MakeNode("Required", {"X"}, {"missing"}, "local"), std::invalid_argument);
  builder.MakeNode("Required", {"X"}, {"provided"}, "local", "", attributes);
  EXPECT_EQ(builder.GetShape("provided").Dtype(), core::symbolic::TensorType::kInt64);
  const ModelProto exported = builder.ToModel();
  ASSERT_EQ(exported.functions(0).attribute_proto().size(), 1u);
  EXPECT_EQ(exported.functions(0).node(0).attribute(0).ref_attr_name(), "dtype");
}

TEST(GraphBuilder, NativeFunctionRejectsRecursionAndWrongArity) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.SetOpsetVersion("local", 1);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  ModelProto model = seed.ToModel();
  FunctionProto function;
  function.set_name("Recursive");
  function.set_domain("local");
  function.add_opset("local", 1);
  function.add_input("a");
  function.add_output("r");
  function.add_node(MakeNode("Recursive", {"a"}, {"r"}, "local"));
  model.add_function(function);
  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_THROW(builder.MakeNode("Recursive", {"X"}, {"recursive"}, "local"), std::invalid_argument);
  EXPECT_THROW(builder.MakeNode("Recursive", {"X", "X"}, {"arity"}, "local"),
               std::invalid_argument);
}

TEST(GraphBuilder, NativeUnknownRankIsNotScalar) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto unknown;
  unknown.set_name("unknown");
  unknown.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  builder.MakeInput(unknown);
  EXPECT_FALSE(builder.HasShape("unknown"));
  builder.MakeInput("scalar", core::symbolic::TensorType::kFloat, MakeShape({}));
  ASSERT_TRUE(builder.HasShape("scalar"));
  EXPECT_EQ(builder.GetShape("scalar").Shape().Rank(), 0u);
  builder.MakeNode("MatMul", {"unknown", "unknown"}, {"deferred"});
  EXPECT_FALSE(builder.HasShape("deferred"));
  EXPECT_THROW(builder.MakeNode("MatMul", {"scalar", "scalar"}, {"invalid"}),
               std::invalid_argument);
}

TEST(GraphBuilder, NativeIfInfersCapturedOutputsBeforeCastLike) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"X"}, {"branch_result"});
    branch.MakeOutput("branch_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  ASSERT_TRUE(builder.HasShape("selected"));
  builder.MakeNode("CastLike", {"selected", "X"}, {"Y"});
  EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 3}));
  builder.MakeOutput("Y");
  core::builder::GraphBuilder imported(builder.ToModel(), SchemaLookup());
  EXPECT_EQ(imported.GetShape("selected").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeSubgraphRejectsAncestorShadowing) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  auto &branch = builder.MakeSubgraph("branch");
  EXPECT_THROW(branch.MakeNode("Identity", {"X"}, {"X"}), core::builder::BuilderError);
  EXPECT_THROW(branch.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
               core::builder::BuilderError);
  branch.MakeNode("Identity", {"X"}, {"Y"});
  EXPECT_EQ(branch.GetShape("Y").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeCleanupPrunesInitializersWithoutNodes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("public", core::symbolic::TensorType::kFloat, MakeShape({1}));
  builder.MakeInitializer(MakeInitializer<float>("public", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("output", {1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("orphan", {1}, {3.0f}));
  builder.MakeOutput("output");
  EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
  ASSERT_EQ(builder.Initializers().size(), 2u);
  EXPECT_EQ(builder.Initializers()[0].name(), "public");
  EXPECT_EQ(builder.Initializers()[1].name(), "output");
  const GraphProto graph = builder.ToGraph();
  EXPECT_EQ(graph.value_info().size(), 0u);
  EXPECT_EQ(graph.input().size(), 1u);
}

TEST(GraphBuilder, NativeCleanupRetainsInitializerCapturedByLiveBranch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInitializer(MakeInitializer<float>("captured", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("orphan", {1}, {2.0f}));
  builder.MakeNode("Neg", {"orphan"}, {"dead"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeOutput("captured");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"Y"}, "", "", attributes);
  builder.MakeOutput("Y");
  EXPECT_EQ(builder.RemoveUnusedNodes(), 1u);
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].name(), "captured");
  EXPECT_EQ(builder.ToGraph().value_info().size(), 0u);
}

TEST(GraphBuilder, NativeFunctionSnapshotsRefreshOnlyOnDemand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeNode("Neg", {"a"}, {"dead"});
  function.MakeOutput("r");
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"X"}, {"result"}, "local");
  const core::shapes::ShapesContext snapshot = builder.Shapes();
  const FunctionProto *definition = snapshot.GetLocalFunction("local:F");
  ASSERT_NE(definition, nullptr);
  ASSERT_EQ(definition->node().size(), 2u);

  auto &growing_function = builder.MakeLocalFunction("Identity", "local");
  growing_function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  std::string previous = "a";
  for (int i = 0; i < 16; ++i) {
    const std::string output = "inner_" + std::to_string(i);
    growing_function.MakeNode("Identity", {previous}, {output});
    previous = output;
    builder.MakeNode("Identity", {"X"}, {"ordinary_" + std::to_string(i)});
    EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F"), definition);
    EXPECT_FALSE(builder.Shapes().HasLocalFunction("local:Identity"));
  }
  growing_function.MakeOutput(previous);
  builder.MakeNode("Identity", {"X"}, {"from_growing"}, "local");
  EXPECT_EQ(builder.GetShape("from_growing").Shape(), MakeShape({2, 3}));
  const core::shapes::ShapesContext updated = builder.Shapes();
  ASSERT_NE(updated.GetLocalFunction("local:Identity"), nullptr);
  EXPECT_EQ(updated.GetLocalFunction("local:Identity")->node().size(), 16u);

  EXPECT_EQ(function.RemoveUnusedNodes(), 1u);
  builder.MakeNode("Add", {"result", "X"}, {"ordinary_after_edit"});
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F"), updated.GetLocalFunction("local:F"));
  builder.MakeNode("F", {"X"}, {"after_edit"}, "local");
  ASSERT_NE(builder.Shapes().GetLocalFunction("local:F"), nullptr);
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F")->node().size(), 1u);
  EXPECT_EQ(builder.GetShape("after_edit").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(definition->node().size(), 2u);
}

TEST(GraphBuilder, NativeDemandRefreshDiscardsRemovedFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"X"}, {"result"}, "local");
  builder.MakeOutput("result");
  const core::shapes::ShapesContext snapshot = builder.Shapes();
  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  EXPECT_FALSE(builder.HasLocalFunction("F"));
  builder.MakeNode("F", {"X"}, {"unregistered"}, "local");
  EXPECT_FALSE(builder.HasShape("unregistered"));
  EXPECT_FALSE(builder.Shapes().HasLocalFunction("local:F"));
  EXPECT_TRUE(snapshot.HasLocalFunction("local:F"));
}

TEST(GraphBuilder, NativeGraphReferencesRefreshLateFunctionDefinitions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("local", 1);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("F", {"X"}, {"branch_result"}, "local");
    branch.MakeOutput("branch_result");
    EXPECT_FALSE(branch.HasShape("branch_result"));
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  EXPECT_FALSE(builder.Shapes().HasLocalFunction("local:F"));
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  EXPECT_EQ(builder.GetShape("selected").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.Subgraph("then_branch").GetShape("branch_result").Shape(), MakeShape({2, 3}));
  const core::shapes::ShapesContext snapshot = builder.Shapes();
  builder.MakeNode("CastLike", {"selected", "X"}, {"Y"});
  EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F"), snapshot.GetLocalFunction("local:F"));
}

TEST(GraphBuilder, NativeDefaultInitializerValidatesDeclaredTensor) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  const std::vector<TensorProto> incompatible = {
      MakeInitializer<int64_t>("x", {2}, {1, 2}), MakeInitializer<float>("x", {1, 2}, {1.0f, 2.0f}),
      MakeInitializer<float>("x", {3}, {1.0f, 2.0f, 3.0f})};
  for (const TensorProto &tensor : incompatible) {
    EXPECT_THROW(builder.MakeInitializer(tensor), core::builder::BuilderError);
    EXPECT_TRUE(builder.Initializers().empty());
    EXPECT_EQ(builder.GetShape("x").Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(builder.GetShape("x").Shape(), MakeShape({2}));
  }
  const TensorProto compatible = MakeInitializer<float>("x", {2}, {1.0f, 2.0f});
  EXPECT_EQ(builder.MakeInitializer(compatible), "x");
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_THROW(builder.MakeInitializer(compatible), core::builder::BuilderError);
}

TEST(GraphBuilder, NativeDefaultInitializerAllowsSymbolicAndUnknownDimensions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("x");
  auto *type = input.mutable_type()->mutable_tensor_type();
  type->set_elem_type(TensorProto::DataType::FLOAT);
  type->mutable_shape()->add_dim()->set_dim_param("batch");
  type->mutable_shape()->add_dim();
  type->mutable_shape()->add_dim()->set_dim_value(3);
  builder.MakeInput(input);
  const auto expected_shape = builder.GetShape("x").Shape();
  EXPECT_EQ(
      builder.MakeInitializer(MakeInitializer<float>("x", {2, 4, 3}, std::vector<float>(24, 1.0f))),
      "x");
  EXPECT_EQ(builder.GetShape("x").Shape(), expected_shape);
  const auto &declared = builder.Inputs()[0].type().tensor_type().shape();
  EXPECT_EQ(declared.dim(0).dim_param(), "batch");
  EXPECT_FALSE(declared.dim(1).has_dim_value());
  EXPECT_FALSE(declared.dim(1).has_dim_param());
  EXPECT_EQ(declared.dim(2).dim_value(), 3);
}

TEST(GraphBuilder, NativeDefaultInitializerAllowsUnknownRankAndType) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("unknown_rank");
  input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  builder.MakeInput(input);
  EXPECT_EQ(builder.MakeInitializer(MakeInitializer<float>("unknown_rank", {2}, {1.0f, 2.0f})),
            "unknown_rank");
  EXPECT_FALSE(builder.Inputs()[0].type().tensor_type().has_shape());
  EXPECT_FALSE(builder.Compute().Shapes().Has("unknown_rank"));
  ValueInfoProto untyped;
  untyped.set_name("untyped");
  builder.MakeInput(untyped);
  EXPECT_EQ(builder.MakeInitializer(MakeInitializer<int64_t>("untyped", {1}, {1})), "untyped");
  builder.MakeInput("scalar", core::symbolic::TensorType::kFloat, MakeShape({}));
  EXPECT_THROW(builder.MakeInitializer(MakeInitializer<float>("scalar", {1}, {1.0f})),
               core::builder::BuilderError);
  EXPECT_EQ(builder.MakeInitializer(MakeInitializer<float>("scalar", {}, {1.0f})), "scalar");
}

TEST(GraphBuilder, NativeDefaultInitializerRejectsNonTensorInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("sequence");
  input.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::DataType::FLOAT);
  builder.MakeInput(input);
  EXPECT_THROW(builder.MakeInitializer(MakeInitializer<float>("sequence", {1}, {1.0f})),
               core::builder::BuilderError);
  EXPECT_TRUE(builder.Initializers().empty());
}

} // namespace Test
