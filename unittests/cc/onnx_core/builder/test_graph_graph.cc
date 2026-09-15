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

// Exercises finalization independently of any production pattern's ordering.
class ReverseReplacementOrder : public core::builder::PatternOptimization {
public:
  explicit ReverseReplacementOrder(std::string producer_input = "X")
      : PatternOptimization(1, "ReverseReplacementOrder"),
        producer_input(std::move(producer_input)) {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (candidate.name() != "rewrite") {
      return {};
    }
    const NodeProto *consumer = graph.NodeBefore("Y");
    return {this, {&candidate, consumer}, consumer};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &) const override {
    utils::RepeatedProtoField<NodeProto> nodes;
    NodeProto consumer = MakeNode("Add", {"p", "X"}, {"Y"});
    consumer.add_metadata(core::compute::kReleaseAfterMetadataKey, "p");
    consumer.add_metadata(core::compute::kNodePeakMemoryMetadataKey, "999999");
    consumer.add_metadata("user.annotation", "retained");
    nodes.push_back(consumer);
    nodes.push_back(MakeNode("Mul", {producer_input, "X"}, {"p"}));
    return nodes;
  }

  std::string producer_input;
};

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

  // Node-level neighbours: the Add node has no predecessor (both inputs are
  // graph inputs) and is followed by both Mul and Sub.
  const NodeProto *add = graph.NodeBefore("z");
  const NodeProto *mul = graph.NodeBefore("out1");
  const NodeProto *sub = graph.NodeBefore("out2");
  EXPECT_TRUE(graph.Predecessors(*add).empty());
  const std::vector<const NodeProto *> add_succ = graph.Successors(*add);
  ASSERT_EQ(add_succ.size(), 2u);
  EXPECT_EQ(add_succ[0], mul);
  EXPECT_EQ(add_succ[1], sub);

  // Mul reads "z" and "x"; only "z" has a producing node, so Add is its single
  // predecessor and it has no successor.
  const std::vector<const NodeProto *> mul_pred = graph.Predecessors(*mul);
  ASSERT_EQ(mul_pred.size(), 1u);
  EXPECT_EQ(mul_pred[0], add);
  EXPECT_TRUE(graph.Successors(*mul).empty());
}

TEST(GraphGraph, NativeCleanupReplaysOrphanInitializers) {
  for (bool add_dead_node : {false, true}) {
    SCOPED_TRACE(add_dead_node);
    core::builder::GraphBuilder builder("g", SchemaLookup());
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInitializer(MakeInitializer<float>("unused", {2}, {1.0f, 2.0f}));
    if (add_dead_node) {
      builder.MakeNode("Add", {"x", "unused"}, {"dead"});
    }
    builder.MakeOutput("x");
    const ModelProto original = builder.ToModel();
    core::builder::GraphGraph graph(builder);
    const auto rewrites = graph.Optimize();
    ASSERT_EQ(rewrites.size(), 1u);
    EXPECT_EQ(rewrites[0].pattern->Name(), "RemoveUnusedNodes");
    ASSERT_EQ(rewrites[0].removed_initializers.size(), 1u);
    const GraphProto optimized = builder.ToGraph();
    EXPECT_EQ(optimized.node_size(), 0);
    EXPECT_EQ(optimized.initializer_size(), 0);
    const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
    EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
  }
}

TEST(GraphGraph, NativeInputDefaultsAreNotOptimizationConstants) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  for (const std::string name : {"x", "z"}) {
    builder.MakeInput(name, core::symbolic::TensorType::kFloat, MakeShape({1}));
    builder.MakeInitializer(MakeInitializer<float>(name.c_str(), {1}, {1.0f}));
  }
  builder.MakeInitializer(MakeInitializer<float>("weight", {1}, {1.0f}));
  builder.MakeNode("Add", {"x", "weight"}, {"sum"});
  builder.MakeNode("Add", {"sum", "z"}, {"out"});
  builder.MakeOutput("out");
  core::builder::GraphGraph graph(builder);
  EXPECT_FALSE(graph.IsConstant("x"));
  EXPECT_FALSE(graph.IsConstant("sum"));
  EXPECT_EQ(graph.GetComputedConstant("x"), nullptr);
  EXPECT_TRUE(graph.IsConstant("weight"));
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
  EXPECT_EQ(builder.ConstantFold(), 0u);
  graph.Optimize();
  EXPECT_EQ(builder.Initializers().size(), 3u);
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

TEST(GraphGraph, NativeExportOrdersDependenciesAndRecomputesCapturedLifetimes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
  builder.MakeNode("Abs", {"p"}, {"Y"});
  builder.MakeOutput("Y");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeOutput("p");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }

  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReverseReplacementOrder>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  EXPECT_FALSE(optimizer.Optimize(1).empty());
  const GraphProto graph = builder.ToGraph();
  ASSERT_EQ(graph.node().size(), 3u);
  EXPECT_EQ(graph.node(0).op_type(), "Mul");
  EXPECT_EQ(graph.node(1).op_type(), "Add");
  EXPECT_EQ(graph.node(2).op_type(), "If");
  bool preserved_annotation = false;
  for (const auto &metadata : graph.node(1).metadata_props()) {
    EXPECT_NE(metadata.key(), core::compute::kReleaseAfterMetadataKey);
    if (metadata.key() == core::compute::kNodePeakMemoryMetadataKey) {
      EXPECT_NE(metadata.value(), "999999");
    }
    preserved_annotation |= metadata.key() == "user.annotation" && metadata.value() == "retained";
  }
  EXPECT_TRUE(preserved_annotation);
  bool released_after_capture = false;
  for (const auto &metadata : graph.node(2).metadata_props()) {
    released_after_capture |=
        metadata.key() == core::compute::kReleaseAfterMetadataKey && metadata.value() == "p";
  }
  EXPECT_TRUE(released_after_capture);
  EXPECT_EQ(builder.ToGraph().node(0).op_type(), "Mul");
}

TEST(GraphGraph, NativeExportOrdersProducerBeforeLexicalCapture) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeOutput("p");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  builder.MakeNode("Abs", {"p"}, {"Y"});
  builder.MakeOutput("Y");
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReverseReplacementOrder>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  optimizer.Optimize(1);
  const GraphProto graph = builder.ToGraph();
  ASSERT_EQ(graph.node().size(), 3u);
  EXPECT_EQ(graph.node(0).op_type(), "Mul");
  EXPECT_EQ(graph.node(1).op_type(), "If");
  EXPECT_EQ(graph.node(2).op_type(), "Add");
}

TEST(GraphGraph, NativeExportRejectsCyclesAndUndefinedDependencies) {
  for (const std::string input : {"Y", "undefined"}) {
    core::builder::GraphBuilder builder("g", SchemaLookup());
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
    builder.MakeNode("Abs", {"p"}, {"Y"});
    builder.MakeOutput("Y");
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<ReverseReplacementOrder>(input));
    core::builder::GraphGraph optimizer(builder, std::move(patterns));
    EXPECT_THROW(
        {
          optimizer.Optimize(1);
          builder.ToGraph();
        },
        core::builder::BuilderError);
  }
}

} // namespace Test
