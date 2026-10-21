// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_verify.h"

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

class MarkReversedProducer : public core::builder::PatternOptimization {
public:
  MarkReversedProducer() : PatternOptimization(1, "MarkReversedProducer") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (candidate.op_type() != "Mul" || candidate.name() == "revisited") {
      return {};
    }
    for (const auto &node : graph.Builder().Nodes()) {
      for (const auto &input : node.input()) {
        const auto *producer = graph.NodeBefore(input.value());
        if (producer != nullptr) {
          EXPECT_LT(graph.Position(*producer), graph.Position(node));
        }
      }
    }
    return {this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &matched) const override {
    utils::RepeatedProtoField<NodeProto> nodes;
    nodes.push_back(*matched[0]);
    nodes[0].set_name("revisited");
    return nodes;
  }
};

class ReplaceLateProducer : public core::builder::PatternOptimization {
public:
  explicit ReplaceLateProducer(bool replace_dependency)
      : PatternOptimization(1, "ReplaceLateProducer"), replace_dependency(replace_dependency) {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (candidate.name() != "rewrite")
      return {};
    const auto *dependency = graph.NodeBefore("a");
    return replace_dependency
               ? core::builder::MatchResult{this, {&candidate, dependency}, dependency}
               : core::builder::MatchResult{this, {&candidate}, dependency};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &) const override {
    utils::RepeatedProtoField<NodeProto> nodes;
    if (replace_dependency)
      nodes.push_back(MakeNode("Abs", {"X"}, {"a"}));
    nodes.push_back(MakeNode("Mul", {"a", "X"}, {"p"}));
    return nodes;
  }

  bool replace_dependency;
};

class ReplaceIndependentNeg : public core::builder::PatternOptimization {
public:
  ReplaceIndependentNeg() : PatternOptimization(1, "ReplaceIndependentNeg") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.op_type() == "Neg" ? core::builder::MatchResult{this, {&candidate}, &candidate}
                                        : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &matched) const override {
    utils::RepeatedProtoField<NodeProto> nodes;
    nodes.push_back(
        MakeNode("Abs", {matched[0]->input(0).value()}, {matched[0]->output(0).value()}));
    return nodes;
  }
};

class CaptureOutputAndDropIdentity : public core::builder::PatternOptimization {
public:
  CaptureOutputAndDropIdentity() : PatternOptimization(1, "CaptureOutputAndDropIdentity") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.name() == "drop" ? core::builder::MatchResult{this, {&candidate}, &candidate}
                                      : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph, const std::vector<const NodeProto *> &) const override {
    auto &outputs =
        const_cast<utils::RepeatedProtoField<ValueInfoProto> &>(graph.Builder().Outputs());
    outputs[0].set_name("source");
    return utils::RepeatedProtoField<NodeProto>();
  }
};

StructTypeProto RewriteRecord() {
  StructTypeProto type;
  type.set_type_id(101);
  type.set_name("rewrite_record");
  auto *field = type.mutable_structure()->add_field();
  field->set_name("value");
  auto *tensor = field->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape();
  return type;
}

class RegisterStructType : public core::builder::PatternOptimization {
public:
  RegisterStructType() : PatternOptimization(1, "RegisterStructType") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.name() == "register_type"
               ? core::builder::MatchResult{this, {&candidate}, &candidate}
               : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override {
    graph.Builder().MakeStructType(RewriteRecord());
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(
        MakeNode("Neg", {nodes[0]->input()[0].value()}, {nodes[0]->output()[0].value()}));
    return replacements;
  }
};

class RegisterLocalFunction : public core::builder::PatternOptimization {
public:
  RegisterLocalFunction() : PatternOptimization(1, "RegisterLocalFunction") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (candidate.op_type() != "Identity" || !candidate.name().value().starts_with("rewrite_") ||
        !graph.HasShape(candidate.input()[0].value())) {
      return {};
    }
    return {this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override {
    if (nodes[0]->name() == "rewrite_first") {
      auto &function = graph.Builder().MakeLocalFunction("Generated", "local.test");
      function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
      function.MakeNode("Identity", {"arg"}, {"result"});
      function.MakeOutput("result");
    }
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(
        MakeNode("Neg", {nodes[0]->input()[0].value()}, {nodes[0]->output()[0].value()}));
    return replacements;
  }
};

} // namespace

TEST(GraphGraph, IndexesPredecessorsAndSuccessors) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInitializer(MakeInitializer<float>("unused", {2}, {1.0f, 2.0f}));
    if (add_dead_node) {
      builder.MakeNode("Add", {"x", "unused"}, {"dead"});
    }
    builder.MakeOutput("x");
    const ModelProto original = builder.ToModel();
    core::builder::GraphGraph graph(builder);
    const auto rewrites = graph.Optimize();
    ASSERT_EQ(rewrites.rewrites().size(), 1u);
    EXPECT_EQ(rewrites.rewrites(0).pattern_name(), "RemoveUnusedNodes");
    ASSERT_EQ(rewrites.rewrites(0).removed_initializers().size(), 1u);
    const GraphProto optimized = builder.ToGraph();
    EXPECT_EQ(optimized.node_size(), 0);
    EXPECT_EQ(optimized.initializer_size(), 0);
    const GraphProto replayed = core::builder::GraphBuilder(original, SchemaLookup())
                                    .ApplyRewritings(rewrites)
                                    .BuildGraph();
    EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
  }
}

TEST(GraphGraph, CleanupInvalidatesUnusedEncodedAndPagedInitializers) {
  for (const bool add_dead_node : {false, true}) {
    SCOPED_TRACE(add_dead_node);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeStructType(RewriteRecord());
    EncodedValueProto encoded;
    encoded.set_name("encoded");
    encoded.mutable_struct_type()->set_type_ref(101);
    encoded.set_raw_data(std::string(sizeof(float), '\0'));
    builder.MakeEncodedInitializer(encoded);
    PagedCacheProto cache;
    cache.set_name("cache");
    auto *block = cache.add_blocks();
    block->set_start(0);
    block->set_length(1);
    *block->mutable_key() = MakeInitializer<float>("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
    *block->mutable_value() = block->key();
    builder.MakePagedCacheInitializer(cache);
    if (add_dead_node)
      builder.MakeNode("Abs", {"X"}, {"dead"});
    builder.MakeOutput("X");
    ASSERT_TRUE(builder.Shapes().HasEncodedValue("encoded"));
    ASSERT_TRUE(builder.Shapes().HasType("cache"));
    core::builder::GraphGraph optimizer(builder);
    optimizer.Optimize();
    EXPECT_TRUE(builder.EncodedInitializers().empty());
    EXPECT_TRUE(builder.PagedCacheInitializers().empty());
    for (const std::string name : {"encoded", "cache"}) {
      EXPECT_FALSE(builder.Shapes().HasEncodedValue(name));
      EXPECT_FALSE(builder.Shapes().HasType(name));
      EXPECT_FALSE(builder.Compute().IsConstantValue(name));
      EXPECT_EQ(builder.Compute().ValueTags().count(name), 0u);
    }
  }
}

TEST(GraphGraph, BuildRejectsNestedOutputMadeCapturedOnlyByRewrite) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("source", core::symbolic::TensorType::kFloat, MakeShape({2}));
  auto &branch = builder.MakeSubgraph("branch");
  branch.MakeNode("Identity", {"source"}, {"local_result"}, "", "drop");
  branch.MakeOutput("local_result");
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  auto &else_branch = builder.MakeSubgraph("else_branch");
  else_branch.MakeNode("Identity", {"source"}, {"else_result"});
  else_branch.MakeOutput("else_result");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &[attribute_name, graph_name] :
       {std::pair{"then_branch", "branch"}, std::pair{"else_branch", "else_branch"}}) {
    AttributeProto reference;
    reference.set_name(std::string(attribute_name) + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(graph_name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  EXPECT_NO_THROW(VerifyModel(builder.ToModel()));

  core::builder::GraphGraph graph(builder,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<CaptureOutputAndDropIdentity>()});

  EXPECT_THROW(graph.Optimize(), core::builder::BuilderError);
  EXPECT_TRUE(branch.Nodes().empty());
  EXPECT_THROW(branch.BuildGraph(), core::builder::BuilderError);
  EXPECT_THROW(builder.ToModel(), core::builder::BuilderError);
}

TEST(GraphGraph, NestedRewriteRegistersStructTypeInModelScope) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("captured", core::symbolic::TensorType::kFloat, MakeShape({2}));
  auto &branch = builder.MakeSubgraph("branch");
  branch.MakeNode("Identity", {"captured"}, {"result"}, "", "register_type");
  branch.MakeOutput("result");

  core::builder::GraphGraph graph(branch,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<RegisterStructType>()});
  ASSERT_EQ(graph.Optimize().rewrites().size(), 1u);

  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.struct_types().size(), 1u);
  EXPECT_EQ(model.struct_types()[0].SerializeAsString(), RewriteRecord().SerializeAsString());
}

TEST(GraphGraph, NestedDisjointRewritesPreserveCapturedShapeAfterLocalFunctionRegistration) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("captured", core::symbolic::TensorType::kFloat, MakeShape({2}));
  auto &branch = builder.MakeSubgraph("branch");
  branch.MakeNode("Identity", {"captured"}, {"first"}, "", "rewrite_first");
  branch.MakeNode("Identity", {"captured"}, {"second"}, "", "rewrite_second");
  branch.MakeOutput("first");
  branch.MakeOutput("second");

  core::builder::GraphGraph graph(branch,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<RegisterLocalFunction>()});
  const auto rewrites = graph.Optimize();

  EXPECT_EQ(rewrites.rewrites().size(), 2u);
  EXPECT_TRUE(branch.HasShape("captured"));
  ASSERT_EQ(branch.Nodes().size(), 2u);
  EXPECT_EQ(branch.Nodes()[0].op_type(), "Neg");
  EXPECT_EQ(branch.Nodes()[1].op_type(), "Neg");
}

TEST(GraphGraph, OptionalUnusedCleanupPreservesDeadEnds) {
  for (bool remove_unused : {false, true}) {
    SCOPED_TRACE(remove_unused);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInitializer(MakeInitializer<float>("weight", {2}, {1.0f, 2.0f}));
    builder.MakeNode("Add", {"x", "weight"}, {"dead"});
    builder.MakeNode("Neg", {"x"}, {"out"});
    builder.MakeOutput("out");
    EXPECT_EQ(builder.ToGraph().node_size(), 2);

    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
    const auto rewrites = graph.Optimize(-1, nullptr, {}, remove_unused);
    EXPECT_EQ(builder.ToGraph().node_size(), remove_unused ? 1 : 2);
    EXPECT_EQ(builder.ToGraph().initializer_size(), remove_unused ? 0 : 1);
    EXPECT_EQ(rewrites.rewrites().size(), remove_unused ? 1u : 0u);
  }
}

TEST(GraphGraph, OptionalUnusedCleanupRecursesIntoSubgraphs) {
  for (bool remove_unused : {false, true}) {
    SCOPED_TRACE(remove_unused);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));

    auto &then_branch = builder.MakeSubgraph("then_branch");
    then_branch.MakeInitializer(MakeInitializer<float>("weight", {2}, {1.0f, 2.0f}));
    then_branch.MakeNode("Add", {"x", "weight"}, {"dead"});
    then_branch.MakeNode("Identity", {"x"}, {"branch_result"});
    then_branch.MakeOutput("branch_result");
    auto &else_branch = builder.MakeSubgraph("else_branch");
    else_branch.MakeNode("Identity", {"x"}, {"branch_result"});
    else_branch.MakeOutput("branch_result");

    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const std::string name : {"then_branch", "else_branch"}) {
      AttributeProto reference;
      reference.set_name(name + "_ref");
      reference.set_type(AttributeProto::AttributeType::STRING);
      reference.set_s(name);
      attributes.push_back(reference);
    }
    builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
    builder.MakeOutput("selected");

    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
    graph.Optimize(-1, nullptr, {}, remove_unused);

    EXPECT_EQ(then_branch.ToGraph().node_size(), remove_unused ? 1 : 2);
    EXPECT_EQ(then_branch.ToGraph().initializer_size(), remove_unused ? 0 : 1);
  }
}

TEST(GraphGraph, NativeInputDefaultsAreNotOptimizationConstants) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  const std::vector<std::string> y = builder.MakeNode("Neg", {"x"}, {"y"});

  core::builder::GraphGraph graph(builder);
  EXPECT_EQ(graph.GetComputedConstant("y"), nullptr);

  graph.SetComputedConstant("y", MakeInitializer<float>("y", {1}, {3.0f}));
  const TensorProto *tensor = graph.GetComputedConstant("y");
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->name().value(), "y");
}

TEST(GraphGraph, ReverseReplacementOrderStaysTopologicalAcrossIterationsAndReplay) {
  for (const bool folding : {false, true}) {
    SCOPED_TRACE(folding);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
    builder.MakeNode("Abs", {"p"}, {"Y"});
    builder.MakeOutput("Y");
    ModelProto original;
    *original.mutable_graph() = builder.BuildGraph();
    auto &opset = *original.add_opset_import();
    opset.set_domain("");
    opset.set_version(28);

    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<ReverseReplacementOrder>());
    patterns.push_back(std::make_unique<MarkReversedProducer>());
    core::builder::GraphGraph optimizer(builder, std::move(patterns));
    core::builder::ConstantFoldingOptions options;
    options.enabled = folding;
    const RewritingProto plan = optimizer.Optimize(2, nullptr, options);
    ASSERT_EQ(plan.rewrites().size(), 2u);
    EXPECT_EQ(plan.rewrites(0).added_nodes_positions().values(), (std::vector<uint64_t>{1, 0}));
    EXPECT_EQ(plan.rewrites(1).pattern_name(), "MarkReversedProducer");
    EXPECT_EQ(plan.rewrites(1).matched_nodes().values(), (std::vector<uint64_t>{0}));
    ASSERT_EQ(builder.Nodes().size(), 2u);
    EXPECT_EQ(builder.Nodes()[0].op_type(), "Mul");
    EXPECT_EQ(builder.Nodes()[1].op_type(), "Add");
    EXPECT_EQ(builder.Nodes()[0].name(), "revisited");

    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    core::builder::GraphBuilder source(original, SchemaLookup());
    auto rewritten = source.ApplyRewritings(loaded);
    ASSERT_EQ(rewritten.Nodes().size(), 2u);
    EXPECT_EQ(rewritten.Nodes()[0].op_type(), "Mul");
    EXPECT_EQ(rewritten.Nodes()[1].op_type(), "Add");
    const GraphProto replayed = rewritten.BuildGraph();
    EXPECT_EQ(replayed.SerializeAsString(), builder.BuildGraph().SerializeAsString());
    EXPECT_EQ(source.BuildGraph().SerializeAsString(), original.graph().SerializeAsString());
    EXPECT_EQ(rewritten.GetShape("p"), source.GetShape("p"));
    EXPECT_EQ(rewritten.GetShape("Y"), source.GetShape("Y"));
    EXPECT_TRUE(rewritten.Compute().NodeReuse(0).empty());
    auto unchanged = rewritten.ApplyRewritings(RewritingProto{});
    EXPECT_EQ(unchanged.BuildGraph().SerializeAsString(), replayed.SerializeAsString());
    EXPECT_EQ(unchanged.Compute().NodeTags(), rewritten.Compute().NodeTags());
    EXPECT_EQ(unchanged.Compute().NodeConstant(), rewritten.Compute().NodeConstant());
    EXPECT_EQ(unchanged.Compute().Memory(), rewritten.Compute().Memory());
    EXPECT_EQ(unchanged.Compute().PeakMemory(), rewritten.Compute().PeakMemory());
    for (std::size_t index = 0; index < rewritten.Nodes().size(); ++index) {
      EXPECT_EQ(unchanged.Compute().NodeReuse(index), rewritten.Compute().NodeReuse(index));
    }
    RewritingProto follow_up;
    auto &replace = *follow_up.add_rewrites();
    replace.add_matched_nodes(0);
    replace.add_added_nodes_positions(0);
    replace.add_added_nodes(MakeNode("Neg", {"X"}, {"p"}));
    auto composed = rewritten.ApplyRewritings(follow_up);
    EXPECT_EQ(unchanged.ApplyRewritings(follow_up).BuildGraph().SerializeAsString(),
              composed.BuildGraph().SerializeAsString());
    ASSERT_EQ(composed.Nodes().size(), 2u);
    EXPECT_EQ(composed.Nodes()[0].op_type(), "Neg");
    EXPECT_EQ(composed.Nodes()[1].op_type(), "Add");
    EXPECT_EQ(composed.GetShape("p"), source.GetShape("p"));
    EXPECT_EQ(composed.GetShape("Y"), source.GetShape("Y"));
    EXPECT_TRUE(composed.Compute().NodeReuse(0).empty());
    EXPECT_EQ(rewritten.BuildGraph().SerializeAsString(), replayed.SerializeAsString());
    const auto composed_graph = composed.ToGraph();
    EXPECT_EQ(composed_graph.node(0).op_type(), "Neg");
    EXPECT_EQ(composed_graph.node(1).op_type(), "Add");
    rewritten.MakeNode("Identity", {"Y"}, {"copy"});
    EXPECT_EQ(rewritten.GetShape("copy"), source.GetShape("Y"));
    const GraphProto finalized = builder.ToGraph();
    EXPECT_EQ(finalized.node(0).op_type(), "Mul");
    EXPECT_EQ(finalized.node(1).op_type(), "Add");
    EXPECT_EQ(builder.Nodes()[0].op_type(), "Mul");
    EXPECT_EQ(builder.Nodes()[1].op_type(), "Add");
  }
}

TEST(GraphGraph, DisjointRewritesRebaseCoordinatesAfterDependencyOrdering) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
  builder.MakeNode("Abs", {"p"}, {"Y"});
  builder.MakeNode("Mul", {"X", "X"}, {"Z"});
  builder.MakeOutput("Y");
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  original.add_opset("", 28);
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReverseReplacementOrder>("Z"));
  patterns.push_back(std::make_unique<MarkReversedProducer>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  core::builder::ConstantFoldingOptions options;
  options.enabled = false;
  const auto plan = optimizer.Optimize(1, nullptr, options);
  ASSERT_EQ(plan.rewrites().size(), 2u);
  EXPECT_EQ(plan.rewrites(0).added_nodes_positions().values(), (std::vector<uint64_t>{2, 1}));
  EXPECT_EQ(plan.rewrites(1).matched_nodes().values(), (std::vector<uint64_t>{0}));
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].output(0), "Z");
  EXPECT_EQ(builder.Nodes()[1].output(0), "p");
  EXPECT_EQ(builder.Nodes()[2].output(0), "Y");
  auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan);
  EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), builder.BuildGraph().SerializeAsString());
  EXPECT_EQ(replayed.Compute().Memory(), builder.Compute().Memory());
}

TEST(GraphGraph, ReplayBatchDoesNotHideAnInvalidIntermediateDependency) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  source.MakeNode("Neg", {"X"}, {"p"});
  source.MakeNode("Abs", {"p"}, {"Y"});
  source.MakeOutput("Y");
  RewritingProto plan;
  plan.add_rewrites()->add_matched_nodes(0);
  auto &restore = *plan.add_rewrites();
  restore.add_matched_nodes(0);
  restore.add_added_nodes(MakeNode("Neg", {"X"}, {"p"}));
  restore.add_added_nodes_positions(0);
  restore.add_added_nodes(MakeNode("Abs", {"p"}, {"Y"}));
  restore.add_added_nodes_positions(1);
  EXPECT_THROW(source.ApplyRewritings(plan), core::builder::BuilderError);
  EXPECT_EQ(source.Nodes()[0].op_type(), "Neg");
  EXPECT_EQ(source.Nodes()[1].op_type(), "Abs");
}

TEST(GraphGraph, NormalizationRecordsDisplacedRetainedNodesForCoordinateReplay) {
  for (const auto &[replace_dependency, batch_dependency] :
       {std::pair{false, false}, std::pair{true, false}, std::pair{false, true}}) {
    SCOPED_TRACE(replace_dependency);
    SCOPED_TRACE(batch_dependency);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
    builder.MakeNode("Relu", {"p"}, {"C"});
    builder.MakeNode("Exp", {"X"}, {"I"});
    builder.MakeNode(batch_dependency ? "Neg" : "Abs", {"X"}, {"a"});
    builder.MakeOutput("C");
    builder.MakeOutput("I");
    ModelProto original;
    *original.mutable_graph() = builder.BuildGraph();
    original.add_opset("", 28);
    std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_shared<ReplaceLateProducer>(replace_dependency));
    if (batch_dependency)
      patterns.push_back(std::make_shared<ReplaceIndependentNeg>());
    core::builder::GraphGraph optimizer(builder, std::move(patterns));
    core::builder::ConstantFoldingOptions options;
    options.enabled = false;
    auto plan = optimizer.Optimize(1, nullptr, options);
    for (const auto &edit : plan.rewrites()) {
      for (const auto &node : edit.added_nodes()) {
        ASSERT_FALSE(node.op_type().empty());
        ASSERT_EQ(node.output().size(), 1u);
      }
    }
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded);
    EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), builder.BuildGraph().SerializeAsString());
    ASSERT_EQ(builder.Nodes().size(), 4u);
    EXPECT_EQ(builder.Nodes()[0].output(0), "I");
    EXPECT_EQ(builder.Nodes()[1].output(0), "a");
    EXPECT_EQ(builder.Nodes()[2].output(0), "p");
    EXPECT_EQ(builder.Nodes()[3].output(0), "C");
    RewritingProto follow_up;
    auto &rewrite = *follow_up.add_rewrites();
    rewrite.add_matched_nodes(3);
    rewrite.add_added_nodes(MakeNode("Abs", {"p"}, {"C"}));
    rewrite.add_added_nodes_positions(3);
    plan.ref_rewrites().push_back(rewrite);
    auto composed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan);
    auto expected = builder.ApplyRewritings(follow_up);
    EXPECT_EQ(composed.BuildGraph().SerializeAsString(), expected.BuildGraph().SerializeAsString());
    EXPECT_EQ(replayed.ApplyRewritings(follow_up).BuildGraph().SerializeAsString(),
              expected.BuildGraph().SerializeAsString());
  }
}

TEST(GraphGraph, NormalizationRebasesManyIndependentReplacements) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  for (std::size_t index = 0; index < 512; ++index) {
    const auto input = "in_" + std::to_string(index);
    const auto output = "out_" + std::to_string(index);
    builder.MakeInput(input, core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeNode("Neg", {input}, {output});
    builder.MakeOutput(output);
  }
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  original.add_opset("", 28);
  core::builder::GraphGraph optimizer(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                   std::make_shared<ReplaceIndependentNeg>()});
  core::builder::ConstantFoldingOptions options;
  options.enabled = false;
  const auto plan = optimizer.Optimize(1, nullptr, options);
  ASSERT_EQ(plan.rewrites().size(), 512u);
  for (std::size_t index = 0; index < plan.rewrites().size(); ++index) {
    EXPECT_EQ(plan.rewrites(index).matched_nodes(0), index);
    EXPECT_EQ(plan.rewrites(index).added_nodes_positions(0), index);
  }
  auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan);
  EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), builder.BuildGraph().SerializeAsString());
}

TEST(GraphGraph, ReplayRestoresDependenciesAndReuseBeforeExport) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  source.MakeNode("Neg", {"X"}, {"p"});
  source.MakeNode("Abs", {"X"}, {"a"});
  source.MakeNode("Relu", {"p"}, {"Y"});
  source.MakeNode("Abs", {"Y"}, {"Z"});
  source.MakeOutput("a");
  source.MakeOutput("Z");
  ASSERT_FALSE(source.Compute().NodeReuse(2).empty());
  RewritingProto plan;
  auto &rewrite = *plan.add_rewrites();
  rewrite.add_matched_nodes(0);
  rewrite.add_matched_nodes(3);
  rewrite.add_added_nodes_positions(0);
  rewrite.add_added_nodes(MakeNode("Abs", {"Y"}, {"Z"}));
  rewrite.add_added_nodes_positions(3);
  rewrite.add_added_nodes(MakeNode("Neg", {"X"}, {"p"}));
  auto reversed = source.ApplyRewritings(plan);
  ASSERT_EQ(reversed.Nodes().size(), 4u);
  EXPECT_EQ(reversed.Nodes()[0].output(0), "a");
  EXPECT_EQ(reversed.Nodes()[1].output(0), "p");
  EXPECT_EQ(reversed.Nodes()[2].output(0), "Y");
  EXPECT_EQ(reversed.Nodes()[3].output(0), "Z");
  auto replayed = reversed.ApplyRewritings(RewritingProto{});
  EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), reversed.BuildGraph().SerializeAsString());
  EXPECT_FALSE(replayed.Compute().NodeReuse(2).empty());
  EXPECT_EQ(replayed.Compute().Memory(), reversed.Compute().Memory());
  EXPECT_EQ(replayed.Compute().PeakMemory(), reversed.Compute().PeakMemory());
}

TEST(GraphGraph, FoldedInitializerRefreshesLastUseWithoutChangingConsumerReferences) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  source.MakeNode("Neg", {"X"}, {"p"});
  source.MakeNode("Relu", {"p"}, {"Y"});
  source.MakeOutput("Y");
  auto graph = source.BuildGraph();
  core::compute::ComputeContext compute;
  const auto tags = compute.ComputeValueAndNodeTags(graph);
  compute.ComputeInPlaceReuseGraph(graph, source.Shapes(), false, tags.first);
  graph.ref_node().erase(graph.ref_node().begin());
  graph.ref_initializer().push_back(MakeInitializer<float>("p", {2}, {1.f, 2.f}));
  compute.UpdateNodes(graph.node(), {1}, {{"p"}}, {"p"}, core::symbolic::Device::kCPU);
  compute.SeedReuseInput("p", false, true, false);
  compute.RefreshNodes(graph.node(), {{0, {"p"}}}, source.Shapes(), {"p"},
                       core::symbolic::Device::kCPU);
  compute.WriteToGraph(graph);
  bool marked_last_use = false;
  for (const auto &metadata : graph.node(0).metadata_props()) {
    if (metadata.key() == core::compute::kNotUsedAfterMetadataKey)
      marked_last_use = metadata.value() == "p";
    if (metadata.key() == core::compute::kReleaseAfterMetadataKey) {
      EXPECT_NE(metadata.value(), "p");
    }
  }
  EXPECT_TRUE(marked_last_use);
}

TEST(GraphGraph, ReplayInsertsProducerBeforeConsumersWithoutSortingIndependentNodes) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  source.MakeNode("Neg", {"X"}, {"p"});
  source.MakeNode("Relu", {"p"}, {"Y"});
  source.MakeNode("Abs", {"X"}, {"independent"});
  source.MakeNode("Identity", {"p"}, {"tail"});
  source.MakeOutput("Y");
  source.MakeOutput("independent");
  source.MakeOutput("tail");
  RewritingProto plan;
  auto &rewrite = *plan.add_rewrites();
  rewrite.add_matched_nodes(0);
  rewrite.add_added_nodes(MakeNode("Mul", {"X", "X"}, {"p"}));
  rewrite.add_added_nodes_positions(3);
  auto result = source.ApplyRewritings(plan);
  ASSERT_EQ(result.Nodes().size(), 4u);
  const std::vector<std::string> expected{"p", "Y", "independent", "tail"};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(result.Nodes()[index].output(0), expected[index]);
    EXPECT_EQ(result.GetShape(expected[index]), source.GetShape(expected[index]));
  }
  EXPECT_TRUE(result.Compute().NodeReuse(0).empty());
  EXPECT_TRUE(result.Compute().NodeReuse(1).empty());
  auto unchanged = result.ApplyRewritings(RewritingProto{});
  EXPECT_EQ(unchanged.BuildGraph().SerializeAsString(), result.BuildGraph().SerializeAsString());
  EXPECT_EQ(unchanged.Compute().Memory(), result.Compute().Memory());
  EXPECT_EQ(unchanged.Compute().PeakMemory(), result.Compute().PeakMemory());
  const auto exported = result.ToGraph();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(exported.node(index).output(0), expected[index]);
  }
}

TEST(GraphGraph, RewritingReindexesRetainedControlFlowContexts) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  source.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  source.MakeNode("Neg", {"X"}, {"p"});
  source.MakeNode("Abs", {"X"}, {"unused"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = source.MakeSubgraph(name);
    branch.MakeNode("Identity", {"p"}, {"branch_result"});
    branch.MakeOutput("branch_result");
    auto &reference = attributes.add();
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s(name);
  }
  source.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  source.MakeOutput("selected");
  ASSERT_TRUE(source.Shapes().HasSubgraphContext(2, "then_branch"));
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    RewritingProto plan;
    auto &rewrite = *plan.add_rewrites();
    rewrite.add_matched_nodes(1);
    if (reverse) {
      rewrite.add_matched_nodes(0);
      rewrite.add_added_nodes(MakeNode("Neg", {"X"}, {"p"}));
      rewrite.add_added_nodes_positions(1);
    }
    auto result = source.ApplyRewritings(plan);
    auto replayed = result.ApplyRewritings(RewritingProto{});
    EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), result.BuildGraph().SerializeAsString());
    const int64_t position = 1;
    EXPECT_EQ(result.Nodes()[static_cast<std::size_t>(position)].op_type(), "If");
    EXPECT_FALSE(result.Shapes().HasSubgraphContext(2, "then_branch"));
    for (const std::string name : {"then_branch", "else_branch"}) {
      ASSERT_TRUE(result.Shapes().HasSubgraphContext(position, name));
      EXPECT_EQ(result.Shapes().GetSubgraphContext(position, name).Get("branch_result"),
                source.Shapes().GetSubgraphContext(2, name).Get("branch_result"));
      ASSERT_TRUE(replayed.Shapes().HasSubgraphContext(position, name));
      EXPECT_EQ(replayed.Shapes().GetSubgraphContext(position, name).Get("branch_result"),
                result.Shapes().GetSubgraphContext(position, name).Get("branch_result"));
    }
  }
}

TEST(GraphGraph, NativeExportOrdersDependenciesAndRecomputesCapturedLifetimes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
  builder.MakeNode("Abs", {"p"}, {"Y"});
  ASSERT_EQ(builder.Compute().NodeReuse(1).size(), 1u);
  builder.MakeOutput("Y");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"p"}, {"branch_result"});
    branch.MakeOutput("branch_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }

  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  EXPECT_TRUE(builder.Compute().NodeReuse(1).empty());
  builder.MakeOutput("selected");
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReverseReplacementOrder>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  EXPECT_FALSE(optimizer.Optimize(1).rewrites().empty());
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
  EXPECT_EQ(builder.ToGraph().SerializeAsString(), graph.SerializeAsString());
}

TEST(GraphGraph, NativeExportOrdersProducerBeforeLexicalCapture) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"p"}, {"branch_result"});
    branch.MakeOutput("branch_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  builder.MakeNode("Abs", {"X"}, {"independent"});
  builder.MakeOutput("independent");
  builder.MakeNode("Abs", {"p"}, {"Y"});
  builder.MakeOutput("Y");
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  original.add_opset("", 28);
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReverseReplacementOrder>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  const auto plan = optimizer.Optimize(1);
  ASSERT_EQ(builder.Nodes().size(), 4u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Mul");
  EXPECT_EQ(builder.Nodes()[1].op_type(), "If");
  EXPECT_EQ(builder.Nodes()[2].output(0), "independent");
  EXPECT_EQ(builder.Nodes()[3].op_type(), "Add");
  auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan);
  EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), builder.BuildGraph().SerializeAsString());
  const GraphProto graph = builder.ToGraph();
  ASSERT_EQ(graph.node().size(), 4u);
  EXPECT_EQ(graph.node(0).op_type(), "Mul");
  EXPECT_EQ(graph.node(1).op_type(), "If");
  EXPECT_EQ(graph.node(2).output(0), "independent");
  EXPECT_EQ(graph.node(3).op_type(), "Add");
}

TEST(GraphGraph, NativeExportRejectsCyclesAndUndefinedDependencies) {
  for (const std::string input : {"Y", "undefined"}) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
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
