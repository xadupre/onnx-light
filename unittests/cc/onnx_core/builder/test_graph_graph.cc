// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_verify.h"

#include <gtest/gtest.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
  explicit ReplaceIndependentNeg(bool expand = false,
                                 core::symbolic::Device device = core::symbolic::Device::kUndefined)
      : PatternOptimization(1, "ReplaceIndependentNeg", device), expand(expand) {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    return candidate.op_type() == "Neg" && (device == core::symbolic::Device::kUndefined ||
                                            graph.Builder().device() == device)
               ? core::builder::MatchResult{this, {&candidate}, &candidate}
               : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &matched) const override {
    utils::RepeatedProtoField<NodeProto> nodes;
    auto input = matched[0]->input(0).value();
    if (expand) {
      const auto intermediate = matched[0]->output(0).value() + "_abs";
      nodes.push_back(MakeNode("Abs", {input}, {intermediate}));
      input = intermediate;
    }
    nodes.push_back(MakeNode("Abs", {input}, {matched[0]->output(0).value()}));
    return nodes;
  }

  bool expand;
};

class ReplaceNegWithIdentity : public core::builder::PatternOptimization {
public:
  ReplaceNegWithIdentity() : PatternOptimization(1, "ReplaceNegWithIdentity") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.op_type() == "Neg" ? core::builder::MatchResult{this, {&candidate}, &candidate}
                                        : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &matched) const override {
    utils::RepeatedProtoField<NodeProto> result;
    result.push_back(
        MakeNode("Identity", {matched[0]->input(0).value()}, {matched[0]->output(0).value()}));
    return result;
  }
};

class ReplaceNestedNegWithScaler : public core::builder::PatternOptimization {
public:
  ReplaceNestedNegWithScaler() : PatternOptimization(1, "ReplaceNestedNegWithScaler") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.name() == "nested_rewrite"
               ? core::builder::MatchResult{this, {&candidate}, &candidate}
               : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &matched) const override {
    graph.Builder().SetOpsetVersion("ai.onnx.ml", 1);
    auto node = MakeNode("Scaler", {matched[0]->input(0).value()}, {matched[0]->output(0).value()},
                         "ai.onnx.ml");
    auto &scale = node.ref_attribute().add();
    scale.set_name("scale");
    scale.set_type(AttributeProto::FLOATS);
    scale.ref_floats().push_back(-1.0f);
    utils::RepeatedProtoField<NodeProto> result;
    result.push_back(std::move(node));
    return result;
  }
};

class SwapAddInputs : public core::builder::PatternOptimization {
public:
  SwapAddInputs() : PatternOptimization(1, "SwapAddInputs") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.op_type() == "Add" && candidate.input(0) == "W"
               ? core::builder::MatchResult{this, {&candidate}, &candidate}
               : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &matched) const override {
    utils::RepeatedProtoField<NodeProto> nodes;
    auto node = *matched[0];
    std::swap(node.ref_input()[0], node.ref_input()[1]);
    nodes.push_back(std::move(node));
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
    const GraphProto canonical = builder.BuildGraph();
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
    EXPECT_EQ(builder.UndoRewritings(rewrites).BuildGraph().SerializeAsString(),
              canonical.SerializeAsString());
  }
}

TEST(GraphGraph, UnchangedLargePayloadsShareAllocationsThroughCaptureUndoAndReplay) {
  constexpr int64_t elements = 1024 * 1024;
  TensorProto weight =
      MakeInitializer<float>("weight", {elements}, std::vector<float>(elements, 1));
  TensorProto unused =
      MakeInitializer<float>("unused", {elements}, std::vector<float>(elements, 2));
  weight.set_raw_data(std::as_const(weight).ref_float_data().data(), elements * sizeof(float));
  weight.clr_float_data();
  unused.set_raw_data(std::as_const(unused).ref_float_data().data(), elements * sizeof(float));
  unused.clr_float_data();
  const auto *weight_bytes = std::as_const(weight).ref_raw_data().data();
  const auto *unused_bytes = std::as_const(unused).ref_raw_data().data();
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({elements}));
  source.MakeInitializer(weight);
  source.MakeInitializer(unused);
  source.MakeNode("Add", {"x", "weight"}, {"y"});
  source.MakeOutput("y");
  source.AddRewriting("existing", RewritingProto{});
  const ModelProto original = source.ToModel();
  core::builder::GraphBuilder working(original, SchemaLookup());
  const GraphProto imported = working.BuildGraph();
  const ModelProto exported = working.ToModel();
  core::builder::GraphGraph graph(
      working, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
  const auto plan = graph.Optimize();
  ASSERT_EQ(plan.rewrites().size(), 1u);
  ASSERT_EQ(plan.rewrites(0).removed_initializers().size(), 1u);
  auto restored_builder = working.UndoRewritings(plan);
  const ModelProto restored = restored_builder.ToModel();
  const ModelProto optimized = working.ToModel();
  const ModelProto replayed =
      core::builder::GraphBuilder(restored, SchemaLookup()).ApplyRewritings(plan).ToModel();
  EXPECT_EQ(restored.rewritings().count("existing"), 1u);
  std::unordered_set<const uint8_t *> allocations;
  size_t logical_bytes = 0;
  for (const GraphProto *representation :
       {&original.graph(), &imported, &exported.graph(), &restored.graph(), &optimized.graph(),
        &replayed.graph()}) {
    for (const auto &initializer : representation->initializer()) {
      const auto *bytes = initializer.ref_raw_data().data();
      EXPECT_EQ(bytes, initializer.name() == "weight" ? weight_bytes : unused_bytes);
      EXPECT_EQ(initializer.ref_raw_data().size(), elements * sizeof(float));
      allocations.insert(bytes);
      logical_bytes += initializer.ref_raw_data().size();
    }
  }
  EXPECT_EQ(allocations.size(), 2u);
  EXPECT_EQ(logical_bytes, 10u * elements * sizeof(float));
  EXPECT_EQ(restored.graph().SerializeAsString(), original.graph().SerializeAsString());
  EXPECT_EQ(replayed.graph().SerializeAsString(), optimized.graph().SerializeAsString());
  auto changed = restored_builder.ToModel();
  changed.mutable_graph()->mutable_initializer(0)->ref_raw_data()[0] = 9;
  EXPECT_EQ(restored.graph().initializer(0).ref_raw_data()[0], 0);
  EXPECT_EQ(std::as_const(weight).ref_raw_data().data(), weight_bytes);
  EXPECT_EQ(original.graph().initializer(0).ref_raw_data().data(), weight_bytes);
}

TEST(GraphGraph, CanonicalIdentityNamesAndMultipleInverseEdits) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Identity", {"x"}, {"a"});
  builder.MakeNode("Identity", {"a"}, {"b"});
  builder.MakeNode("Mul", {"a", "x"}, {"p"});
  builder.MakeNode("Mul", {"b", "z"}, {"q"});
  builder.MakeNode("Add", {"p", "q"}, {"y"});
  builder.MakeOutput("y");
  core::builder::GraphGraph graph(builder,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<MarkReversedProducer>()});
  EXPECT_EQ(graph.Canonicalize(), 2u);
  const GraphProto canonical = builder.BuildGraph();
  ASSERT_EQ(canonical.node().size(), 3u);
  EXPECT_EQ(canonical.node(0).input(0), "x");
  EXPECT_EQ(canonical.node(1).input(0), "x");
  const auto plan = graph.Optimize();
  ASSERT_EQ(plan.rewrites().size(), 2u);
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
  EXPECT_EQ(builder.UndoRewritings(loaded).BuildGraph().SerializeAsString(),
            canonical.SerializeAsString());
  EXPECT_EQ(builder.BuildGraph().output(0).name(), "y");
}

TEST(GraphGraph, CanonicalReferencePrunesAnnotatedIdentityValues) {
  for (const bool nested : {false, true}) {
    SCOPED_TRACE(nested);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
    source.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    auto *scope = &source;
    if (nested) {
      source.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
      scope = &source.MakeSubgraph("then_branch");
    }
    scope->MakeNode("Identity", {"x"}, {"a"});
    scope->MakeNode("Neg", {"a"}, {"y"});
    scope->MakeOutput("y");
    if (nested) {
      auto &other = source.MakeSubgraph("else_branch");
      other.MakeNode("Abs", {"x"}, {"y"});
      other.MakeOutput("y");
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const auto &name : {"then_branch", "else_branch"}) {
        auto &reference = attributes.add();
        reference.set_name(std::string(name) + "_ref");
        reference.set_type(AttributeProto::STRING);
        reference.set_s(name);
      }
      source.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
      source.MakeOutput("selected");
    }
    ModelProto model;
    model.add_opset("", 28);
    *model.mutable_graph() = source.BuildGraph();
    auto *annotated = model.mutable_graph();
    if (nested)
      for (auto &attribute : annotated->ref_node()[0].ref_attribute())
        if (attribute.name() == "then_branch") {
          annotated = attribute.mutable_g();
          break;
        }
    auto &info = *annotated->add_value_info();
    info.set_name("a");
    auto &annotation = *info.add_metadata_props();
    annotation.set_key("annotation");
    annotation.set_value("removed identity");
    core::builder::GraphBuilder builder(model, SchemaLookup());
    source.RemoveIdentityNodes();
    const auto canonical = source.BuildGraph();
    const auto plan =
        core::builder::GraphGraph(
            builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{})
            .Optimize();
    ASSERT_TRUE(plan.rewrites().empty());
    EXPECT_EQ(builder.BuildGraph().SerializeAsString(), canonical.SerializeAsString());
    *model.mutable_graph() = canonical;
    EXPECT_EQ(core::builder::GraphBuilder(model, SchemaLookup())
                  .ApplyRewritings(plan)
                  .BuildGraph()
                  .SerializeAsString(),
              canonical.SerializeAsString());
    EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
              canonical.SerializeAsString());
  }
}

TEST(GraphGraph, TargetDeviceIncludesFunctionsAndRejectsConflictsAtomically) {
  using core::symbolic::Device;
  for (const bool conflict : {false, true}) {
    SCOPED_TRACE(conflict);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"local", 1}});
    auto &branch = builder.MakeSubgraph("branch");
    auto &function = builder.MakeLocalFunction("Body", "local");
    auto &nested = function.MakeLocalFunction("Nested", "local");
    auto &body_branch = nested.MakeSubgraph("body_branch");
    if (conflict)
      nested.set_device(Device::kCPU);
    core::builder::GraphGraph graph(builder);
    if (conflict) {
      EXPECT_THROW(graph.SetTargetDevice(Device::kGPU0), std::invalid_argument);
      EXPECT_EQ(builder.device(), Device::kUndefined);
      EXPECT_EQ(branch.device(), Device::kUndefined);
      EXPECT_EQ(function.device(), Device::kUndefined);
      EXPECT_EQ(nested.device(), Device::kCPU);
      EXPECT_EQ(body_branch.device(), Device::kUndefined);
    } else {
      graph.SetTargetDevice(Device::kGPU0);
      EXPECT_EQ(builder.device(), Device::kGPU0);
      EXPECT_EQ(branch.device(), Device::kGPU0);
      EXPECT_EQ(function.device(), Device::kGPU0);
      EXPECT_EQ(nested.device(), Device::kGPU0);
      EXPECT_EQ(body_branch.device(), Device::kGPU0);
    }
  }
}

TEST(GraphGraph, LocalFunctionsInheritDeviceForSelectedPatterns) {
  using core::symbolic::Device;
  for (const bool select_target : {false, true}) {
    SCOPED_TRACE(select_target);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"local", 1}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                     std::make_shared<ReplaceIndependentNeg>(false, Device::kGPU0)});
    if (select_target)
      graph.SetTargetDevice(Device::kGPU0);
    else
      builder.set_device(Device::kGPU0);
    auto &function = builder.MakeLocalFunction("Body", "local");
    function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
    auto &nested = function.MakeLocalFunction("Nested", "local");
    nested.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2}));
    nested.MakeNode("Neg", {"value"}, {"result"});
    nested.MakeOutput("result");
    function.MakeNode("Nested", {"arg"}, {"result"}, "local");
    function.MakeOutput("result");
    builder.MakeNode("Body", {"x"}, {"y"}, "local");
    builder.MakeOutput("y");
    core::builder::OptimizationReport report;
    graph.Optimize(-1, &report);
    EXPECT_GT(report.rewrites, 0u);
    EXPECT_EQ(function.device(), Device::kGPU0);
    EXPECT_EQ(nested.device(), Device::kGPU0);
    ASSERT_EQ(nested.Nodes().size(), 1u);
    EXPECT_EQ(nested.Nodes()[0].op_type(), "Abs");
  }
}

TEST(GraphGraph, InversePreservesPerScopeDevicesIncludingRestoredSubgraphs) {
  for (const bool remove_branch : {false, true}) {
    SCOPED_TRACE(remove_branch);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"local", 1}});
    builder.set_device(core::symbolic::Device::kGPU0);
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    auto &function = builder.MakeLocalFunction("Body", "local");
    function.set_device(core::symbolic::Device::kCPU);
    function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
    function.MakeInput("flag", core::symbolic::TensorType::kBool, MakeShape({}));
    utils::RepeatedProtoField<AttributeProto> function_attributes;
    for (const auto &name : {"body_then", "body_else"}) {
      auto &branch = function.MakeSubgraph(name);
      if (std::string(name) == "body_then")
        branch.set_device(core::symbolic::Device::kGPU0);
      branch.MakeNode("Neg", {"arg"}, {"body_result"});
      branch.MakeOutput("body_result");
      auto &reference = function_attributes.add();
      reference.set_name(std::string(name) == "body_then" ? "then_branch_ref" : "else_branch_ref");
      reference.set_type(AttributeProto::STRING);
      reference.set_s(name);
    }
    function.MakeNode("If", {"flag"}, {"result"}, "", "", function_attributes);
    function.MakeOutput("result");
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const auto &name : {"then_branch", "else_branch"}) {
      auto &branch = builder.MakeSubgraph(name);
      if (std::string(name) == "then_branch")
        branch.set_device(core::symbolic::Device::kCPU);
      branch.MakeNode("Neg", {"x"}, {"result"});
      branch.MakeOutput("result");
      auto &reference = attributes.add();
      reference.set_name(std::string(name) + "_ref");
      reference.set_type(AttributeProto::STRING);
      reference.set_s(name);
    }
    builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
    builder.MakeNode("Body", {"x", "condition"}, {"y"}, "local");
    builder.MakeOutput("y");
    if (!remove_branch)
      builder.MakeOutput("selected");
    const auto canonical = builder.BuildGraph().SerializeAsString();
    const auto plan =
        core::builder::GraphGraph(builder,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<ReplaceIndependentNeg>()})
            .Optimize();
    if (remove_branch) {
      EXPECT_FALSE(builder.HasSubgraph("then_branch"));
    }
    auto restored = builder.UndoRewritings(plan);
    EXPECT_EQ(restored.BuildGraph().SerializeAsString(), canonical);
    EXPECT_EQ(restored.device(), core::symbolic::Device::kGPU0);
    EXPECT_EQ(restored.Subgraph("then_branch").device(), core::symbolic::Device::kCPU);
    EXPECT_EQ(restored.Subgraph("else_branch").device(), core::symbolic::Device::kUndefined);
    EXPECT_EQ(restored.LocalFunction("Body").device(), core::symbolic::Device::kCPU);
    EXPECT_EQ(restored.LocalFunction("Body").Subgraph("body_then").device(),
              core::symbolic::Device::kGPU0);
    EXPECT_EQ(restored.LocalFunction("Body").Subgraph("body_else").device(),
              core::symbolic::Device::kUndefined);
  }
}

TEST(GraphGraph, InverseRestoresBranchOwnedFunctionsWithAndWithoutUnusedCleanup) {
  for (const bool remove_branch : {false, true}) {
    SCOPED_TRACE(remove_branch);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"local", 1}});
    builder.set_device(core::symbolic::Device::kGPU0);
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    auto &branch = builder.MakeSubgraph("then_branch");
    branch.set_device(core::symbolic::Device::kCPU);
    auto &function = branch.MakeLocalFunction("Body", "local");
    function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
    function.MakeInitializer(MakeInitializer<float>("bias", {2}, {1, 2}));
    function.MakeNode("Add", {"arg", "bias"}, {"result"});
    function.MakeOutput("result");
    function.set_device(core::symbolic::Device::kCPU);
    branch.MakeNode("Body", {"x"}, {"y"}, "local");
    branch.MakeOutput("y");
    auto &other = builder.MakeSubgraph("else_branch");
    other.MakeNode("Neg", {"x"}, {"y"});
    other.MakeOutput("y");
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const auto &name : {"then_branch", "else_branch"}) {
      auto &reference = attributes.add();
      reference.set_name(std::string(name) + "_ref");
      reference.set_type(AttributeProto::STRING);
      reference.set_s(name);
    }
    builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
    if (remove_branch) {
      builder.MakeNode("Abs", {"x"}, {"live"});
      builder.MakeOutput("live");
    } else {
      builder.MakeOutput("selected");
    }
    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
    graph.Canonicalize();
    const auto original_opsets = builder.Shapes().Opsets();
    const auto canonical = builder.BuildGraph().SerializeAsString();
    const auto node = function.Nodes()[0].SerializeAsString();
    const auto initializer = function.Initializers()[0].SerializeAsString();
    const auto tags = function.Compute().ValueTags();
    const auto root_tags = builder.Compute().CaptureTagState();
    const auto branch_tags = branch.Compute().CaptureTagState();
    core::builder::ConstantFoldingOptions options;
    options.enabled = false;
    const auto plan = graph.Optimize(1, nullptr, options);
    if (remove_branch) {
      ASSERT_FALSE(plan.rewrites().empty());
      EXPECT_FALSE(builder.HasSubgraph("then_branch"));
      EXPECT_FALSE(builder.HasSubgraph("else_branch"));
      ASSERT_EQ(builder.Nodes().size(), 1u);
      EXPECT_EQ(builder.Nodes()[0].op_type(), "Abs");
      EXPECT_TRUE(builder.ToModel().functions().empty());
    } else {
      EXPECT_TRUE(plan.rewrites().empty());
      EXPECT_TRUE(builder.HasSubgraph("then_branch"));
    }
    const auto optimized = builder.BuildGraph();
    auto restored = builder.UndoRewritings(plan);
    EXPECT_EQ(restored.BuildGraph().SerializeAsString(), canonical);
    EXPECT_EQ(restored.Shapes().Opsets(), original_opsets);
    ASSERT_TRUE(restored.HasSubgraph("then_branch"));
    ASSERT_TRUE(restored.HasSubgraph("else_branch"));
    ASSERT_EQ(restored.Subgraph("then_branch").Nodes().size(), 1u);
    EXPECT_EQ(restored.Subgraph("then_branch").Nodes()[0].op_type(), "Body");
    EXPECT_EQ(restored.Subgraph("then_branch").Nodes()[0].domain(), "local");
    EXPECT_EQ(restored.Compute().CaptureTagState(), root_tags);
    EXPECT_EQ(restored.Subgraph("then_branch").Compute().CaptureTagState(), branch_tags);
    EXPECT_EQ(restored.device(), core::symbolic::Device::kGPU0);
    EXPECT_EQ(restored.Subgraph("then_branch").device(), core::symbolic::Device::kCPU);
    ASSERT_TRUE(restored.HasLocalFunction("Body"));
    ASSERT_EQ(restored.LocalFunction("Body").Nodes().size(), 2u);
    EXPECT_EQ(restored.LocalFunction("Body").Nodes()[1].SerializeAsString(), node);
    EXPECT_EQ(restored.LocalFunction("Body").Nodes()[0].attribute(0).t().SerializeAsString(),
              initializer);
    EXPECT_EQ(restored.LocalFunction("Body").Compute().ValueTags(), tags);
    EXPECT_EQ(restored.LocalFunction("Body").device(), core::symbolic::Device::kCPU);
    const auto restored_model = restored.ToModel();
    ASSERT_EQ(restored_model.functions().size(), 1u);
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    auto replayed = restored.ApplyRewritings(loaded);
    EXPECT_TRUE(replayed.BuildGraph().Equals(optimized));
    const auto replayed_model = replayed.ToModel();
    EXPECT_NO_THROW(VerifyModel(replayed_model));
    // Replay retains the function hoisted by undo even when its branch is removed again.
    ASSERT_EQ(replayed_model.functions().size(), 1u);
    EXPECT_EQ(replayed.LocalFunction("Body").Shapes().Opsets(),
              restored.LocalFunction("Body").Shapes().Opsets());
    auto expected_function = restored_model.functions()[0];
    expected_function.ref_opset_import() = replayed_model.functions()[0].opset_import();
    std::string difference;
    EXPECT_TRUE(replayed_model.functions()[0].Equals(expected_function, &difference)) << difference;
  }
}

TEST(GraphGraph, CanonicalizesParentSubstitutionsBeforeNestedDuplicates) {
  for (const int depth : {1, 2}) {
    SCOPED_TRACE(depth);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    builder.MakeNode("Identity", {"x"}, {"a"});
    const auto make_body = [&](const auto &self, core::builder::GraphBuilder &scope,
                               int remaining) -> void {
      if (remaining == 0) {
        scope.MakeNode("Abs", {"a"}, {"p"});
        scope.MakeNode("Abs", {"x"}, {"q"});
        scope.MakeNode("Add", {"p", "q"}, {"sum"});
        scope.MakeNode("Neg", {"sum"}, {"y"});
      } else {
        utils::RepeatedProtoField<AttributeProto> attributes;
        for (const auto &name : {"then_branch", "else_branch"}) {
          const auto graph_name = scope.name() + "_" + name;
          auto &child = scope.MakeSubgraph(graph_name);
          self(self, child, remaining - 1);
          auto &reference = attributes.add();
          reference.set_name(std::string(name) + "_ref");
          reference.set_type(AttributeProto::STRING);
          reference.set_s(graph_name);
        }
        scope.MakeNode("If", {"condition"}, {"y"}, "", "", attributes);
      }
      scope.MakeOutput("y");
    };
    make_body(make_body, builder, depth);
    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                     std::make_shared<ReplaceIndependentNeg>()});
    EXPECT_EQ(graph.Canonicalize(), 1u + (1u << depth));
    const auto canonical = builder.BuildGraph();
    EXPECT_EQ(graph.Canonicalize(), 0u);
    EXPECT_EQ(builder.BuildGraph().SerializeAsString(), canonical.SerializeAsString());
    const auto plan = graph.Optimize();
    ASSERT_EQ(plan.rewrites().size(), 1u << depth);
    ModelProto reference;
    reference.add_opset("", 28);
    *reference.mutable_graph() = canonical;
    auto replayed = core::builder::GraphBuilder(reference, SchemaLookup()).ApplyRewritings(plan);
    EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), builder.BuildGraph().SerializeAsString());
    EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
              canonical.SerializeAsString());
  }
}

TEST(GraphGraph, InverseRejectsMalformedPlansAtomically) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Mul", {"x", "x"}, {"y"});
  builder.MakeOutput("y");
  core::builder::GraphGraph graph(builder,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<MarkReversedProducer>()});
  const auto plan = graph.Optimize();
  const auto optimized = builder.BuildGraph().SerializeAsString();
  ASSERT_EQ(plan.rewrites().size(), 1u);
  for (int kind = 0; kind < 5; ++kind) {
    SCOPED_TRACE(kind);
    auto invalid = plan;
    auto &edit = invalid.ref_rewrites()[0];
    if (kind == 0)
      invalid.clr_rewrites();
    else if (kind == 1)
      edit.ref_matched_nodes()[0] = 99;
    else if (kind == 2)
      edit.ref_added_nodes_positions()[0] = 99;
    else if (kind == 3)
      edit.ref_added_nodes()[0].ref_input()[0] = "undefined";
    else if (kind == 4)
      edit.ref_added_nodes()[0].set_op_type("Abs");
    EXPECT_THROW(builder.UndoRewritings(invalid), core::builder::BuilderError);
    EXPECT_EQ(builder.BuildGraph().SerializeAsString(), optimized);
  }
  EXPECT_THROW(builder.UndoRewritings(RewritingProto{}), core::builder::BuilderError);
  core::builder::GraphBuilder reloaded(builder.ToModel(), SchemaLookup());
  EXPECT_THROW(reloaded.UndoRewritings(plan), core::builder::BuilderError);
  core::builder::GraphBuilder moved(std::move(builder));
  EXPECT_NO_THROW(moved.UndoRewritings(plan));
  moved.MakeNode("Abs", {"y"}, {"extra"});
  EXPECT_THROW(moved.UndoRewritings(plan), core::builder::BuilderError);
}

TEST(GraphGraph, DeclarationMutationInvalidatesEmptyCaptureAfterMove) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Neg", {"x"}, {"y"});
  builder.MakeOutput("y");
  const auto canonical = builder.BuildGraph();
  const auto plan = core::builder::GraphGraph(
                        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{})
                        .Optimize();
  ASSERT_TRUE(plan.rewrites().empty());
  EXPECT_TRUE(builder.UndoRewritings(plan).BuildGraph().Equals(canonical));

  core::builder::GraphBuilder moved("destination", SchemaLookup(), {{"", 28}});
  moved = std::move(builder);
  EXPECT_TRUE(moved.UndoRewritings(plan).BuildGraph().Equals(canonical));
  moved.MakeOutput("x");
  ASSERT_EQ(moved.Nodes().size(), canonical.node().size());
  ASSERT_EQ(moved.Outputs().size(), 2u);
  const auto mutated = moved.BuildGraph();
  EXPECT_THROW(moved.UndoRewritings(plan), core::builder::BuilderError);
  EXPECT_TRUE(moved.BuildGraph().Equals(mutated));
}

TEST(GraphGraph, NestedDeclarationMutationInvalidatesAncestorCapture) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Neg", {"x"}, {"y"});
    branch.MakeOutput("y");
    auto &reference = attributes.add();
    reference.set_name(std::string(name) + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s(name);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  const auto plan = core::builder::GraphGraph(
                        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{})
                        .Optimize();
  ASSERT_TRUE(plan.rewrites().empty());
  EXPECT_NO_THROW(builder.UndoRewritings(plan));
  const auto node_count = builder.Nodes().size();
  const auto branch_node_count = builder.Subgraph("then_branch").Nodes().size();
  builder.Subgraph("then_branch").SetOpsetVersion("nested.test", 1);
  ASSERT_EQ(builder.Nodes().size(), node_count);
  ASSERT_EQ(builder.Subgraph("then_branch").Nodes().size(), branch_node_count);
  const auto mutated = builder.ToModel();
  EXPECT_THROW(builder.UndoRewritings(plan), core::builder::BuilderError);
  EXPECT_TRUE(builder.ToModel().Equals(mutated));
  EXPECT_EQ(builder.Subgraph("then_branch").OpsetVersion("nested.test"), 1);
}

TEST(GraphGraph, NestedRewriteCapturesOrdinaryImportsWithoutFunctionImportConflicts) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"local", 1}});
  builder.set_device(core::symbolic::Device::kGPU0);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  auto &function = builder.MakeLocalFunction("Body", "local");
  function.SetOpsetVersion("ai.onnx.ml", 2);
  function.set_device(core::symbolic::Device::kCPU);
  function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
  function.MakeNode("Scaler", {"arg"}, {"result"}, "ai.onnx.ml");
  function.MakeOutput("result");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    if (std::string(name) == "then_branch") {
      branch.set_device(core::symbolic::Device::kCPU);
      branch.MakeNode("Neg", {"x"}, {"y"}, "", "nested_rewrite");
    } else {
      branch.MakeNode("Body", {"x"}, {"y"}, "local");
    }
    branch.MakeOutput("y");
    auto &reference = attributes.add();
    reference.set_name(std::string(name) + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s(name);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  builder.MakeOutput("selected");
  core::builder::GraphGraph optimizer(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                   std::make_shared<ReplaceNestedNegWithScaler>()});
  optimizer.Canonicalize();
  const auto canonical = builder.BuildGraph();
  const auto original_opsets = builder.Shapes().Opsets();
  const auto tags = builder.Compute().CaptureTagState();
  const auto branch_tags = builder.Subgraph("then_branch").Compute().CaptureTagState();
  const auto function_tags = function.Compute().CaptureTagState();
  const auto function_body = function.BuildGraph();
  const auto root_ml_version = builder.OpsetVersion("ai.onnx.ml");
  core::builder::ConstantFoldingOptions options;
  options.enabled = false;
  const auto plan = optimizer.Optimize(1, nullptr, options);
  ASSERT_EQ(plan.rewrites().size(), 1u);
  ASSERT_EQ(plan.rewrites(0).graph_path().size(), 1u);
  EXPECT_EQ(plan.rewrites(0).graph_path()[0], "then_branch");
  ASSERT_EQ(plan.required_opset_imports().size(), 1u);
  EXPECT_EQ(plan.required_opset_imports()[0].domain(), "ai.onnx.ml");
  EXPECT_EQ(plan.required_opset_imports()[0].version(), 1);
  EXPECT_TRUE(plan.added_functions().empty());
  EXPECT_EQ(builder.OpsetVersion("ai.onnx.ml"), 1);
  EXPECT_EQ(builder.Subgraph("then_branch").OpsetVersion("ai.onnx.ml"), 1);
  EXPECT_EQ(builder.Subgraph("then_branch").Nodes()[0].op_type(), "Scaler");
  const auto optimized = builder.BuildGraph();

  auto restored = builder.UndoRewritings(plan);
  std::string difference;
  EXPECT_TRUE(restored.BuildGraph().Equals(canonical, &difference)) << difference;
  EXPECT_EQ(restored.Shapes().Opsets(), original_opsets);
  EXPECT_EQ(restored.Compute().CaptureTagState(), tags);
  EXPECT_EQ(restored.Subgraph("then_branch").Compute().CaptureTagState(), branch_tags);
  EXPECT_EQ(restored.LocalFunction("Body").Compute().CaptureTagState(), function_tags);
  EXPECT_TRUE(restored.LocalFunction("Body").BuildGraph().Equals(function_body));
  EXPECT_EQ(restored.device(), core::symbolic::Device::kGPU0);
  EXPECT_EQ(restored.Subgraph("then_branch").device(), core::symbolic::Device::kCPU);
  EXPECT_EQ(restored.LocalFunction("Body").device(), core::symbolic::Device::kCPU);
  EXPECT_EQ(restored.OpsetVersion("ai.onnx.ml"), root_ml_version);
  EXPECT_EQ(restored.LocalFunction("Body").OpsetVersion("ai.onnx.ml"), 2);
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
  auto replayed = restored.ApplyRewritings(loaded);
  EXPECT_TRUE(replayed.BuildGraph().Equals(optimized, &difference)) << difference;
  const auto exported = replayed.ToModel();
  EXPECT_NO_THROW(VerifyModel(exported));
  std::map<std::string, int64_t> exported_imports;
  for (const auto &opset : exported.opset_import())
    exported_imports.emplace(opset.domain().value(), opset.version());
  std::map<std::string, int64_t> expected_imports(original_opsets.begin(), original_opsets.end());
  expected_imports.emplace("ai.onnx.ml", 1);
  EXPECT_EQ(exported_imports, expected_imports);
  EXPECT_EQ(replayed.LocalFunction("Body").OpsetVersion("ai.onnx.ml"), 2);
  auto missing_import = loaded;
  missing_import.clr_required_opset_imports();
  EXPECT_THROW(restored.ApplyRewritings(missing_import), core::builder::BuilderError);
  core::builder::GraphBuilder conflicting(restored.ToModel(), SchemaLookup());
  conflicting.SetOpsetVersion("ai.onnx.ml", 3);
  const auto before_conflict = conflicting.ToModel();
  EXPECT_THROW(conflicting.ApplyRewritings(loaded), core::builder::BuilderError);
  EXPECT_TRUE(conflicting.ToModel().Equals(before_conflict));
}

TEST(GraphGraph, CaptureKeepsNewIdentityAndDistinctConsumerNames) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Neg", {"a"}, {"b"});
  builder.MakeNode("Abs", {"b"}, {"left"});
  builder.MakeNode("Abs", {"a"}, {"right"});
  builder.MakeNode("Add", {"left", "right"}, {"y"});
  builder.MakeOutput("y");
  const GraphProto canonical = builder.BuildGraph();
  core::builder::GraphGraph graph(builder,
                                  std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                                      std::make_shared<ReplaceNegWithIdentity>()});
  const auto plan = graph.Optimize();
  ASSERT_EQ(plan.rewrites().size(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 4u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Identity");
  EXPECT_EQ(builder.Nodes()[1].input(0), "b");
  EXPECT_EQ(builder.Nodes()[2].input(0), "a");
  EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
            canonical.SerializeAsString());
}

TEST(GraphGraph, InverseRestoresValueInfoOrderAndExistingDependencies) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}, {"existing", 1}});
  source.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  auto &function = source.MakeLocalFunction("Existing", "existing");
  function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
  function.MakeNode("Identity", {"arg"}, {"result"});
  function.MakeOutput("result");
  source.MakeNode("Abs", {"x"}, {"first_dead"});
  source.MakeNode("Neg", {"x"}, {"second_dead"});
  source.MakeNode("Mul", {"x", "x"}, {"y"});
  source.MakeOutput("y");
  ModelProto model;
  model.add_opset("", 28);
  model.add_opset("existing", 1);
  model.ref_functions().push_back(function.ToFunction("existing"));
  *model.mutable_graph() = source.BuildGraph();
  for (const auto &name : {"second_dead", "first_dead"}) {
    auto &value = *model.mutable_graph()->add_value_info();
    value.set_name(name);
    auto &metadata = *value.add_metadata_props();
    metadata.set_key("annotation");
    metadata.set_value(name);
  }
  core::builder::GraphBuilder builder(model, SchemaLookup());
  const auto canonical = builder.BuildGraph();
  core::builder::GraphGraph graph(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
  const auto plan = graph.Optimize();
  EXPECT_TRUE(builder.BuildGraph().value_info().empty());
  EXPECT_TRUE(plan.added_functions().empty());
  auto restored = builder.UndoRewritings(plan);
  EXPECT_TRUE(restored.HasLocalFunction("Existing"));
  EXPECT_EQ(restored.OpsetVersion("existing"), 1);
  EXPECT_EQ(restored.BuildGraph().SerializeAsString(), canonical.SerializeAsString());
  ASSERT_EQ(restored.BuildGraph().value_info().size(), 2u);
  EXPECT_EQ(restored.BuildGraph().value_info(0).name(), "second_dead");
  EXPECT_EQ(restored.BuildGraph().value_info(1).name(), "first_dead");
  auto redundant_dependencies = plan;
  redundant_dependencies.ref_added_functions().push_back(model.functions(0));
  auto &opset = *redundant_dependencies.add_required_opset_imports();
  opset.set_domain("existing");
  opset.set_version(1);
  EXPECT_THROW(builder.UndoRewritings(redundant_dependencies), core::builder::BuilderError);
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
    const auto *encoded_payload = std::as_const(encoded).raw_data().data();
    EXPECT_EQ(builder.Shapes().GetEncodedValue("encoded").raw_data().data(), encoded_payload);
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
    const ModelProto original = builder.ToModel();
    EXPECT_EQ(original.graph().extended_initializer(0).encoded_value().raw_data().data(),
              encoded_payload);
    const GraphProto canonical = builder.BuildGraph();
    core::builder::GraphGraph optimizer(builder);
    const auto plan = optimizer.Optimize();
    ASSERT_EQ(plan.rewrites().size(), 1u);
    EXPECT_EQ(plan.rewrites(0).removed_extended_initializers().values(),
              std::vector<uint64_t>({0, 1}));
    EXPECT_TRUE(builder.ExtendedInitializers().empty());
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded);
    EXPECT_EQ(replayed.ToGraph().SerializeAsString(), builder.ToGraph().SerializeAsString());
    const auto restored = builder.UndoRewritings(loaded).BuildGraph();
    EXPECT_EQ(restored.SerializeAsString(), canonical.SerializeAsString());
    EXPECT_EQ(restored.extended_initializer(0).encoded_value().raw_data().data(), encoded_payload);
    for (const std::string name : {"encoded", "cache"}) {
      EXPECT_FALSE(builder.Shapes().HasEncodedValue(name));
      EXPECT_FALSE(builder.Shapes().HasType(name));
      EXPECT_FALSE(builder.Compute().IsConstantValue(name));
      EXPECT_EQ(builder.Compute().ValueTags().count(name), 0u);
      EXPECT_FALSE(replayed.Shapes().HasEncodedValue(name));
      EXPECT_FALSE(replayed.Shapes().HasType(name));
      EXPECT_FALSE(replayed.Compute().IsConstantValue(name));
      EXPECT_EQ(replayed.Compute().ValueTags().count(name), 0u);
    }
  }
}

TEST(GraphGraph, CleanupReplaysEachInitializerKindAndPreservesLiveValues) {
  for (const int kinds : {1, 2, 4, 7}) {
    for (const bool add_dead_node : {false, true}) {
      SCOPED_TRACE(kinds);
      SCOPED_TRACE(add_dead_node);
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
      builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
      builder.MakeOutput("X");
      if (kinds & 1) {
        builder.MakeInitializer(MakeInitializer<float>("live_dense", {1}, {1.f}));
        builder.MakeInitializer(MakeInitializer<float>("dead_dense", {1}, {2.f}));
        builder.MakeOutput("live_dense");
      }
      if (kinds & 2) {
        builder.MakeStructType(RewriteRecord());
        EncodedValueProto encoded;
        encoded.set_name("live_encoded");
        encoded.mutable_struct_type()->set_type_ref(101);
        encoded.set_raw_data(std::string(sizeof(float), '\0'));
        builder.MakeEncodedInitializer(encoded);
        encoded.set_name("dead_encoded");
        builder.MakeEncodedInitializer(encoded);
        builder.MakeOutput("live_encoded");
      }
      if (kinds & 4) {
        PagedCacheProto cache;
        cache.set_name("live_cache");
        auto *block = cache.add_blocks();
        block->set_start(0);
        block->set_length(1);
        *block->mutable_key() = MakeInitializer<float>("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
        *block->mutable_value() = block->key();
        builder.MakePagedCacheInitializer(cache);
        cache.set_name("dead_cache");
        builder.MakePagedCacheInitializer(cache);
        builder.MakeOutput("live_cache");
      }
      if (add_dead_node) {
        builder.MakeNode("Abs", {"X"}, {"dead"});
        builder.MakeNode("Neg", {"X"}, {"Y"});
        builder.MakeOutput("Y");
      }
      core::builder::GraphGraph optimizer(builder);
      optimizer.Canonicalize();
      const ModelProto original = builder.ToModel();
      const auto plan = optimizer.Optimize();
      ASSERT_EQ(plan.rewrites().size(), (kinds == 2 && !add_dead_node) ? 0u : 1u);
      if (!plan.rewrites().empty()) {
        const auto &edit = plan.rewrites(0);
        EXPECT_EQ(edit.removed_initializers().values(),
                  (kinds & 1) ? std::vector<uint64_t>({1}) : std::vector<uint64_t>());
        std::vector<uint64_t> removed_extended;
        if (kinds & 4)
          removed_extended.push_back((kinds & 2) ? 2 : 1);
        EXPECT_EQ(edit.removed_extended_initializers().values(), removed_extended);
      }
      RewritingProto loaded;
      ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
      auto expected = builder.ToGraph();
      if (add_dead_node) {
        auto *follow_up = loaded.add_rewrites();
        follow_up->add_matched_nodes(0);
        follow_up->ref_added_nodes().push_back(MakeNode("Abs", {"X"}, {"Y"}));
        follow_up->add_added_nodes_positions(0);
        expected.ref_node()[0].set_op_type("Abs");
      }
      auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded);
      EXPECT_EQ(replayed.ToGraph().SerializeAsString(), expected.SerializeAsString());
    }
  }
}

TEST(GraphGraph, ReplaysUnifiedInitializerPositionsWithoutReordering) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  source.MakeOutput("X");
  source.MakeStructType(RewriteRecord());
  EncodedValueProto encoded;
  encoded.set_name("unused_encoded");
  encoded.mutable_struct_type()->set_type_ref(101);
  encoded.set_raw_data(std::string(sizeof(float), '\0'));
  source.MakeEncodedInitializer(encoded);
  PagedCacheProto cache;
  cache.set_name("live_cache");
  auto *block = cache.add_blocks();
  block->set_start(0);
  block->set_length(1);
  *block->mutable_key() = MakeInitializer<float>("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
  *block->mutable_value() = block->key();
  source.MakePagedCacheInitializer(cache);
  source.MakeOutput("live_cache");
  ModelProto model = source.ToModel();
  auto &initializers = model.ref_graph().ref_extended_initializer();
  std::swap(initializers.get(0), initializers.get(1));
  core::builder::GraphBuilder builder(model, SchemaLookup());
  ASSERT_TRUE(builder.ExtendedInitializers()[0].has_paged_cache());
  EXPECT_EQ(builder.ToGraph().extended_initializer(1).name(), "unused_encoded");
  core::builder::GraphGraph optimizer(builder);
  const auto plan = optimizer.Optimize();
  ASSERT_EQ(plan.rewrites().size(), 1u);
  EXPECT_EQ(plan.rewrites(0).removed_extended_initializers().values(), std::vector<uint64_t>({1}));
  auto replayed = core::builder::GraphBuilder(model, SchemaLookup()).ApplyRewritings(plan);
  EXPECT_EQ(replayed.ToGraph().SerializeAsString(), builder.ToGraph().SerializeAsString());
  ASSERT_EQ(replayed.ExtendedInitializers().size(), 1u);
  EXPECT_TRUE(replayed.ExtendedInitializers()[0].has_paged_cache());
  EXPECT_FALSE(replayed.Shapes().HasEncodedValue("unused_encoded"));

  model.mutable_graph()->add_extended_initializer();
  EXPECT_THROW((core::builder::GraphBuilder(model, SchemaLookup())), core::builder::BuilderError);
}

TEST(GraphGraph, ReplayRejectsInvalidStructuredInitializerRemovals) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Neg", {"X"}, {"Y"});
  builder.MakeOutput("Y");
  EncodedValueProto encoded;
  encoded.set_name("encoded");
  builder.MakeStructType(RewriteRecord());
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
  const auto original = builder.BuildGraph().SerializeAsString();
  for (const bool paged : {false, true}) {
    for (const uint64_t position : {uint64_t{0}, uint64_t{2}, UINT64_MAX}) {
      RewritingProto plan;
      auto *node_edit = plan.add_rewrites();
      node_edit->add_matched_nodes(0);
      node_edit->ref_added_nodes().push_back(MakeNode("Abs", {"X"}, {"Y"}));
      node_edit->add_added_nodes_positions(0);
      auto *edit = plan.add_rewrites();
      edit->add_matched_nodes(0);
      edit->ref_added_nodes().push_back(MakeNode("Neg", {"X"}, {"Y"}));
      edit->add_added_nodes_positions(0);
      auto &positions = edit->ref_removed_extended_initializers();
      positions.push_back(position == 0 && paged ? 1 : position);
      if (position == 0)
        positions.push_back(paged ? 1 : 0);
      EXPECT_THROW(builder.ApplyRewritings(plan), core::builder::BuilderError);
      EXPECT_EQ(builder.BuildGraph().SerializeAsString(), original);
    }
    RewritingProto replacement;
    auto &edit = *replacement.add_rewrites();
    edit.add_removed_extended_initializers(paged ? 1 : 0);
    edit.add_added_initializers(MakeInitializer<float>(paged ? "cache" : "encoded", {1}, {1.f}));
    edit.add_added_initializer_positions(0);
    EXPECT_THROW(builder.ApplyRewritings(replacement), core::builder::BuilderError);
    EXPECT_EQ(builder.BuildGraph().SerializeAsString(), original);
  }
}

TEST(GraphGraph, CleanupReplaysDuplicateEncodedInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeOutput("X");
  builder.MakeStructType(RewriteRecord());
  EncodedValueProto encoded;
  encoded.set_name("first");
  encoded.mutable_struct_type()->set_type_ref(101);
  encoded.set_raw_data(std::string(sizeof(float), '\0'));
  builder.MakeEncodedInitializer(encoded);
  encoded.set_name("duplicate");
  builder.MakeEncodedInitializer(encoded);
  core::builder::GraphGraph optimizer(builder);
  EXPECT_EQ(optimizer.Canonicalize(), 1u);
  const ModelProto original = builder.ToModel();
  const GraphProto canonical = builder.BuildGraph();
  const auto plan = optimizer.Optimize(-1, nullptr, {}, false);
  EXPECT_TRUE(plan.rewrites().empty());
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
  auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded);
  EXPECT_EQ(replayed.ToGraph().SerializeAsString(), builder.ToGraph().SerializeAsString());
  EXPECT_FALSE(replayed.Shapes().HasEncodedValue("duplicate"));
  EXPECT_TRUE(replayed.Shapes().HasEncodedValue("first"));
  EXPECT_EQ(builder.UndoRewritings(loaded).BuildGraph().SerializeAsString(),
            canonical.SerializeAsString());
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
    if (batch_dependency)
      builder.MakeInput("D", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    builder.MakeNode("Neg", {"X"}, {"p"}, "", "rewrite");
    builder.MakeNode("Relu", {"p"}, {"C"});
    builder.MakeNode("Exp", {"X"}, {"I"});
    builder.MakeNode(batch_dependency ? "Neg" : "Abs", {batch_dependency ? "D" : "X"}, {"a"});
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
  for (const bool expand : {false, true}) {
    SCOPED_TRACE(expand);
    const std::size_t count = expand ? 128 : 512;
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInitializer(MakeInitializer<float>("unused_weight", {2}, {1, 2}));
    for (std::size_t index = 0; index < count; ++index) {
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
                     std::make_shared<ReplaceIndependentNeg>(expand)});
    core::builder::ConstantFoldingOptions options;
    options.enabled = false;
    const auto plan = optimizer.Optimize(1, nullptr, options, /*remove_unused=*/false);
    EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
              original.graph().SerializeAsString());
    ASSERT_EQ(plan.rewrites().size(), count);
    for (std::size_t index = 0; index < plan.rewrites().size(); ++index) {
      const auto &edit = plan.rewrites(index);
      EXPECT_EQ(edit.matched_nodes(0), index * (expand ? 2 : 1));
      EXPECT_EQ(edit.added_nodes_positions(0), index * (expand ? 2 : 1));
    }
    auto replayed = core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan);
    EXPECT_EQ(replayed.BuildGraph().SerializeAsString(), builder.BuildGraph().SerializeAsString());
  }
}

TEST(GraphGraph, ExportCaptureAndUndoShareOwnedInitializerAllocations) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({32768}));
  TensorProto weight;
  weight.set_name("W");
  weight.set_data_type(TensorProto::FLOAT);
  weight.add_dims(32768);
  std::vector<float> values(32768, 1.f);
  weight.set_raw_data(values.data(), values.size() * sizeof(float));
  const auto *payload = weight.raw_data().data();
  source.MakeInitializer(weight);
  weight.set_name("unused");
  weight.ref_raw_data()[0] = 42;
  const auto *removed_payload = weight.raw_data().data();
  source.MakeInitializer(weight);
  source.MakeNode("Add", {"X", "W"}, {"Y"});
  source.MakeOutput("Y");
  auto canonical = source.ToModel();
  core::builder::GraphBuilder working(canonical, SchemaLookup());
  ASSERT_EQ(working.Initializers().size(), 2u);
  EXPECT_EQ(source.Initializers()[0].raw_data().data(), payload);
  EXPECT_EQ(canonical.graph().initializer(0).raw_data().data(), payload);
  EXPECT_EQ(working.Initializers()[0].raw_data().data(), payload);
  EXPECT_EQ(working.Initializers()[1].raw_data().data(), removed_payload);

  core::builder::GraphGraph optimizer(
      working, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
  core::builder::ConstantFoldingOptions options;
  options.enabled = false;
  const auto plan = optimizer.Optimize(1, nullptr, options);
  ASSERT_EQ(working.Initializers().size(), 1u);
  auto restored = working.UndoRewritings(plan);
  auto exported = restored.ToModel();
  EXPECT_TRUE(exported.Equals(canonical));
  ASSERT_EQ(restored.Initializers().size(), 2u);
  EXPECT_EQ(restored.Initializers()[0].raw_data().data(), payload);
  EXPECT_EQ(restored.Initializers()[1].raw_data().data(), removed_payload);
  EXPECT_EQ(exported.graph().initializer(0).raw_data().data(), payload);
  EXPECT_EQ(exported.graph().initializer(1).raw_data().data(), removed_payload);

  exported.mutable_graph()->ref_initializer()[0].ref_raw_data()[0] = 17;
  EXPECT_NE(exported.graph().initializer(0).raw_data().data(), payload);
  EXPECT_EQ(working.Initializers()[0].raw_data()[0], 0);
  EXPECT_EQ(source.Initializers()[0].raw_data()[0], 0);
  EXPECT_EQ(restored.Initializers()[0].raw_data()[0], 0);
  canonical.mutable_graph()->ref_initializer()[1].set_raw_data("changed");
  EXPECT_EQ(restored.Initializers()[1].raw_data().data(), removed_payload);
  EXPECT_EQ(restored.Compute().CaptureTagState(), source.Compute().CaptureTagState());
  EXPECT_EQ(restored.Compute().Memory(), source.Compute().Memory());
  EXPECT_EQ(restored.Compute().PeakMemory(), source.Compute().PeakMemory());
}

TEST(GraphGraph, NestedAndFunctionPayloadsRetainBorrowedOwnersThroughUndo) {
  for (const bool function_scope : {false, true}) {
    SCOPED_TRACE(function_scope);
    auto owner = std::make_shared<std::vector<float>>(2, 3.f);
    std::weak_ptr<std::vector<float>> lifetime = owner;
    const auto *payload = reinterpret_cast<const uint8_t *>(owner->data());
    TensorProto weight;
    weight.set_name("W");
    weight.set_data_type(TensorProto::FLOAT);
    weight.add_dims(2);
    weight.ref_raw_data().assign_borrowed(payload, 2 * sizeof(float), owner);
    ModelProto canonical;
    ModelProto exported;
    {
      core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
      source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
      auto &scope = function_scope ? source.MakeLocalFunction("Body", "local")
                                   : source.MakeSubgraph("branch");
      if (function_scope)
        scope.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
      scope.MakeInitializer(weight);
      scope.MakeNode("Add", {"X", "W"}, {"Y"});
      scope.MakeOutput("Y");
      if (function_scope) {
        source.SetOpsetVersion("local", 1);
        source.MakeNode("Body", {"X"}, {"Z"}, "local");
      } else {
        source.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
        auto &other = source.MakeSubgraph("other");
        other.MakeNode("Neg", {"X"}, {"E"});
        other.MakeOutput("E");
        utils::RepeatedProtoField<AttributeProto> attributes;
        for (const std::string name : {"then_branch", "else_branch"}) {
          auto &reference = attributes.add();
          reference.set_name(name + "_ref");
          reference.set_type(AttributeProto::STRING);
          reference.set_s(name == "then_branch" ? "branch" : "other");
        }
        source.MakeNode("If", {"condition"}, {"Z"}, "", "", attributes);
      }
      source.MakeOutput("Z");
      canonical = source.ToModel();
      core::builder::GraphBuilder working(canonical, SchemaLookup());
      // Import materializes function initializers as Constant nodes with their own scheduling.
      auto imported = working.ToModel();
      core::builder::GraphGraph optimizer(
          working, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
      core::builder::ConstantFoldingOptions options;
      options.enabled = false;
      auto plan = optimizer.Optimize(1, nullptr, options);
      exported = working.UndoRewritings(plan).ToModel();
      const auto sort_opsets = [](auto &proto) {
        std::map<std::string, int64_t> opsets;
        for (const auto &opset : proto.opset_import())
          opsets.emplace(opset.domain().value(), opset.version());
        proto.clear_opset_import();
        for (const auto &[domain, version] : opsets)
          proto.add_opset(domain, version);
      };
      for (ModelProto *model : {&exported, &imported}) {
        sort_opsets(*model);
        for (auto &function : model->ref_functions())
          sort_opsets(function);
      }
      std::string difference;
      EXPECT_TRUE(exported.Equals(imported, &difference)) << difference;
      owner.reset();
      weight.Clear();
    }
    canonical.Clear();
    EXPECT_FALSE(lifetime.expired());
    const auto &tensor = function_scope ? exported.functions(0).node(0).attribute(0).t()
                                        : exported.graph().node(0).attribute(0).g().initializer(0);
    EXPECT_EQ(tensor.raw_data().data(), payload);
    EXPECT_EQ(tensor.raw_data().size(), 2 * sizeof(float));
    exported.Clear();
    EXPECT_TRUE(lifetime.expired());
  }
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

TEST(GraphGraph, InverseAcceptsCapturedTagsThatDifferFromReimportedTags) {
  onnx_shapes::RegisterShapeFunctions();
  core::builder::GraphBuilder body("WithGraph", SchemaLookup(), {{"", 28}});
  body.MakeInput("Y", core::symbolic::TensorType::kInt64, MakeShape({2}));
  body.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  utils::RepeatedProtoField<AttributeProto> references;
  for (const char *name : {"then_branch", "else_branch"}) {
    auto &attribute = references.add();
    attribute.set_name(name);
    attribute.set_ref_attr_name(name);
    attribute.set_type(AttributeProto::GRAPH);
  }
  body.MakeNode("If", {"condition"}, {"result"}, "", "", references);
  body.MakeOutput("result", core::symbolic::TensorType::kInt64, MakeShape({2}));
  auto function = body.ToFunction("local");
  function.add_attribute("then_branch");
  function.add_attribute("else_branch");
  for (const bool nested : {false, true}) {
    SCOPED_TRACE(nested);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"local", 1}});
    builder.MakeLocalFunction(function);
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    auto *scope = &builder;
    if (nested) {
      scope = &builder.MakeSubgraph("then_branch");
    }
    scope->MakeInitializer(MakeInitializer<int64_t>("W", {2}, {1, 2}));
    scope->MakeNode("Shape", {"W"}, {"S"});
    scope->MakeNode("Add", {"W", "S"}, {"Y"});
    utils::RepeatedProtoField<AttributeProto> arguments;
    for (const char *name : {"then_branch", "else_branch"}) {
      auto &attribute = arguments.add();
      attribute.set_name(name);
      attribute.set_type(AttributeProto::GRAPH);
      auto &branch = *attribute.mutable_g();
      branch.set_name(std::string("argument_") + name);
      branch.add_node(MakeNode("Neg", {"Y"}, {"branch_result"}));
      branch.add_output()->set_name("branch_result");
    }
    scope->MakeNode("WithGraph", {"Y", "condition"}, {"Z"}, "local", "", arguments);
    scope->MakeOutput("Z");
    if (nested) {
      auto &other = builder.MakeSubgraph("else_branch");
      other.MakeInitializer(MakeInitializer<int64_t>("Z", {2}, {1, 2}));
      other.MakeOutput("Z");
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const auto &name : {"then_branch", "else_branch"}) {
        auto &reference = attributes.add();
        reference.set_name(std::string(name) + "_ref");
        reference.set_type(AttributeProto::STRING);
        reference.set_s(name);
      }
      builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
      builder.MakeOutput("selected");
    }
    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                     std::make_shared<SwapAddInputs>()});
    graph.Canonicalize();
    const auto canonical = builder.BuildGraph().SerializeAsString();
    const auto tags = scope->Compute().CaptureTagState();
    const auto canonical_memory = scope->Compute().Memory();
    const auto canonical_peak_memory = scope->Compute().PeakMemory();
    const auto canonical_releases = scope->Compute().ReleaseAfterShapeTagged();
    core::builder::ConstantFoldingOptions options;
    options.enabled = false;
    const auto plan = graph.Optimize(1, nullptr, options);
    ASSERT_EQ(plan.rewrites().size(), 1u);
    EXPECT_EQ(scope->Compute().ValueTags().at("Y"), "weight");
    const auto optimized = builder.BuildGraph().SerializeAsString();
    EXPECT_THROW(builder.ApplyRewritings(RewritingProto{}), core::builder::BuilderError);
    auto restored = builder.UndoRewritings(plan);
    const auto &restored_scope = nested ? restored.Subgraph("then_branch") : restored;
    EXPECT_EQ(restored.BuildGraph().SerializeAsString(), canonical);
    EXPECT_EQ(restored_scope.Compute().CaptureTagState(), tags);
    EXPECT_EQ(restored_scope.Compute().Memory(), canonical_memory);
    EXPECT_EQ(restored_scope.Compute().PeakMemory(), canonical_peak_memory);
    EXPECT_EQ(restored_scope.Compute().ReleaseAfterShapeTagged(), canonical_releases);
    EXPECT_EQ(builder.BuildGraph().SerializeAsString(), optimized);
    const auto optimized_tags = scope->Compute().CaptureTagState();
    const auto memory = scope->Compute().Memory();
    const auto peak_memory = scope->Compute().PeakMemory();
    const auto releases = scope->Compute().ReleaseAfterShapeTagged();
    core::builder::GraphGraph no_patterns(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
    const auto empty_plan = no_patterns.Optimize(1, nullptr, options);
    ASSERT_TRUE(empty_plan.rewrites().empty());
    auto unchanged = builder.UndoRewritings(empty_plan);
    const auto &unchanged_scope = nested ? unchanged.Subgraph("then_branch") : unchanged;
    EXPECT_EQ(unchanged.BuildGraph().SerializeAsString(), optimized);
    EXPECT_EQ(unchanged_scope.Compute().CaptureTagState(), optimized_tags);
    EXPECT_EQ(unchanged_scope.Compute().Memory(), memory);
    EXPECT_EQ(unchanged_scope.Compute().PeakMemory(), peak_memory);
    EXPECT_EQ(unchanged_scope.Compute().ReleaseAfterShapeTagged(), releases);
    EXPECT_EQ(scope->Compute().CaptureTagState(), optimized_tags);
  }
}

TEST(GraphGraph, ReplayRejectsReconstructedTagChangesBeforeFollowUpRewrites) {
  for (const bool nested : {false, true}) {
    SCOPED_TRACE(nested);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
    auto *scope = &source;
    if (nested) {
      source.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
      scope = &source.MakeSubgraph("then_branch");
    }
    scope->MakeInitializer(MakeInitializer<int64_t>("W", {2}, {1, 2}));
    scope->MakeNode("Shape", {"W"}, {"S"});
    scope->MakeNode("Add", {"W", "S"}, {"Y"});
    scope->MakeNode("Identity", {"Y"}, {"Z"});
    scope->MakeOutput("Z");
    if (nested) {
      auto &other = source.MakeSubgraph("else_branch");
      other.MakeInitializer(MakeInitializer<int64_t>("Z", {2}, {1, 2}));
      other.MakeOutput("Z");
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const std::string name : {"then_branch", "else_branch"}) {
        auto &reference = attributes.add();
        reference.set_name(name + "_ref");
        reference.set_type(AttributeProto::STRING);
        reference.set_s(name);
      }
      source.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
      source.MakeOutput("selected");
    }
    RewritingProto first;
    auto &swap = *first.add_rewrites();
    if (nested)
      swap.add_graph_path("then_branch");
    swap.add_matched_nodes(1);
    swap.add_added_nodes_positions(1);
    swap.add_added_nodes(MakeNode("Add", {"S", "W"}, {"Y"}));
    auto rewritten = source.ApplyRewritings(first);
    const auto &rewritten_scope = nested ? rewritten.Subgraph("then_branch") : rewritten;
    EXPECT_EQ(rewritten_scope.Compute().ValueTags().at("Y"), "weight");
    EXPECT_EQ(rewritten_scope.Compute().ValueTags().at("Z"), "weight");
    const auto before = rewritten.BuildGraph().SerializeAsString();
    const auto tags = rewritten_scope.Compute().ValueTags();
    const auto node_tags = rewritten_scope.Compute().NodeTags();
    const auto memory = rewritten_scope.Compute().Memory();
    const auto peak_memory = rewritten_scope.Compute().PeakMemory();
    RewritingProto follow_up;
    auto &replace = *follow_up.add_rewrites();
    if (nested)
      replace.add_graph_path("then_branch");
    replace.add_matched_nodes(2);
    replace.add_added_nodes_positions(2);
    replace.add_added_nodes(MakeNode("Neg", {"Y"}, {"Z"}));
    ModelProto fresh_model;
    *fresh_model.mutable_graph() = rewritten.BuildGraph();
    fresh_model.add_opset("", 28);
    const auto fresh =
        core::builder::GraphBuilder(fresh_model, SchemaLookup()).ApplyRewritings(follow_up);
    const auto &fresh_scope = nested ? fresh.Subgraph("then_branch") : fresh;
    EXPECT_EQ(fresh_scope.Nodes()[2].op_type(), "Neg");
    for (const auto &plan : {RewritingProto{}, follow_up}) {
      try {
        rewritten.ApplyRewritings(plan);
        FAIL() << "Expected reconstructed tags to be rejected.";
      } catch (const core::builder::BuilderError &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("ApplyRewritings"), std::string::npos);
        EXPECT_TRUE(message.find("'Y'") != std::string::npos ||
                    message.find("'Z'") != std::string::npos);
        EXPECT_NE(message.find("existing tag 'weight'"), std::string::npos);
        EXPECT_NE(message.find("reconstructed tag 'shape'"), std::string::npos);
        if (nested) {
          EXPECT_NE(message.find("then_branch"), std::string::npos);
        }
      }
      EXPECT_EQ(rewritten.BuildGraph().SerializeAsString(), before);
      EXPECT_EQ(rewritten_scope.Compute().ValueTags(), tags);
      EXPECT_EQ(rewritten_scope.Compute().NodeTags(), node_tags);
      EXPECT_EQ(rewritten_scope.Compute().Memory(), memory);
      EXPECT_EQ(rewritten_scope.Compute().PeakMemory(), peak_memory);
    }
  }
}

TEST(GraphGraph, ReplayValidatesScopedTagsAndAllowsUntaggedValues) {
  for (const bool function : {false, true}) {
    SCOPED_TRACE(function);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
    source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
    auto &scope =
        function ? source.MakeLocalFunction("Body", "local") : source.MakeSubgraph("Body");
    if (function)
      scope.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
    scope.MakeNode("Neg", {"X"}, {"Y"});
    scope.MakeOutput("Y");
    if (!function) {
      source.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const std::string name : {"then_branch", "else_branch"}) {
        auto &reference = attributes.add();
        reference.set_name(name + "_ref");
        reference.set_type(AttributeProto::STRING);
        reference.set_s("Body");
      }
      source.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
      source.MakeOutput("selected");
    }
    const auto original_tags = scope.Compute().ValueTags();
    auto matching = source.ApplyRewritings(RewritingProto{});
    const auto &matching_scope =
        function ? matching.LocalFunction("Body") : matching.Subgraph("Body");
    EXPECT_EQ(matching_scope.Compute().ValueTags(), original_tags);
    if (!function) {
      EXPECT_EQ(original_tags.count("Y"), 0u);
    }
    auto &compute = const_cast<core::compute::ComputeContext &>(scope.Compute());
    compute.SeedValueTag("Y", "shape", scope.Nodes());
    const auto before = source.BuildGraph().SerializeAsString();
    const auto tags = scope.Compute().ValueTags();
    try {
      source.ApplyRewritings(RewritingProto{});
      FAIL() << "Expected inconsistent scoped tags to be rejected.";
    } catch (const core::builder::BuilderError &error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("Body"), std::string::npos);
      EXPECT_NE(message.find("'Y'"), std::string::npos);
      EXPECT_NE(message.find("existing tag 'shape'"), std::string::npos);
      EXPECT_NE(
          message.find(function ? "reconstructed tag 'weight'" : "reconstructed tag <absent>"),
          std::string::npos);
    }
    EXPECT_EQ(source.BuildGraph().SerializeAsString(), before);
    EXPECT_EQ(scope.Compute().ValueTags(), tags);
  }
}

TEST(GraphGraph, ReplayValidatesSurvivingSubgraphsInChangedFunctions) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}, {"local", 1}});
  auto &function = source.MakeLocalFunction("Body", "local");
  function.SetOpsetVersion("", 28);
  function.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  function.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = function.MakeSubgraph(name);
    branch.MakeNode("Neg", {"X"}, {name + "_value"});
    branch.MakeOutput(name + "_value");
  }
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &reference = attributes.add();
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s(name);
  }
  function.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  function.MakeNode("Identity", {"selected"}, {"Y"});
  function.MakeOutput("Y");
  auto &then_branch = function.Subgraph("then_branch");
  const_cast<core::compute::ComputeContext &>(then_branch.Compute())
      .SeedValueTag("then_branch_value", "weight", then_branch.Nodes());
  ASSERT_EQ(then_branch.Compute().ValueTags().at("then_branch_value"), "weight");

  FunctionProto replacement = function.ToFunction("local");
  replacement.ref_node()[1].set_op_type("Neg");
  bool changed_tag = false;
  const auto set_tag = [&](ValueInfoProto &value) {
    if (value.name() != "then_branch_value")
      return;
    for (auto &metadata : value.ref_metadata_props()) {
      if (metadata.key() == core::compute::kValueTagMetadataKey) {
        metadata.set_value("shape");
        changed_tag = true;
        return;
      }
    }
    auto &metadata = value.ref_metadata_props().add();
    metadata.set_key(core::compute::kValueTagMetadataKey);
    metadata.set_value("shape");
    changed_tag = true;
  };
  const auto change_graph = [&](const auto &self, GraphProto &graph) -> void {
    for (auto &value : graph.ref_input())
      set_tag(value);
    for (auto &value : graph.ref_output())
      set_tag(value);
    for (auto &value : graph.ref_value_info())
      set_tag(value);
    for (auto &node : graph.ref_node()) {
      for (auto &attribute : node.ref_attribute()) {
        if (attribute.has_g())
          self(self, *attribute.mutable_g());
        for (auto &nested : attribute.ref_graphs())
          self(self, nested);
      }
    }
  };
  for (auto &node : replacement.ref_node())
    for (auto &attribute : node.ref_attribute()) {
      if (attribute.has_g())
        change_graph(change_graph, *attribute.mutable_g());
      for (auto &nested : attribute.ref_graphs())
        change_graph(change_graph, nested);
    }
  ASSERT_TRUE(changed_tag);

  RewritingProto plan;
  plan.add_replaced_function_ids("local:Body");
  plan.ref_added_functions().push_back(std::move(replacement));
  try {
    source.ApplyRewritings(plan);
    FAIL() << "Expected a changed nested value tag to be rejected.";
  } catch (const core::builder::BuilderError &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("then_branch"), std::string::npos);
    EXPECT_NE(message.find("then_branch_value"), std::string::npos);
    EXPECT_NE(message.find("existing tag 'weight'"), std::string::npos);
    EXPECT_NE(message.find("reconstructed tag 'shape'"), std::string::npos);
  }
}

TEST(GraphGraph, ReplayValidatesEveryOccurrenceOfSharedFunctionSubgraphs) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}, {"local", 1}});
  auto &function = source.MakeLocalFunction("Body", "local");
  function.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  function.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  auto &branch = function.MakeSubgraph("body");
  branch.MakeNode("Neg", {"X"}, {"branch_value"});
  ValueInfoProto branch_output;
  branch_output.set_name("branch_value");
  auto &tag = branch_output.ref_metadata_props().add();
  tag.set_key(core::compute::kValueTagMetadataKey);
  tag.set_value("weight");
  branch.MakeOutput(branch_output);
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &reference = attributes.add();
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s("body");
  }
  function.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  function.MakeNode("Identity", {"selected"}, {"Y"});
  function.MakeOutput("Y");
  ASSERT_EQ(branch.Compute().ValueTags().at("branch_value"), "weight");
  RewritingProto plan;
  plan.add_replaced_function_ids("local:Body");
  auto replacement = function.ToFunction("local");
  replacement.ref_node()[1].set_op_type("Neg");
  plan.ref_added_functions().push_back(std::move(replacement));
  EXPECT_NO_THROW(source.ApplyRewritings(plan));

  auto &second_branch = *plan.ref_added_functions()[0].ref_node()[0].ref_attribute()[1].mutable_g();
  ASSERT_EQ(second_branch.name(), "body");
  bool changed_tag = false;
  const auto change_tag = [&](auto &values) {
    for (auto &value : values) {
      if (value.name() != "branch_value")
        continue;
      for (auto &metadata : value.ref_metadata_props()) {
        if (metadata.key() == core::compute::kValueTagMetadataKey) {
          metadata.set_value("shape");
          changed_tag = true;
        }
      }
    }
  };
  change_tag(second_branch.ref_output());
  change_tag(second_branch.ref_value_info());
  ASSERT_TRUE(changed_tag);
  EXPECT_THROW(source.ApplyRewritings(plan), core::builder::BuilderError);
}

TEST(GraphGraph, ReplayErrorsIdentifyInvalidReplacementFunctionIds) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"local", 1}});
  auto &function = builder.MakeLocalFunction("Body", "local");
  function.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
  function.MakeNode("Identity", {"X"}, {"Y"});
  function.MakeOutput("Y");

  RewritingProto unknown;
  unknown.add_replaced_function_ids("local:Missing");
  try {
    builder.ApplyRewritings(unknown);
    FAIL() << "Expected an unknown replacement function identity to be rejected.";
  } catch (const core::builder::BuilderError &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("local:Missing"), std::string::npos);
    EXPECT_NE(message.find("not replaceable"), std::string::npos);
  }

  RewritingProto duplicate;
  duplicate.add_replaced_function_ids("local:Body");
  duplicate.add_replaced_function_ids("local:Body");
  duplicate.ref_added_functions().push_back(function.ToFunction("local"));
  try {
    builder.ApplyRewritings(duplicate);
    FAIL() << "Expected a duplicate replacement function identity to be rejected.";
  } catch (const core::builder::BuilderError &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("local:Body"), std::string::npos);
    EXPECT_NE(message.find("duplicate"), std::string::npos);
  }
}

TEST(GraphGraph, ReplayMatchesFunctionTagsByDomainAndOverload) {
  for (const bool overload : {false, true}) {
    SCOPED_TRACE(overload);
    ModelProto model;
    model.mutable_graph()->set_name("g");
    model.add_opset("", 28);
    for (const bool shape : {false, true}) {
      auto &function = *model.add_functions();
      function.set_name("Body");
      function.set_domain(shape && !overload ? "other" : "local");
      if (shape && overload)
        function.set_overload("shape");
      function.add_opset("", 28);
      function.add_input("X");
      function.add_output("Y");
      function.add_node(MakeNode(shape ? "Shape" : "Neg", {"X"}, {"Y"}));
    }
    core::builder::GraphBuilder source(model, SchemaLookup());
    ASSERT_EQ(source.LocalFunctions().size(), 2u);
    EXPECT_EQ(source.LocalFunctions()[0]->Compute().ValueTags().at("Y"), "weight");
    EXPECT_EQ(source.LocalFunctions()[1]->Compute().ValueTags().at("Y"), "shape");
    auto matching = source.ApplyRewritings(RewritingProto{});
    ASSERT_EQ(matching.LocalFunctions().size(), 2u);
    for (std::size_t index = 0; index < 2; ++index) {
      EXPECT_EQ(matching.LocalFunctions()[index]->Compute().ValueTags(),
                source.LocalFunctions()[index]->Compute().ValueTags());
    }
    auto &compute =
        const_cast<core::compute::ComputeContext &>(source.LocalFunctions()[1]->Compute());
    compute.SeedValueTag("Y", "axes", source.LocalFunctions()[1]->Nodes());
    EXPECT_THROW(source.ApplyRewritings(RewritingProto{}), core::builder::BuilderError);
  }
}

TEST(GraphGraph, ReplayValidatesHoistedFunctionTagsAcrossSourceScopes) {
  for (const bool function_owner : {false, true}) {
    SCOPED_TRACE(function_owner);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
    source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
    auto &owner =
        function_owner ? source.MakeLocalFunction("Owner", "local") : source.MakeSubgraph("Owner");
    if (function_owner)
      owner.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
    owner.MakeNode("Neg", {"X"}, {"result"});
    owner.MakeOutput("result");
    if (function_owner) {
      source.MakeOutput("X");
    } else {
      source.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const std::string name : {"then_branch", "else_branch"}) {
        auto &reference = attributes.add();
        reference.set_name(name + "_ref");
        reference.set_type(AttributeProto::STRING);
        reference.set_s("Owner");
      }
      source.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
      source.MakeOutput("selected");
    }
    auto &function = owner.MakeLocalFunction("Body", "local");
    function.MakeInput("U", core::symbolic::TensorType::kFloat, MakeShape({2}));
    function.MakeNode("Neg", {"U"}, {"V"});
    function.MakeOutput("V");
    RewritingProto plan;
    plan.ref_added_functions().push_back(function.ToFunction("local"));
    auto matching = source.ApplyRewritings(plan);
    EXPECT_EQ(matching.LocalFunction("Body").Compute().ValueTags(), function.Compute().ValueTags());
    auto &compute = const_cast<core::compute::ComputeContext &>(function.Compute());
    compute.SeedValueTag("V", "axes", function.Nodes());
    const auto before = source.BuildGraph().SerializeAsString();
    const auto tags = function.Compute().ValueTags();
    try {
      source.ApplyRewritings(plan);
      FAIL() << "Expected inconsistent hoisted function tags to be rejected.";
    } catch (const core::builder::BuilderError &error) {
      const std::string message = error.what();
      const std::string path = function_owner ? "g/function:local::Owner:/function:local::Body:"
                                              : "g/subgraph:Owner/function:local::Body:";
      EXPECT_NE(message.find(path), std::string::npos);
      EXPECT_NE(message.find("'V'"), std::string::npos);
      EXPECT_NE(message.find("existing tag 'axes'"), std::string::npos);
      EXPECT_NE(message.find("reconstructed tag 'weight'"), std::string::npos);
    }
    EXPECT_EQ(source.BuildGraph().SerializeAsString(), before);
    EXPECT_EQ(function.Compute().ValueTags(), tags);
  }
}

TEST(GraphGraph, ReplayValidatesEverySourceOfAHoistedFunction) {
  for (const std::string changed_scope : {"first", "second"}) {
    SCOPED_TRACE(changed_scope);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
    for (const std::string name : {"first", "second"}) {
      auto &function = source.MakeSubgraph(name).MakeLocalFunction("Body", "local");
      function.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2}));
      function.MakeNode("Neg", {"X"}, {"Y"});
      function.MakeOutput("Y");
    }
    RewritingProto plan;
    plan.ref_added_functions().push_back(
        source.Subgraph("first").LocalFunction("Body").ToFunction("local"));
    EXPECT_NO_THROW(source.ApplyRewritings(plan));
    auto &function = source.Subgraph(changed_scope).LocalFunction("Body");
    auto &compute = const_cast<core::compute::ComputeContext &>(function.Compute());
    compute.SeedValueTag("Y", "axes", function.Nodes());
    try {
      source.ApplyRewritings(plan);
      FAIL() << "Expected every original function scope to be validated.";
    } catch (const core::builder::BuilderError &error) {
      EXPECT_NE(
          std::string(error.what()).find("g/subgraph:" + changed_scope + "/function:local::Body:"),
          std::string::npos)
          << error.what();
    }
  }
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

TEST(GraphGraph, ReplayRejectsCyclesDuringDependencyInsertion) {
  for (const int cycle_size : {1, 2, 3}) {
    for (const bool external_prerequisite : {false, true}) {
      SCOPED_TRACE(cycle_size);
      SCOPED_TRACE(external_prerequisite);
      core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
      source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
      source.MakeNode("Abs", {"X"}, {"a"});
      source.MakeOutput("a");
      RewritingProto plan;
      auto &rewrite = *plan.add_rewrites();
      for (int index = 0; index < cycle_size; ++index) {
        const std::string input = "v" + std::to_string((index + 1) % cycle_size);
        const std::string output = "v" + std::to_string(index);
        rewrite.add_added_nodes_positions(index);
        rewrite.add_added_nodes(external_prerequisite && index == 0
                                    ? MakeNode("Add", {input, "a"}, {output})
                                    : MakeNode("Neg", {input}, {output}));
      }
      try {
        source.ApplyRewritings(plan);
        ADD_FAILURE() << "Expected a cyclic dependency error.";
      } catch (const core::builder::BuilderError &error) {
        EXPECT_NE(std::string(error.what()).find("cyclic node dependencies"), std::string::npos);
      }
    }
  }
}

TEST(GraphGraph, ReplayAcceptsSharedPrerequisitesDuringDependencyInsertion) {
  for (const bool diamond : {false, true}) {
    SCOPED_TRACE(diamond);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 28}});
    source.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    source.MakeNode("Abs", {"X"}, {"a"});
    source.MakeOutput("a");
    core::builder::GraphBuilder expected("g", SchemaLookup(), {{"", 28}});
    expected.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    if (diamond) {
      expected.MakeNode("Abs", {"X"}, {"a"});
      expected.MakeNode("Add", {"X", "a"}, {"c"});
      expected.MakeNode("Neg", {"c"}, {"b"});
      expected.MakeNode("Relu", {"c"}, {"d"});
      expected.MakeNode("Add", {"b", "d"}, {"e"});
    } else {
      expected.MakeNode("Neg", {"X"}, {"b"});
      expected.MakeNode("Abs", {"X"}, {"a"});
      expected.MakeNode("Add", {"b", "a"}, {"c"});
    }
    const std::string output = diamond ? "e" : "c";
    expected.MakeOutput("a");
    expected.MakeOutput(output);
    source.MakeNode("Identity", {"a"}, {output});
    source.MakeOutput(output);
    RewritingProto plan;
    auto &rewrite = *plan.add_rewrites();
    rewrite.add_matched_nodes(1);
    rewrite.add_added_nodes_positions(0);
    rewrite.add_added_nodes(MakeNode("Add", {diamond ? "X" : "b", "a"}, {"c"}));
    rewrite.add_added_nodes_positions(1);
    rewrite.add_added_nodes(MakeNode("Neg", {diamond ? "c" : "X"}, {"b"}));
    if (diamond) {
      rewrite.add_added_nodes_positions(2);
      rewrite.add_added_nodes(MakeNode("Relu", {"c"}, {"d"}));
      rewrite.add_added_nodes_positions(3);
      rewrite.add_added_nodes(MakeNode("Add", {"b", "d"}, {"e"}));
    }
    auto result = source.ApplyRewritings(plan);
    ASSERT_EQ(result.Nodes().size(), expected.Nodes().size());
    for (std::size_t index = 0; index < expected.Nodes().size(); ++index) {
      const std::string output = expected.Nodes()[index].output(0);
      EXPECT_EQ(result.Nodes()[index].output(0), output);
      EXPECT_EQ(result.GetShape(output), expected.GetShape(output));
      EXPECT_EQ(result.Compute().NodeReuse(index), expected.Compute().NodeReuse(index));
    }
    EXPECT_EQ(result.Compute().Memory(), expected.Compute().Memory());
    EXPECT_EQ(result.Compute().PeakMemory(), expected.Compute().PeakMemory());
    auto actual_graph = result.BuildGraph();
    auto expected_graph = expected.BuildGraph();
    result.Compute().WriteToGraph(actual_graph);
    expected.Compute().WriteToGraph(expected_graph);
    EXPECT_EQ(actual_graph.SerializeAsString(), expected_graph.SerializeAsString());
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
