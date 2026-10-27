// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/patterns/attention/attention_pattern.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"
#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"
#include "onnx_extensions/patterns/canonicalization/constant_pattern.h"
#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"
#include "onnx_extensions/patterns/canonicalization/dropout_pattern.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
#include "onnx_extensions/patterns/canonicalization/initializer_unsqueeze_cast_pattern.h"
#include "onnx_extensions/patterns/canonicalization/not_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/patterns/transpose/transpose_pattern.h"

#include "onnx_helper.h"
#include "onnx_lib/common/scoped_resource.h"
#include "onnx_op/operator_sets.h"
#include "onnx_threads/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <initializer_list>
#include <limits>
#include <memory>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  return shape;
}

core::symbolic::SymShape Shape(std::initializer_list<int64_t> dimensions) {
  core::symbolic::SymShape shape;
  for (int64_t dimension : dimensions) {
    shape.PushBack(core::symbolic::SymDim(dimension));
  }
  return shape;
}

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType to, const std::string &name = "") {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("to");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(static_cast<int64_t>(to));
  builder.MakeNode("Cast", {input}, {output}, "", name, attributes);
}

void AddSubgraphReference(core::builder::GraphBuilder &builder, const std::string &attribute_name,
                          const std::string &subgraph_name, const std::string &output) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name(attribute_name + "_ref");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s(subgraph_name);
  builder.MakeNode("SubgraphCarrier", {}, {output}, "", "", attributes);
}

void AddOpsetImports(ModelProto &model, const core::builder::GraphBuilder &builder) {
  for (const auto &[domain, version] : builder.OpsetVersions()) {
    model.add_opset(domain, version);
  }
}

class ReplaceIdentityWithConstantAddPattern final : public core::builder::PatternOptimization {
public:
  ReplaceIdentityWithConstantAddPattern()
      : PatternOptimization(1, "ReplaceIdentityWithConstantAdd") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (candidate.op_type().value() != "Identity" || candidate.output_size() != 1 ||
        !graph.IsConstant("a") || !graph.IsConstant("b")) {
      return {};
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(MakeNode("Add", {"a", "b"}, {nodes[0]->output()[0].value()}));
    return replacements;
  }
};

class CountingFastOpTypePattern final : public core::builder::PatternOptimization {
public:
  CountingFastOpTypePattern() : PatternOptimization(1, "CountingFastOpType") {}

  std::set<std::string> FastOpType() const override {
    ++fast_op_type_calls;
    return {"Add"};
  }

  bool CanMatchInParallel() const noexcept override { return true; }

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(*nodes[0]);
    return replacements;
  }

  mutable std::size_t fast_op_type_calls = 0;
};

class ThreadTrackingPattern : public core::builder::PatternOptimization {
public:
  explicit ThreadTrackingPattern(std::string name = "SequentialTracking")
      : PatternOptimization(1, std::move(name)), calling_thread_(std::this_thread::get_id()) {}

  std::set<std::string> FastOpType() const override { return {"Add"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &, const NodeProto &) const override {
    if (std::this_thread::get_id() != calling_thread_) {
      worker_calls.fetch_add(1, std::memory_order_relaxed);
    }
    return {};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &) const override {
    throw core::builder::BuilderError("ThreadTrackingPattern never matches.");
  }

  mutable std::atomic<std::size_t> worker_calls{0};

private:
  std::thread::id calling_thread_;
};

class ParallelThreadTrackingPattern final : public ThreadTrackingPattern {
public:
  ParallelThreadTrackingPattern() : ThreadTrackingPattern("ParallelTracking") {}

  bool CanMatchInParallel() const noexcept override { return true; }
};

class ReorderedFoldingPattern final : public core::builder::PatternOptimization {
public:
  explicit ReorderedFoldingPattern(std::string op_type, bool shared_anchor = false)
      : PatternOptimization(1, "Fold" + op_type), op_type_(std::move(op_type)),
        shared_anchor_(shared_anchor) {}

  std::set<std::string> FastOpType() const override { return {op_type_}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    return {this, {&candidate}, shared_anchor_ ? graph.NodeBefore("anchor") : &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override {
    const std::string output = nodes[0]->output(0).value();
    TensorProto literal;
    literal.set_name(shared_anchor_ ? "second_literal" : output + "_literal");
    literal.set_data_type(TensorProto::FLOAT);
    literal.add_dims(1);
    literal.add_float_data(op_type_ == "Relu" ? 2.0f : 5.0f);
    if (!shared_anchor_ || op_type_ == "Relu")
      graph.Builder().MakeInitializer(literal);
    utils::RepeatedProtoField<NodeProto> result;
    result.push_back(
        MakeNode("Add",
                 {shared_anchor_ && op_type_ == "Identity" ? literal.name().value() : "base",
                  literal.name().value()},
                 {output + "_weight"}));
    result.push_back(MakeNode("Add", {"x", output + "_weight"}, {output}));
    return result;
  }

private:
  std::string op_type_;
  bool shared_anchor_;
};

class NullPlaceholderPattern final : public core::builder::PatternOptimization {
public:
  NullPlaceholderPattern() : PatternOptimization(1, "NullPlaceholder") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return core::builder::MatchResult{this, {nullptr, &candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    if (nodes.size() != 2 || nodes[0] != nullptr || nodes[1] == nullptr) {
      throw core::builder::BuilderError("NullPlaceholderPattern received invalid role slots.");
    }
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(
        MakeNode("Relu", {nodes[1]->input()[0].value()}, {nodes[1]->output()[0].value()}));
    return replacements;
  }
};

class ReplaceIdentityWithTwoNodesPattern final : public core::builder::PatternOptimization {
public:
  ReplaceIdentityWithTwoNodesPattern() : PatternOptimization(1, "ReplaceIdentityWithTwoNodes") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    if (candidate.op_type().value() != "Identity" || candidate.input_size() != 1 ||
        candidate.output_size() != 1) {
      return {};
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    const NodeProto &identity = *nodes[0];
    const std::string intermediate = identity.output()[0].value() + "_neg";
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(
        MakeNode("Neg", {identity.input()[0].value()}, {intermediate}, "", "replacement_neg"));
    replacements.push_back(
        MakeNode("Relu", {intermediate}, {identity.output()[0].value()}, "", "replacement_relu"));
    return replacements;
  }
};

class ScopedAddKernel {
public:
  ScopedAddKernel() {
    const auto existing = core::runtime::GlobalCustomKernels().find(kDefaultOnnxDomain + ":Add");
    if (existing != core::runtime::GlobalCustomKernels().end()) {
      previous_ = existing->second;
    }
    core::runtime::RegisterGlobalCustomKernel(
        "", "Add", [](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
          const core::runtime::Tensor &left = runtime.tensors().at(node.input()[0].value());
          const core::runtime::Tensor &right = runtime.tensors().at(node.input()[1].value());
          runtime.tensors()[node.output()[0].value()] = core::runtime::Tensor::FromFloat(
              node.output()[0].value(), left.shape, {left.AsFloat()[0] + right.AsFloat()[0]});
        });
  }

  ~ScopedAddKernel() {
    if (previous_) {
      core::runtime::RegisterGlobalCustomKernelFactory("", "Add", std::move(previous_));
    } else {
      core::runtime::UnregisterGlobalCustomKernel("", "Add");
    }
  }

private:
  core::runtime::NodeKernelFn previous_;
};

class ScopedIdentityKernel {
public:
  ScopedIdentityKernel() {
    const auto &kernels = core::runtime::GlobalCustomKernels();
    const auto existing = kernels.find(kDefaultOnnxDomain + ":Identity");
    if (existing != kernels.end())
      previous_ = existing->second;
    core::runtime::RegisterGlobalCustomKernel(
        "", "Identity", [](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
          runtime.Set(node.output(0).value(), runtime.Get(node.input(0).value()));
        });
  }

  ~ScopedIdentityKernel() {
    if (previous_)
      core::runtime::RegisterGlobalCustomKernelFactory("", "Identity", std::move(previous_));
    else
      core::runtime::UnregisterGlobalCustomKernel("", "Identity");
  }

private:
  core::runtime::NodeKernelFn previous_;
};

class ScopedConstantKernel {
public:
  ScopedConstantKernel() {
    const auto &kernels = core::runtime::GlobalCustomKernels();
    const auto existing = kernels.find(kDefaultOnnxDomain + ":Constant");
    if (existing != kernels.end())
      previous_ = existing->second;
    core::runtime::RegisterGlobalCustomKernel(
        "", "Constant", [](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
          ASSERT_EQ(node.attribute_size(), 1);
          const auto &value = node.attribute(0);
          ASSERT_EQ(value.name(), "value");
          ASSERT_TRUE(value.has_t());
          runtime.Set(node.output(0).value(), core::runtime::TensorFromProto(value.t()));
        });
  }

  ~ScopedConstantKernel() {
    if (previous_)
      core::runtime::RegisterGlobalCustomKernelFactory("", "Constant", std::move(previous_));
    else
      core::runtime::UnregisterGlobalCustomKernel("", "Constant");
  }

private:
  core::runtime::NodeKernelFn previous_;
};

class ScopedAbsShapeCounter {
public:
  ScopedAbsShapeCounter() : previous_(core::shapes::DispatchTable().at("ai.onnx:Abs")) {
    core::shapes::RegisterComputeShapeFn("", "Abs", [this](auto &ctx, const auto &node) {
      ++calls;
      previous_(ctx, node);
    });
  }

  ~ScopedAbsShapeCounter() {
    core::shapes::RegisterComputeShapeFn("", "Abs", std::move(previous_));
  }

  std::size_t calls = 0;

private:
  core::shapes::ComputeShapeFn previous_;
};

class RegisterNestedFunctionPattern final : public core::builder::PatternOptimization {
public:
  RegisterNestedFunctionPattern() : PatternOptimization(1, "RegisterNestedFunction") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return candidate.op_type() == "Identity"
               ? core::builder::MatchResult{this, {&candidate}, &candidate}
               : core::builder::MatchResult{};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override {
    auto &builder = graph.Builder();
    builder.SetOpsetVersion("nested.test", nodes[0]->name() == "version_two" ? 2 : 1);
    auto &function = builder.MakeLocalFunction("Nested", "nested.test");
    function.SetOpsetVersion("", 17);
    function.MakeInput("arg", core::symbolic::TensorType::kFloat, Shape());
    function.MakeNode("Identity", {"arg"}, {"result"});
    function.MakeOutput("result");
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(MakeNode("Nested", {nodes[0]->input(0).value()},
                                    {nodes[0]->output(0).value()}, "nested.test"));
    return replacements;
  }
};

class RegisterLateAddOverridePattern final : public core::builder::PatternOptimization {
public:
  RegisterLateAddOverridePattern() : PatternOptimization(1, "RegisterLateAddOverride") {}

  std::set<std::string> FastOpType() const override { return {"Relu"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (graph.NodeBefore("y") != nullptr || !graph.IsConstant("y"))
      return {};
    return {this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override {
    auto &function = graph.Builder().MakeLocalFunction("Add");
    function.MakeInput("A", core::symbolic::TensorType::kFloat, Shape());
    function.MakeInput("B", core::symbolic::TensorType::kFloat, Shape());
    function.MakeNode("Mul", {"A", "B"}, {"C"});
    function.MakeOutput("C");
    utils::RepeatedProtoField<NodeProto> result;
    result.push_back(
        MakeNode("Identity", {nodes[0]->input(0).value()}, {nodes[0]->output(0).value()}));
    return result;
  }
};

FunctionProto PreparationAdd() {
  FunctionProto function;
  function.set_name("Prepare");
  function.set_domain("test.prepare");
  function.add_input("A");
  function.add_input("B");
  function.add_output("C");
  function.add_output("D");
  function.add_opset("", 18);
  function.ref_node().push_back(MakeNode("Add", {"A", "B"}, {"C"}));
  function.ref_node().push_back(MakeNode("Add", {"C", "B"}, {"D"}));
  return function;
}

RewritingProto PreparationPlan() {
  RewritingProto plan;
  plan.ref_added_functions().push_back(PreparationAdd());
  auto &opset = *plan.add_required_opset_imports();
  opset.set_domain("test.prepare");
  opset.set_version(1);
  auto &edit = *plan.add_rewrites();
  edit.add_matched_nodes(0);
  edit.add_removed_initializers(0);
  edit.add_removed_initializers(1);
  edit.ref_added_nodes().push_back(MakeNode("Add", {"e", "f"}, {"y"}));
  edit.add_added_nodes_positions(0);
  edit.ref_added_initializers().push_back(MakeInitializer<float>("literal", {1}, {2}));
  edit.add_added_initializer_positions(2);
  edit.ref_initializer_calls().push_back(
      MakeNode("Prepare", {"a", "b"}, {"c", "d"}, "test.prepare"));
  edit.ref_initializer_calls().push_back(
      MakeNode("Prepare", {"c", "literal"}, {"e", "f"}, "test.prepare"));
  for (uint64_t position : {1, 0, 3, 4})
    edit.add_computed_initializer_positions(position);
  return plan;
}

ModelProto PreparationModel() {
  ModelProto model;
  model.set_producer_name("canonical");
  model.add_opset("", 18);
  auto &graph = *model.mutable_graph();
  graph.set_name("preparation");
  graph.ref_initializer().push_back(MakeInitializer<float>("a", {1}, {1}));
  graph.ref_initializer().push_back(MakeInitializer<float>("b", {1}, {2}));
  graph.ref_node().push_back(MakeNode("Add", {"a", "b"}, {"y"}));
  auto &output = *graph.add_output();
  output.set_name("y");
  auto &type = *output.mutable_type()->mutable_tensor_type();
  type.set_elem_type(TensorProto::FLOAT);
  type.mutable_shape()->add_dim()->set_dim_value(1);
  return model;
}

} // namespace

TEST(PatternOptimization, RegistersNamedBuiltInPatternSets) {
  onnx_patterns::RegisterPatterns();
  const auto sets = core::builder::RegisteredPatternSets();

  ASSERT_TRUE(sets.contains("all"));
  ASSERT_TRUE(sets.contains("L1"));
  ASSERT_TRUE(sets.contains("L2"));
  ASSERT_TRUE(sets.contains("Layer"));
  EXPECT_NE(std::find(sets.at("L1").begin(), sets.at("L1").end(), "MatMulAdd"),
            sets.at("L1").end());
  EXPECT_NE(std::find(sets.at("L2").begin(), sets.at("L2").end(), "GatherToSlice"),
            sets.at("L2").end());
  EXPECT_EQ(sets.at("Layer"), (std::vector<std::string>{"RepeatedLayer"}));
  for (const std::string &name : sets.at("L1")) {
    EXPECT_NE(std::find(sets.at("all").begin(), sets.at("all").end(), name), sets.at("all").end());
  }
  for (const std::string &name : sets.at("L2")) {
    EXPECT_NE(std::find(sets.at("all").begin(), sets.at("all").end(), name), sets.at("all").end());
  }
  for (const std::string &name : sets.at("Layer")) {
    EXPECT_NE(std::find(sets.at("all").begin(), sets.at("all").end(), name), sets.at("all").end());
  }
}

TEST(PatternOptimization, NamedVariantsRoundTripSelectIndependentlyAndKeepCanonical) {
  ScopedAddKernel kernel;
  core::builder::GraphBuilder source(PreparationModel(), SchemaLookup());
  const auto canonical = source.ToModel();
  source.AddRewriting("prepared", PreparationPlan());
  source.AddRewriting("unchanged", RewritingProto{});
  EXPECT_THROW(source.AddRewriting("prepared", RewritingProto{}), core::builder::BuilderError);
  EXPECT_THROW(source.AddRewriting("", RewritingProto{}), core::builder::BuilderError);
  const auto bundled = source.ToModel();
  EXPECT_TRUE(bundled.graph().Equals(canonical.graph()));
  ModelProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(bundled.SerializeAsString()));
  ASSERT_EQ(loaded.rewritings_size(), 2);
  core::builder::GraphBuilder restored(loaded, SchemaLookup());
  EXPECT_EQ(restored.Rewritings().size(), 2u);
  const auto prepared = restored.ApplyRewritings("prepared").ToModel();
  const auto unchanged = restored.ApplyRewritings("unchanged").ToModel();
  EXPECT_TRUE(prepared.rewritings().empty());
  EXPECT_TRUE(unchanged.rewritings().empty());
  EXPECT_FLOAT_EQ(core::runtime::RunModel(prepared, {})[0].AsFloat()[0], 12);
  EXPECT_FLOAT_EQ(core::runtime::RunModel(unchanged, {})[0].AsFloat()[0], 3);
  EXPECT_TRUE(restored.ToModel().graph().Equals(canonical.graph()));
  EXPECT_EQ(restored.Rewritings().size(), 2u);
  EXPECT_THROW(restored.ApplyRewritings("absent"), core::builder::BuilderError);
  EXPECT_THROW(restored.ToStandardModel(), core::builder::BuilderError);
  EXPECT_TRUE(restored.ApplyRewritings(RewritingProto{}).Rewritings().empty());
}

TEST(PatternOptimization, NamedVariantsInvalidateOnCanonicalMutations) {
  using Mutation = std::function<void(core::builder::GraphBuilder &)>;
  const std::vector<Mutation> mutations{
      [](auto &builder) { builder.MakeNode("Identity", {"y"}, {"extra"}); },
      [](auto &builder) { builder.MakeInitializer(MakeInitializer<float>("extra", {1}, {1})); },
      [](auto &builder) { builder.MakeOutput("a"); },
      [](auto &builder) { builder.SetOpsetVersion("", 19); },
      [](auto &builder) { builder.MakeLocalFunction("Extra", "test.variant"); }};
  for (std::size_t i = 0; i < mutations.size(); ++i) {
    SCOPED_TRACE(i);
    core::builder::GraphBuilder builder(PreparationModel(), SchemaLookup());
    builder.AddRewriting("variant", RewritingProto{});
    mutations[i](builder);
    EXPECT_TRUE(builder.Rewritings().empty());
    EXPECT_FALSE(builder.ToModel().has_rewritings());
    EXPECT_THROW(builder.ApplyRewritings("variant"), core::builder::BuilderError);
  }
  core::builder::GraphBuilder parent(PreparationModel(), SchemaLookup());
  auto &function = parent.MakeLocalFunction("Nested", "test.variant");
  function.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  function.MakeNode("Identity", {"x"}, {"y"});
  function.MakeOutput("y");
  parent.AddRewriting("variant", RewritingProto{});
  function.MakeNode("Neg", {"y"}, {"extra"});
  EXPECT_TRUE(parent.Rewritings().empty());
  auto &body = parent.MakeSubgraph("body");
  body.MakeNode("Identity", {"a"}, {"out"});
  body.MakeOutput("out");
  AddSubgraphReference(parent, "body", "body", "subgraph_out");
  parent.AddRewriting("nested", RewritingProto{});
  body.MakeNode("Neg", {"out"}, {"unused"});
  EXPECT_TRUE(parent.Rewritings().empty());
}

TEST(PatternOptimization, NamedVariantsInvalidateOnFinalizationReordering) {
  core::builder::GraphBuilder builder("canonical", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Neg", {"x"}, {"a"});
  builder.MakeNode("Neg", {"a"}, {"b"});
  builder.MakeNode("Shape", {"a"}, {"s"});
  builder.MakeOutput("b");
  builder.MakeOutput("s");
  builder.AddRewriting("before_finalization", RewritingProto{});
  const auto finalized = builder.ToModel();
  ASSERT_EQ(finalized.graph().node_size(), 3);
  EXPECT_EQ(finalized.graph().node(1).op_type(), "Shape");
  EXPECT_FALSE(finalized.has_rewritings());
  EXPECT_TRUE(builder.Rewritings().empty());
  builder.AddRewriting("finalized", RewritingProto{});
  EXPECT_EQ(builder.ToModel().rewritings_size(), 1);
}

TEST(PatternOptimization, NamedVariantsDiagnoseMissingImplementationsAndInvalidPlans) {
  core::builder::GraphBuilder builder(PreparationModel(), SchemaLookup());
  RewritingProto invalid;
  invalid.add_rewrites()->add_matched_nodes(UINT64_MAX);
  builder.AddRewriting("invalid", invalid);
  EXPECT_THROW(builder.ApplyRewritings("invalid"), core::builder::BuilderError);
  RewritingProto unavailable;
  auto &edit = *unavailable.add_rewrites();
  edit.add_matched_nodes(0);
  edit.ref_added_nodes().push_back(MakeNode("MissingVariantKernel", {"a", "b"}, {"y"}));
  edit.add_added_nodes_positions(0);
  builder.AddRewriting("missing", unavailable);
  EXPECT_THROW(
      {
        try {
          builder.ApplyRewritings("missing");
        } catch (const core::builder::BuilderError &error) {
          EXPECT_NE(std::string(error.what()).find("no implementation"), std::string::npos);
          EXPECT_NE(std::string(error.what()).find("MissingVariantKernel"), std::string::npos);
          throw;
        }
      },
      core::builder::BuilderError);
  EXPECT_EQ(builder.Rewritings().size(), 2u);
}

TEST(PatternOptimization, NamedVariantsCheckControlFlowAndBoundFunctionGraphs) {
  ScopedIdentityKernel kernel;
  for (int mode : {0, 1, 2}) {
    for (bool missing : {false, true}) {
      SCOPED_TRACE(mode);
      SCOPED_TRACE(missing);
      auto model = PreparationModel();
      auto &condition = *model.mutable_graph()->add_input();
      condition.set_name("condition");
      condition.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::BOOL);
      condition.mutable_type()->mutable_tensor_type()->mutable_shape();
      FunctionProto wrapper;
      wrapper.set_domain("variant.test");
      wrapper.set_name("Wrapper");
      wrapper.add_input("a");
      wrapper.add_input("condition");
      wrapper.add_output("C");
      wrapper.add_opset("", 18);
      NodeProto conditional = MakeNode("If", {"condition"}, {mode == 0 ? "y" : "C"});
      NodeProto call = MakeNode("Wrapper", {"a", "condition"}, {"y"}, "variant.test");
      for (const char *name : {"then_branch", "else_branch"}) {
        AttributeProto branch;
        branch.set_name(name);
        branch.set_type(AttributeProto::GRAPH);
        branch.mutable_g()->set_name(name);
        branch.mutable_g()->ref_node().push_back(
            MakeNode(missing ? "Neg" : "Identity", {"a"}, {"out"}));
        auto output = model.graph().output(0);
        output.set_name("out");
        branch.mutable_g()->add_output(output);
        if (mode == 0) {
          conditional.add_attribute(branch);
        } else {
          wrapper.add_attribute_proto(branch);
          auto &reference = *conditional.add_attribute();
          reference.set_name(name);
          reference.set_type(AttributeProto::GRAPH);
          reference.set_ref_attr_name(name);
          if (mode == 2) {
            branch.mutable_g()->ref_node()[0].set_op_type("Identity");
            call.add_attribute(branch);
          }
        }
      }
      if (mode != 0) {
        wrapper.ref_node().push_back(std::move(conditional));
        model.add_function(wrapper);
        model.add_opset("variant.test", 1);
      }
      model.mutable_graph()->ref_node()[0] = mode == 0 ? conditional : call;
      core::builder::GraphBuilder builder(model, SchemaLookup());
      builder.set_device(core::symbolic::Device::kGPU0);
      builder.AddRewriting("variant", RewritingProto{});
      if (missing && mode != 2)
        EXPECT_THROW(builder.ApplyRewritings("variant"), core::builder::BuilderError);
      else
        EXPECT_NO_THROW(builder.ApplyRewritings("variant"));
    }
  }
}

TEST(PatternOptimization, NamedRewritingsRequireSequenceMapPackingCallback) {
  auto previous = core::runtime::GetSequenceMapPackFn();
  int pack_calls = 0;
  ScopeExit restore(
      [&]() noexcept { core::runtime::RegisterSequenceMapPackFn(std::move(previous)); });
  core::runtime::RegisterSequenceMapPackFn({});

  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 18);
  auto &graph = *model.mutable_graph();
  graph.set_name("sequence_map");
  auto &input = *graph.add_input();
  input.set_name("xs");
  auto &element =
      *input.mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  element.set_elem_type(TensorProto::FLOAT);
  element.mutable_shape()->add_dim()->set_dim_value(1);
  auto output = input;
  output.set_name("ys");
  graph.add_output(output);
  auto node = MakeNode("SequenceMap", {"xs"}, {"ys"});
  auto &attribute = *node.add_attribute();
  attribute.set_name("body");
  attribute.set_type(AttributeProto::GRAPH);
  auto &body = *attribute.mutable_g();
  body.set_name("passthrough");
  auto &item = *body.add_input();
  item.set_name("item");
  item.mutable_type()->mutable_tensor_type()->CopyFrom(element);
  body.add_output(item);
  graph.ref_node().push_back(std::move(node));
  core::builder::GraphBuilder builder(model, SchemaLookup());
  builder.AddRewriting("mapped", RewritingProto{});
  EXPECT_THROW(builder.ApplyRewritings("mapped"), core::builder::BuilderError);

  core::runtime::RegisterSequenceMapPackFn([&](core::runtime::RuntimeContext &,
                                               const core::runtime::Sequence &sequence,
                                               const std::vector<core::runtime::Tensors> &outputs) {
    ++pack_calls;
    return core::runtime::Sequences{
        core::runtime::Sequence("packed", sequence.elem_type, outputs.at(0))};
  });
  const auto selected = builder.ApplyRewritings("mapped").BuildGraph();
  EXPECT_EQ(pack_calls, 0);
  core::runtime::RuntimeContext runtime(
      core::runtime::KernelContext(core::runtime::DefaultOpset(18)));
  runtime.PutSequence(
      "xs", core::runtime::Sequence("xs", TensorProto::FLOAT,
                                    {core::runtime::Tensor::FromFloat("first", {1}, {2}),
                                     core::runtime::Tensor::FromFloat("second", {1}, {3})}));
  core::runtime::RuntimeSession session(runtime.GetExecutionPlan(selected));
  session.Run(runtime);
  EXPECT_EQ(pack_calls, 1);
  ASSERT_EQ(runtime.GetSequence("ys").size(), 2u);
  EXPECT_FLOAT_EQ(runtime.GetSequence("ys").at(0).AsFloat()[0], 2);
  EXPECT_FLOAT_EQ(runtime.GetSequence("ys").at(1).AsFloat()[0], 3);
}

TEST(PatternOptimization, ModelReplayPreparesDependentOutputsBeforeRemovingSources) {
  ScopedAddKernel kernel;
  const ModelProto original = PreparationModel();
  const std::string original_bytes = original.SerializeAsString();
  RewritingProto plan = PreparationPlan();
  // Identical definitions are installed only once.
  plan.ref_added_functions().push_back(PreparationAdd());
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
  const auto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded).ToModel();
  EXPECT_EQ(replayed.SerializeAsString(), core::builder::GraphBuilder(original, SchemaLookup())
                                              .ApplyRewritings(plan)
                                              .ToModel()
                                              .SerializeAsString());
  EXPECT_EQ(original.SerializeAsString(), original_bytes);
  EXPECT_EQ(replayed.producer_name(), "canonical");
  ASSERT_EQ(replayed.functions().size(), 1u);
  ASSERT_EQ(replayed.opset_import().size(), 2u);
  const auto &graph = replayed.graph();
  ASSERT_EQ(graph.initializer().size(), 5u);
  const std::vector<std::string> names{"d", "c", "literal", "e", "f"};
  const std::vector<float> values{5, 3, 2, 5, 7};
  for (std::size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(graph.initializer(i).name(), names[i]);
    EXPECT_FLOAT_EQ(core::runtime::TensorFromProto(graph.initializer(i)).AsFloat()[0], values[i]);
  }
  ASSERT_EQ(graph.node().size(), 1u);
  EXPECT_EQ(graph.node(0).op_type(), "Add");
  const auto outputs = core::runtime::RunModel(replayed, {});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_FLOAT_EQ(outputs[0].AsFloat()[0], 12);
}

TEST(PatternOptimization, ModelReplayRejectsInvalidPreparationAndConflictingDefinitions) {
  ScopedAddKernel kernel;
  using Mutation = std::function<void(ModelProto &, RewritingProto &)>;
  const std::vector<Mutation> invalid{
      [](auto &, auto &plan) {
        plan.ref_rewrites()[0].ref_computed_initializer_positions()[0] = 2;
      },
      [](auto &, auto &plan) { plan.ref_rewrites()[0].add_computed_initializer_positions(5); },
      [](auto &, auto &plan) {
        plan.ref_rewrites()[0].ref_initializer_calls()[0].ref_input()[0] = "future";
      },
      [](auto &, auto &plan) {
        plan.ref_rewrites()[0].ref_initializer_calls()[1].ref_output()[0] = "c";
      },
      [](auto &model, auto &) { model.mutable_graph()->add_input()->set_name("a"); },
      [](auto &, auto &plan) {
        plan.ref_added_functions()[0].ref_node()[0].set_op_type("RandomUniform");
      },
      [](auto &, auto &plan) {
        plan.ref_added_functions()[0].ref_node()[0].set_op_type("MissingPreparationKernel");
      },
      [](auto &, auto &plan) {
        auto &body = plan.ref_added_functions()[0].ref_node()[0];
        body.set_domain("test.prepare");
        body.set_op_type("Prepare");
        body.add_output("extra");
      },
      [](auto &, auto &plan) {
        plan.ref_added_functions()[0].ref_node()[0].ref_input()[0] = "capture";
      },
      [](auto &model, auto &) {
        auto function = PreparationAdd();
        function.ref_node()[0].set_op_type("Mul");
        model.ref_functions().push_back(std::move(function));
      },
      [](auto &, auto &plan) {
        auto function = PreparationAdd();
        function.ref_node()[0].set_op_type("Mul");
        plan.ref_added_functions().push_back(std::move(function));
      },
      [](auto &, auto &plan) {
        auto &opset = *plan.add_required_opset_imports();
        opset.set_domain("ai.onnx");
        opset.set_version(19);
      }};
  for (std::size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    ModelProto model = PreparationModel();
    RewritingProto plan = PreparationPlan();
    invalid[i](model, plan);
    EXPECT_THROW(core::builder::GraphBuilder(model, SchemaLookup()).ApplyRewritings(plan).ToModel(),
                 core::builder::BuilderError);
  }
}

TEST(PatternOptimization, AcceptsIdenticalFunctionsAndRejectsConflictingBodies) {
  ScopedAddKernel kernel;
  for (bool changed_builder : {false, true}) {
    for (bool changed_definition : {false, true}) {
      SCOPED_TRACE(changed_builder);
      SCOPED_TRACE(changed_definition);
      auto original = PreparationModel();
      original.add_opset("test.prepare", 1);
      original.ref_functions().push_back(PreparationAdd());
      core::builder::GraphBuilder builder(original, SchemaLookup());
      if (changed_builder) {
        builder.LocalFunction("Prepare").MakeNode("Neg", {"A"}, {"extra"});
      }
      auto plan = PreparationPlan();
      if (changed_definition) {
        plan.ref_added_functions()[0].ref_node()[0].set_op_type("Mul");
      }
      if (changed_builder || changed_definition) {
        EXPECT_THROW(builder.ApplyRewritings(plan), core::builder::BuilderError);
      } else {
        const auto result = builder.ApplyRewritings(plan).ToModel();
        ASSERT_EQ(result.functions_size(), 1);
        EXPECT_FLOAT_EQ(core::runtime::RunModel(result, {})[0].AsFloat()[0], 12);
      }
    }
  }
}

TEST(PatternOptimization, ModelReplayRetainsFunctionsUsedByReplacementNodes) {
  ScopedAddKernel kernel;
  ModelProto model = PreparationModel();
  RewritingProto plan;
  plan.ref_added_functions().push_back(PreparationAdd());
  auto &import = *plan.add_required_opset_imports();
  import.set_domain("test.prepare");
  import.set_version(1);
  auto &edit = *plan.add_rewrites();
  edit.add_matched_nodes(0);
  edit.ref_added_nodes().push_back(
      MakeNode("Prepare", {"a", "b"}, {"y", "unused"}, "test.prepare"));
  edit.add_added_nodes_positions(0);
  const auto result =
      core::builder::GraphBuilder(model, SchemaLookup()).ApplyRewritings(plan).ToModel();
  ASSERT_EQ(result.functions().size(), 1u);
  EXPECT_FLOAT_EQ(core::runtime::RunModel(result, {})[0].AsFloat()[0], 3);
}

TEST(PatternOptimization, ModelReplayRejectsRecipesReadingNewInputDefaults) {
  ScopedAddKernel kernel;
  for (const std::string name : {"literal", "c"}) {
    SCOPED_TRACE(name);
    auto model = PreparationModel();
    auto &input = *model.mutable_graph()->add_input();
    input.set_name(name);
    auto &type = *input.mutable_type()->mutable_tensor_type();
    type.set_elem_type(TensorProto::FLOAT);
    type.mutable_shape()->add_dim()->set_dim_value(1);
    EXPECT_THROW(core::builder::GraphBuilder(model, SchemaLookup())
                     .ApplyRewritings(PreparationPlan())
                     .ToModel(),
                 core::builder::BuilderError);
  }
}

TEST(PatternOptimization, ModelReplayPreservesDistinctFunctionDomainsAndOverloads) {
  ScopedAddKernel kernel;
  for (const bool overloaded : {false, true}) {
    SCOPED_TRACE(overloaded);
    auto original = PreparationModel();
    auto first = PreparationAdd();
    first.set_domain("first.test");
    original.ref_functions().push_back(first);
    auto plan = PreparationPlan();
    auto &second = plan.ref_added_functions()[0];
    if (overloaded) {
      second.set_domain("first.test");
      second.set_overload("twice");
    } else {
      second.set_domain("second.test");
    }
    // Distinct descriptors make an overload lookup error observable in inference.
    second.clr_output();
    second.add_output("C");
    for (auto &call : plan.ref_rewrites()[0].ref_initializer_calls()) {
      call.set_domain(second.domain().value());
      call.set_overload(second.overload().value());
      call.ref_output().resize(1);
    }
    auto &edit = plan.ref_rewrites()[0];
    edit.ref_computed_initializer_positions().clear();
    edit.add_computed_initializer_positions(0);
    edit.add_computed_initializer_positions(2);
    edit.ref_added_initializer_positions()[0] = 1;
    edit.ref_added_nodes()[0] =
        MakeNode("Prepare", {"c", "e"}, {"y"}, second.domain().value().c_str());
    edit.ref_added_nodes()[0].set_overload(second.overload().value());
    plan.ref_required_opset_imports()[0].set_domain(second.domain().value());
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    const auto result =
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded).ToModel();
    ASSERT_EQ(result.functions().size(), 2u);
    EXPECT_EQ(result.functions(0).domain(), "first.test");
    EXPECT_EQ(result.functions(1).overload().value(), overloaded ? "twice" : "");
    EXPECT_FLOAT_EQ(core::runtime::RunModel(result, {})[0].AsFloat()[0], 8);
    EXPECT_EQ(result.graph().output(0).type().tensor_type().shape().dim().size(), 1u);
  }
}

TEST(PatternOptimization, CapturesNestedGraphImportsWithoutMergingFunctionLocalImports) {
  for (bool conflicting_imports : {false, true}) {
    SCOPED_TRACE(conflicting_imports);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
    auto &branch = builder.MakeSubgraph("branch");
    branch.MakeNode("Identity", {"x"}, {"y"});
    branch.MakeOutput("y");
    AddSubgraphReference(builder, "body", "branch", "result");
    builder.MakeOutput("result");
    if (conflicting_imports) {
      auto &other = builder.MakeSubgraph("other");
      other.SetOpsetVersion("nested.test", 2);
      other.MakeNode("Neg", {"x"}, {"y"});
      other.MakeOutput("y");
      AddSubgraphReference(builder, "body", "other", "other_result");
      builder.MakeOutput("other_result");
    }
    ModelProto original;
    *original.mutable_graph() = builder.BuildGraph();
    AddOpsetImports(original, builder);
    core::builder::GraphGraph optimizer(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                     std::make_shared<RegisterNestedFunctionPattern>()});
    if (conflicting_imports) {
      EXPECT_THROW(optimizer.Optimize(), core::builder::BuilderError);
      continue;
    }
    const auto plan = optimizer.Optimize();
    ASSERT_EQ(plan.required_opset_imports().size(), 1u);
    EXPECT_EQ(plan.required_opset_imports(0).domain(), "nested.test");
    EXPECT_EQ(plan.required_opset_imports(0).version(), 1);
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    const auto replayed =
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded).ToModel();
    EXPECT_EQ(replayed.graph().SerializeAsString(), builder.ToGraph().SerializeAsString());
    const auto default_opset =
        std::find_if(replayed.opset_import().begin(), replayed.opset_import().end(),
                     [](const auto &opset) { return opset.domain() == kDefaultOnnxDomain; });
    ASSERT_NE(default_opset, replayed.opset_import().end());
    EXPECT_EQ(default_opset->version(), 18);
    ASSERT_EQ(replayed.functions().size(), 1u);
    const auto &imports = replayed.functions(0).opset_import();
    EXPECT_TRUE(std::any_of(imports.begin(), imports.end(), [](const auto &opset) {
      return (opset.domain().value().empty() || opset.domain().value() == "ai.onnx") &&
             opset.version() == 17;
    }));
  }
}

TEST(PatternOptimization, ModelReplayUsesNonCollidingOverloadedFunctionSignature) {
  ScopedAddKernel kernel;
  auto original = PreparationModel();
  auto first = PreparationAdd();
  first.set_name("LocalNeg");
  first.set_domain("");
  first.clr_output();
  first.add_output("C");
  original.ref_functions().push_back(first);
  auto second = PreparationAdd();
  second.set_name("LocalNeg");
  second.set_domain("ai.onnx");
  second.set_overload("pair");
  RewritingProto plan;
  plan.ref_added_functions().push_back(second);
  auto &edit = *plan.add_rewrites();
  edit.add_matched_nodes(0);
  edit.add_added_nodes_positions(0);
  auto call = MakeNode("LocalNeg", {"a", "b"}, {"y", "unused"});
  call.set_overload("pair");
  edit.ref_added_nodes().push_back(std::move(call));
  const auto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).ToModel();
  EXPECT_EQ(replayed.graph().node(0).overload(), "pair");
  EXPECT_FLOAT_EQ(core::runtime::RunModel(replayed, {})[0].AsFloat()[0], 3);
}

TEST(PatternOptimization, RejectsFunctionsIncompatibleWithExistingCalls) {
  for (bool multiple_edits : {false, true}) {
    SCOPED_TRACE(multiple_edits);
    auto original = PreparationModel();
    original.mutable_graph()->ref_node()[0] = MakeNode("Neg", {"a"}, {"y"});
    if (multiple_edits) {
      original.mutable_graph()->ref_node().push_back(MakeNode("Neg", {"b"}, {"z"}));
      auto output = original.graph().output(0);
      output.set_name("z");
      original.mutable_graph()->ref_output().push_back(output);
    }
    RewritingProto plan;
    auto function = PreparationAdd();
    function.set_name("Neg");
    function.set_domain("");
    function.clr_output();
    function.add_output("C");
    plan.ref_added_functions().push_back(function);
    EXPECT_THROW(
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).ToModel(),
        core::builder::BuilderError);
    auto &edit = *plan.add_rewrites();
    edit.add_matched_nodes(0);
    edit.add_added_nodes_positions(0);
    edit.ref_added_nodes().push_back(MakeNode("Neg", {"a", "b"}, {"y"}));
    if (multiple_edits) {
      auto literal = original.graph().initializer(0);
      literal.set_name("extra");
      edit.ref_added_initializers().push_back(literal);
      edit.add_added_initializer_positions(original.graph().initializer_size());
      auto &second = *plan.add_rewrites();
      second.add_matched_nodes(1);
      second.add_added_nodes_positions(1);
      second.ref_added_nodes().push_back(MakeNode("Neg", {"b", "a"}, {"z"}));
    }
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    core::builder::GraphBuilder builder(original, SchemaLookup());
    EXPECT_THROW(builder.ApplyRewritings(loaded), core::builder::BuilderError);
    EXPECT_EQ(builder.BuildGraph().SerializeAsString(), original.graph().SerializeAsString());
  }
}

TEST(PatternOptimization, RejectsCollidingFunctionLookupNames) {
  for (bool existing : {false, true}) {
    SCOPED_TRACE(existing);
    auto original = PreparationModel();
    auto first = PreparationAdd();
    first.set_name("F:part");
    first.set_overload("one");
    auto second = first;
    second.set_name("F");
    second.set_overload("part:one");
    RewritingProto plan;
    if (existing) {
      original.ref_functions().push_back(first);
    } else {
      plan.ref_added_functions().push_back(first);
    }
    plan.ref_added_functions().push_back(second);
    EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan),
                 core::builder::BuilderError);
  }
}

TEST(PatternOptimization, RejectsPreparationFunctionNameCollisions) {
  ScopedAddKernel kernel;
  auto original = PreparationModel();
  original.mutable_graph()->ref_node()[0] = MakeNode("Identity", {"a"}, {"y"});
  auto function = PreparationAdd();
  function.set_name("PrepareInitializer_1");
  function.set_domain("onnx.light.rewriting");
  original.ref_functions().push_back(function);
  core::builder::GraphBuilder builder(original, SchemaLookup());
  core::builder::GraphGraph optimizer(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                   std::make_shared<ReplaceIdentityWithConstantAddPattern>()});
  EXPECT_THROW(optimizer.Optimize(), core::builder::BuilderError);
}

TEST(PatternOptimization, UpdatesOnlyAffectedShapesAndRefreshesRemovedLastUses) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 18}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
  source.MakeNode("Abs", {"X"}, {"unchanged"});
  source.MakeNode("Neg", {"X"}, {"p"});
  source.MakeNode("Add", {"p", "p"}, {"Y"});
  source.MakeNode("Relu", {"p"}, {"unused"});
  source.MakeOutput("unchanged");
  source.MakeOutput("Y");
  RewritingProto plan;
  auto &remove = *plan.add_rewrites();
  remove.add_matched_nodes(3);
  auto &replace = *plan.add_rewrites();
  replace.add_matched_nodes(1);
  replace.add_added_nodes_positions(1);
  replace.ref_added_nodes().push_back(MakeNode("Identity", {"X"}, {"p"}));
  ScopedAbsShapeCounter counter;
  auto result = source.ApplyRewritings(plan);
  EXPECT_EQ(counter.calls, 1u);
  EXPECT_EQ(result.GetShape("p"), source.GetShape("p"));
  EXPECT_EQ(result.GetShape("Y"), source.GetShape("Y"));
  EXPECT_FALSE(result.HasShape("unused"));
  EXPECT_EQ(source.Nodes().size(), 4u);
  ModelProto model;
  *model.mutable_graph() = result.BuildGraph();
  AddOpsetImports(model, result);
  core::builder::GraphBuilder expected(model, SchemaLookup());
  EXPECT_EQ(result.Compute().ValueTags(), expected.Compute().ValueTags());
  EXPECT_EQ(result.Compute().NodeTags(), expected.Compute().NodeTags());
  EXPECT_EQ(result.Compute().NodeConstant(), expected.Compute().NodeConstant());
  EXPECT_EQ(result.Compute().Memory(), expected.Compute().Memory());
  EXPECT_EQ(result.Compute().PeakMemory(), expected.Compute().PeakMemory());
  for (std::size_t index = 0; index < result.Nodes().size(); ++index) {
    EXPECT_EQ(result.Compute().NodeReuse(index), expected.Compute().NodeReuse(index));
  }
}

TEST(PatternOptimization, OptimizerDoesNotReinferUnaffectedShapes) {
  auto original = PreparationModel();
  core::builder::GraphBuilder builder(original, SchemaLookup());
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Abs", {"X"}, {"unchanged"});
  builder.MakeOutput("unchanged");
  ScopedAbsShapeCounter counter;
  core::builder::GraphGraph optimizer(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                   std::make_shared<CountingFastOpTypePattern>()});
  core::builder::ConstantFoldingOptions options;
  options.enabled = false;
  EXPECT_FALSE(optimizer.Optimize(1, nullptr, options).rewrites().empty());
  EXPECT_EQ(counter.calls, 0u);
  EXPECT_EQ(builder.GetShape("unchanged"), builder.GetShape("X"));
}

TEST(PatternOptimization, PreservesExistingTagsWhenReplacingInputConsumers) {
  for (bool declared_tag : {false, true}) {
    SCOPED_TRACE(declared_tag);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 18}});
    ValueInfoProto input;
    input.set_name("X");
    auto &type = *input.mutable_type()->mutable_tensor_type();
    type.set_elem_type(TensorProto::FLOAT);
    type.mutable_shape()->add_dim()->set_dim_value(2);
    if (declared_tag) {
      auto &tag = *input.add_metadata_props();
      tag.set_key(core::compute::kValueTagMetadataKey);
      tag.set_value("shape");
    }
    source.MakeInput(input);
    source.MakeNode("Identity", {"X"}, {"p"});
    source.MakeNode("Neg", {"p"}, {"Y"});
    source.MakeNode("Unknown", {"Y"}, {"Z"});
    source.MakeOutput("Z");
    RewritingProto plan;
    auto &rewrite = *plan.add_rewrites();
    rewrite.add_matched_nodes(0);
    rewrite.add_matched_nodes(1);
    rewrite.ref_added_nodes().push_back(MakeNode("Neg", {"X"}, {"Y"}));
    rewrite.add_added_nodes_positions(0);
    const auto result = source.ApplyRewritings(plan);
    EXPECT_EQ(result.Nodes()[0].input(0), "X");
    for (const std::string name : {"X", "Y", "Z"}) {
      EXPECT_EQ(source.Compute().ValueTags().at(name), declared_tag ? "shape" : "weight");
      EXPECT_EQ(result.Compute().ValueTags().at(name), source.Compute().ValueTags().at(name));
    }
    EXPECT_EQ(result.Compute().ValueTags().count("p"), 0u);
  }
}

TEST(PatternOptimization, PreservesExistingTagsWhenRewritingMixedTagInputs) {
  for (bool new_intermediate : {false, true}) {
    SCOPED_TRACE(new_intermediate);
    core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 18}});
    source.MakeInput("X", core::symbolic::TensorType::kInt64, Shape());
    ValueInfoProto shape_input;
    shape_input.set_name("S");
    auto &type = *shape_input.mutable_type()->mutable_tensor_type();
    type.set_elem_type(TensorProto::INT64);
    type.mutable_shape()->add_dim()->set_dim_value(2);
    auto &tag = *shape_input.add_metadata_props();
    tag.set_key(core::compute::kValueTagMetadataKey);
    tag.set_value("shape");
    source.MakeInput(shape_input);
    source.MakeNode("Add", {"X", "S"}, {"Y"});
    source.MakeNode("Neg", {"Y"}, {"Z"});
    source.MakeOutput("Z");
    ASSERT_EQ(source.Compute().ValueTags().at("Y"), "weight");
    RewritingProto plan;
    auto &rewrite = *plan.add_rewrites();
    rewrite.add_matched_nodes(0);
    rewrite.add_added_nodes_positions(0);
    rewrite.ref_added_nodes().push_back(
        MakeNode("Add", {"S", "X"}, {new_intermediate ? "fresh" : "Y"}));
    if (new_intermediate) {
      rewrite.add_added_nodes_positions(1);
      rewrite.ref_added_nodes().push_back(MakeNode("Identity", {"fresh"}, {"Y"}));
    }
    auto result = source.ApplyRewritings(plan);
    for (const std::string name : {"X", "S", "Y", "Z"}) {
      EXPECT_EQ(result.Compute().ValueTags().at(name), source.Compute().ValueTags().at(name));
    }
    EXPECT_EQ(result.Compute().NodeTag(new_intermediate ? 1 : 0), "weight");
    if (new_intermediate) {
      EXPECT_EQ(result.Compute().ValueTags().at("fresh"), "shape");
    }
    auto tagged_output = shape_input;
    tagged_output.set_name("Y");
    result.MakeOutput(tagged_output);
    EXPECT_EQ(result.Compute().ValueTags().at("Y"), "shape");
  }
}

TEST(PatternOptimization, PreservesOutputOnlyTagsWhenFoldingToInitializers) {
  core::builder::GraphBuilder source("g", SchemaLookup(), {{"", 18}});
  source.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
  source.MakeNode("Shape", {"X"}, {"p"});
  source.MakeOutput("p");
  ASSERT_EQ(source.Compute().ValueTags().at("p"), "shape");
  RewritingProto plan;
  auto &rewrite = *plan.add_rewrites();
  rewrite.add_matched_nodes(0);
  rewrite.add_added_initializer_positions(0);
  rewrite.ref_added_initializers().push_back(MakeInitializer<int64_t>("p", {1}, {2}));
  const auto result = source.ApplyRewritings(plan);
  EXPECT_TRUE(result.Nodes().empty());
  ASSERT_EQ(result.Initializers().size(), 1u);
  EXPECT_EQ(result.Compute().ValueTags().at("p"), "shape");
  EXPECT_TRUE(result.Compute().IsConstantValue("p"));
  EXPECT_EQ(result.GetShape("p").Shape(), source.GetShape("p").Shape());
  EXPECT_EQ(result.GetShape("p").Dtype(), source.GetShape("p").Dtype());
  EXPECT_EQ(result.BuildGraph().output(0).name(), "p");
}

TEST(PatternOptimization, RefreshesReplacedCapturesWithoutAParentShapeFunction) {
  auto original = PreparationModel();
  original.mutable_graph()->clr_node();
  original.mutable_graph()->clr_output();
  core::builder::GraphBuilder source(original, SchemaLookup());
  auto &body = source.MakeSubgraph("body");
  body.MakeNode("Identity", {"a"}, {"out"});
  body.MakeOutput("out");
  AddSubgraphReference(source, "body", "body", "result");
  source.MakeOutput("result");
  RewritingProto plan;
  auto &nested = *plan.add_rewrites();
  nested.add_graph_path("body");
  nested.add_matched_nodes(0);
  nested.ref_added_nodes().push_back(MakeNode("Identity", {"b"}, {"out"}));
  nested.add_added_nodes_positions(0);
  auto &rewrite = *plan.add_rewrites();
  rewrite.add_removed_initializers(0);
  auto result = source.ApplyRewritings(plan);
  auto &updated = result.Subgraph("body");
  EXPECT_EQ(updated.Nodes()[0].input(0), "b");
  EXPECT_FALSE(updated.HasShape("a"));
  EXPECT_EQ(updated.GetShape("b"), source.GetShape("b"));
  updated.MakeNode("Identity", {"b"}, {"copy"});
  EXPECT_EQ(updated.GetShape("copy").Shape(), source.GetShape("b").Shape());
  EXPECT_EQ(updated.GetShape("copy").Dtype(), source.GetShape("b").Dtype());
  EXPECT_EQ(body.Nodes()[0].input(0), "a");
  EXPECT_TRUE(body.HasShape("a"));
}

TEST(PatternOptimization, ModelReplayDistinguishesFunctionNamesFromOverloads) {
  auto original = PreparationModel();
  original.add_opset("local.test", 1);
  for (bool overloaded : {false, true}) {
    auto &function = *original.add_functions();
    function.set_name(overloaded ? "F" : "F:pair");
    function.set_domain("local.test");
    if (overloaded)
      function.set_overload("pair");
    function.add_input("A");
    function.add_output("C");
    function.add_opset("", 18);
    auto &cast = *function.add_node();
    cast = MakeNode("Cast", {"A"}, {"C"});
    auto &attribute = *cast.add_attribute();
    attribute.set_name("to");
    attribute.set_type(AttributeProto::INT);
    attribute.set_i(overloaded ? TensorProto::INT64 : TensorProto::FLOAT);
  }
  original.mutable_graph()->ref_node()[0] = MakeNode("F:pair", {"a"}, {"y"}, "local.test");
  original.mutable_graph()->ref_node().push_back(MakeNode("F", {"a"}, {"z"}, "local.test"));
  original.mutable_graph()->ref_node().back().set_overload("pair");
  auto output = original.graph().output(0);
  output.set_name("z");
  output.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  original.mutable_graph()->ref_output().push_back(output);
  const auto replayed = core::builder::GraphBuilder(original, SchemaLookup())
                            .ApplyRewritings(RewritingProto{})
                            .ToModel();
  ASSERT_EQ(replayed.graph().node_size(), 2);
  EXPECT_EQ(replayed.graph().node(0).op_type(), "F:pair");
  EXPECT_EQ(replayed.graph().node(1).op_type(), "F");
  EXPECT_EQ(replayed.graph().node(1).overload(), "pair");
  core::builder::GraphBuilder builder(replayed, SchemaLookup());
  EXPECT_EQ(builder.GetShape("y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(builder.GetShape("z").Dtype(), core::symbolic::TensorType::kInt64);
}

TEST(PatternOptimization, RejectsModelLocalOperatorOverrides) {
  auto original = PreparationModel();
  FunctionProto function;
  function.set_name("Add");
  function.add_input("A");
  function.add_input("B");
  function.add_output("C");
  function.add_opset("", 18);
  function.ref_node().push_back(MakeNode("Mul", {"A", "B"}, {"C"}));
  original.add_function(function);
  EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()), core::builder::BuilderError);
}

TEST(PatternOptimization, RejectsLateFunctionOverridesOfCapturedRecipes) {
  ScopedAddKernel kernel;
  auto original = PreparationModel();
  original.mutable_graph()->ref_node()[0] = MakeNode("Identity", {"a"}, {"y"});
  original.mutable_graph()->ref_node().push_back(MakeNode("Relu", {"a"}, {"marker"}));
  auto output = original.graph().output(0);
  output.set_name("marker");
  original.mutable_graph()->ref_output().push_back(output);
  core::builder::GraphBuilder builder(original, SchemaLookup());
  core::builder::GraphGraph optimizer(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                   std::make_shared<ReplaceIdentityWithConstantAddPattern>(),
                   std::make_shared<RegisterLateAddOverridePattern>()});
  EXPECT_THROW(optimizer.Optimize(), core::builder::BuilderError);
  EXPECT_FALSE(builder.HasLocalFunction("Add"));
  const auto folded = std::find_if(builder.Initializers().begin(), builder.Initializers().end(),
                                   [](const auto &tensor) { return tensor.name() == "y"; });
  ASSERT_NE(folded, builder.Initializers().end());
  EXPECT_FLOAT_EQ(core::runtime::TensorFromProto(*folded).AsFloat()[0], 3);
}

TEST(PatternOptimization, RejectsOperatorOverridesInsideReachableFunctions) {
  ScopedAddKernel kernel;
  for (bool reachable : {false, true}) {
    for (bool indirect : {false, true}) {
      SCOPED_TRACE(reachable);
      SCOPED_TRACE(indirect);
      auto original = PreparationModel();
      original.add_opset("wrapper.test", 1);
      FunctionProto inner;
      inner.set_name("Inner");
      inner.set_domain("wrapper.test");
      inner.add_input("A");
      inner.add_output("C");
      inner.add_opset("", 18);
      inner.ref_node().push_back(MakeNode("Neg", {"A"}, {"C"}));
      original.ref_functions().push_back(inner);
      auto wrapper = inner;
      wrapper.set_name("Wrapper");
      if (indirect) {
        wrapper.add_opset("wrapper.test", 1);
        wrapper.ref_node()[0] = MakeNode("Inner", {"A"}, {"C"}, "wrapper.test");
      }
      original.ref_functions().push_back(wrapper);
      if (reachable)
        original.mutable_graph()->ref_node()[0] = MakeNode("Wrapper", {"a"}, {"y"}, "wrapper.test");
      RewritingProto plan;
      auto &override = *plan.add_added_functions();
      override.set_name("Neg");
      override.set_domain("ai.onnx");
      override.add_input("A");
      override.add_output("C");
      override.add_opset("", 18);
      override.ref_node().push_back(MakeNode("Add", {"A", "A"}, {"C"}));
      RewritingProto loaded;
      ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
      EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded),
                   core::builder::BuilderError);
      const auto functions = original.functions();
      original.ref_functions().clear();
      original.add_function(override);
      for (const auto &function : functions)
        original.add_function(function);
      EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()),
                   core::builder::BuilderError);
    }
  }
}

TEST(PatternOptimization, GraphReplayKeepsReachableFunctionImportsIndependent) {
  auto original = PreparationModel();
  original.add_opset("wrapper.test", 1);
  auto &inner = *original.add_functions();
  inner.set_name("Neg");
  inner.set_domain("function.local");
  inner.add_input("A");
  inner.add_output("C");
  inner.add_opset("", 18);
  inner.ref_node().push_back(MakeNode("Identity", {"A"}, {"C"}));
  auto &wrapper = *original.add_functions();
  wrapper.set_name("Wrapper");
  wrapper.set_domain("wrapper.test");
  wrapper.add_input("A");
  wrapper.add_output("C");
  wrapper.add_opset("function.local", 1);
  wrapper.ref_node().push_back(MakeNode("Neg", {"A"}, {"C"}, "function.local"));
  original.mutable_graph()->ref_node()[0] = MakeNode("Wrapper", {"a"}, {"y"}, "wrapper.test");
  RewritingProto plan;
  auto &opset = *plan.add_required_opset_imports();
  opset.set_domain("function.local");
  opset.set_version(1);
  EXPECT_NO_THROW(
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph());
}

TEST(PatternOptimization, RejectsOperatorOverridesInsideNestedFunctionGraphs) {
  auto original = PreparationModel();
  original.add_opset("wrapper.test", 1);
  TensorProto condition;
  condition.set_name("condition");
  condition.set_data_type(TensorProto::BOOL);
  condition.add_int32_data(1);
  original.mutable_graph()->ref_initializer().push_back(condition);
  auto &wrapper = *original.add_functions();
  wrapper.set_name("Wrapper");
  wrapper.set_domain("wrapper.test");
  wrapper.add_input("A");
  wrapper.add_input("condition");
  wrapper.add_output("C");
  wrapper.add_opset("", 18);
  auto &conditional = *wrapper.add_node();
  conditional = MakeNode("If", {"condition"}, {"C"});
  for (const char *name : {"then_branch", "else_branch"}) {
    auto &attribute = *conditional.add_attribute();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::GRAPH);
    auto &branch = *attribute.mutable_g();
    branch.set_name(name);
    branch.ref_node().push_back(MakeNode("Neg", {"A"}, {"branch_result"}));
    auto output = original.graph().output(0);
    output.set_name("branch_result");
    branch.ref_output().push_back(output);
  }
  original.mutable_graph()->ref_node()[0] =
      MakeNode("Wrapper", {"a", "condition"}, {"y"}, "wrapper.test");
  RewritingProto plan;
  auto &override = *plan.add_added_functions();
  override.set_name("Neg");
  override.add_input("A");
  override.add_output("C");
  override.add_opset("", 18);
  override.ref_node().push_back(MakeNode("Identity", {"A"}, {"C"}));
  EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan),
               core::builder::BuilderError);
  const auto functions = original.functions();
  original.ref_functions().clear();
  original.add_function(override);
  for (const auto &function : functions)
    original.add_function(function);
  EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()), core::builder::BuilderError);
}

TEST(PatternOptimization, BindsFunctionGraphDefaultsAtEachCallSite) {
  ScopedIdentityKernel kernel;
  for (bool explicit_binding : {false, true}) {
    SCOPED_TRACE(explicit_binding);
    auto original = PreparationModel();
    original.add_opset("wrapper.test", 1);
    TensorProto condition;
    condition.set_name("condition");
    condition.set_data_type(TensorProto::BOOL);
    condition.add_int32_data(1);
    original.mutable_graph()->ref_initializer().push_back(condition);
    FunctionProto wrapper;
    wrapper.set_name("Wrapper");
    wrapper.set_domain("wrapper.test");
    wrapper.add_input("a");
    wrapper.add_input("condition");
    wrapper.add_output("C");
    wrapper.add_opset("", 18);
    auto &conditional = *wrapper.add_node();
    conditional = MakeNode("If", {"condition"}, {"C"});
    for (const char *name : {"then_branch", "else_branch"}) {
      auto &reference = *conditional.add_attribute();
      reference.set_name(name);
      reference.set_type(AttributeProto::GRAPH);
      reference.set_ref_attr_name(name);
      auto &attribute = *wrapper.add_attribute_proto();
      attribute.set_name(name);
      attribute.set_type(AttributeProto::GRAPH);
      auto &branch = *attribute.mutable_g();
      branch.set_name(name);
      branch.ref_node().push_back(
          MakeNode(explicit_binding ? "Identity" : "Neg", {"a"}, {"branch_result"}));
      auto output = original.graph().output(0);
      output.set_name("branch_result");
      branch.ref_output().push_back(output);
    }
    original.ref_functions().push_back(wrapper);
    original.mutable_graph()->ref_node()[0] =
        MakeNode("Wrapper", {"a", "condition"}, {"y"}, "wrapper.test");
    if (explicit_binding) {
      auto second = MakeNode("Wrapper", {"a", "condition"}, {"z"}, "wrapper.test");
      second.ref_attribute() = wrapper.attribute_proto();
      for (auto &attribute : second.ref_attribute()) {
        attribute.mutable_g()->ref_node()[0].set_op_type("Neg");
        attribute.mutable_g()->ref_node()[0].ref_input()[0] = "a";
      }
      original.mutable_graph()->ref_node().push_back(std::move(second));
      auto output = original.graph().output(0);
      output.set_name("z");
      original.mutable_graph()->ref_output().push_back(output);
    }
    RewritingProto plan;
    RewritingProto loaded;
    ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
    const auto replayed =
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded).ToModel();
    const auto outputs = core::runtime::RunModel(replayed, {});
    EXPECT_FLOAT_EQ(outputs[0].AsFloat()[0], explicit_binding ? 1 : -1);
    if (explicit_binding) {
      ASSERT_EQ(outputs.size(), 2u);
      EXPECT_FLOAT_EQ(outputs[1].AsFloat()[0], -1);
    }
  }
}

TEST(PatternOptimization, ScopedCustomKernelRestoresFactory) {
  ScopedAddKernel outer;
  {
    ScopedAddKernel inner;
  }

  const NodeProto node = MakeNode("Add", {"a", "b"}, {"y"});
  core::runtime::RuntimeContext runtime;
  runtime.Set("a", core::runtime::Tensor::FromFloat("a", {1}, {1.0f}));
  runtime.Set("b", core::runtime::Tensor::FromFloat("b", {1}, {2.0f}));
  core::runtime::RunNode(node, runtime);

  ASSERT_TRUE(runtime.Has("y"));
  EXPECT_FLOAT_EQ(runtime.Get("y").AsFloat()[0], 3.0f);
}

TEST(PatternOptimization, SequentialRebasingPreservesReorderedFoldingBatch) {
  ScopedAddKernel kernel;
  core::builder::GraphBuilder builder("reordered", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  TensorProto base;
  base.set_name("base");
  base.set_data_type(TensorProto::FLOAT);
  base.add_dims(1);
  base.add_float_data(1.0f);
  builder.MakeInitializer(base);
  builder.MakeNode("Identity", {"x"}, {"first"});
  builder.MakeNode("Relu", {"x"}, {"second"});
  builder.MakeOutput("first");
  builder.MakeOutput("second");
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReorderedFoldingPattern>("Relu"));
  patterns.push_back(std::make_unique<ReorderedFoldingPattern>("Identity"));
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  const RewritingProto plan = optimizer.Optimize(1, nullptr, {}, false);
  ASSERT_EQ(plan.rewrites().size(), 2u);
  EXPECT_EQ(plan.rewrites(0).pattern_name(), "FoldRelu");
  EXPECT_EQ(plan.rewrites(0).matched_nodes().values(), (std::vector<uint64_t>{1}));
  EXPECT_EQ(plan.rewrites(1).matched_nodes().values(), (std::vector<uint64_t>{0}));
  EXPECT_EQ(plan.rewrites(0).added_nodes_positions().values(), (std::vector<uint64_t>{1}));
  EXPECT_EQ(plan.rewrites(1).added_nodes_positions().values(), (std::vector<uint64_t>{0}));
  EXPECT_EQ(plan.rewrites(0).added_initializer_positions().values(), (std::vector<uint64_t>{1}));
  EXPECT_EQ(plan.rewrites(0).computed_initializer_positions().values(), (std::vector<uint64_t>{2}));
  EXPECT_EQ(plan.rewrites(1).added_initializer_positions().values(), (std::vector<uint64_t>{2}));
  EXPECT_EQ(plan.rewrites(1).computed_initializer_positions().values(), (std::vector<uint64_t>{3}));
  ASSERT_EQ(plan.added_functions().size(), 1u);
  std::string bytes;
  ASSERT_TRUE(plan.SerializeToString(bytes));
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(bytes));
  EXPECT_TRUE(loaded.Equals(plan));
  EXPECT_EQ(builder.UndoRewritings(loaded).BuildGraph().SerializeAsString(),
            original.graph().SerializeAsString());
  EXPECT_EQ(core::builder::GraphBuilder(original, SchemaLookup())
                .ApplyRewritings(loaded)
                .BuildGraph()
                .SerializeAsString(),
            builder.BuildGraph().SerializeAsString());
}

TEST(PatternOptimization, SequentialRebasingPreservesSharedAnchorWithReorderedPreparation) {
  ScopedAddKernel kernel;
  core::builder::GraphBuilder builder("shared_anchor", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("base", {1}, {1.0f}));
  builder.MakeNode("Identity", {"x"}, {"first"});
  builder.MakeNode("Relu", {"x"}, {"second"});
  builder.MakeNode("Abs", {"x"}, {"anchor"});
  builder.MakeOutput("first");
  builder.MakeOutput("second");
  builder.MakeOutput("anchor");
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReorderedFoldingPattern>("Identity", true));
  patterns.push_back(std::make_unique<ReorderedFoldingPattern>("Relu", true));
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  const auto plan = optimizer.Optimize(1, nullptr, {}, false);
  ASSERT_EQ(plan.rewrites().size(), 2u);
  const auto &producer = plan.rewrites(0);
  const auto &consumer = plan.rewrites(1);
  EXPECT_EQ(producer.pattern_name(), "FoldRelu");
  EXPECT_EQ(consumer.pattern_name(), "FoldIdentity");
  ASSERT_EQ(producer.added_nodes().size(), 1u);
  ASSERT_EQ(consumer.added_nodes().size(), 1u);
  ASSERT_EQ(producer.initializer_calls().size(), 1u);
  ASSERT_EQ(consumer.initializer_calls().size(), 1u);
  ASSERT_EQ(producer.added_initializers().size(), 1u);
  EXPECT_EQ(producer.added_initializers(0).name(), "second_literal");
  ASSERT_EQ(consumer.initializer_calls(0).input().size(), 1u);
  EXPECT_EQ(consumer.initializer_calls(0).input(0), "second_literal");
  EXPECT_EQ(producer.added_nodes_positions().values(), (std::vector<uint64_t>{1}));
  EXPECT_EQ(consumer.added_nodes_positions().values(), (std::vector<uint64_t>{0}));
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].output(0), "first");
  EXPECT_EQ(builder.Nodes()[1].output(0), "second");
  EXPECT_EQ(builder.Nodes()[2].output(0), "anchor");
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
  EXPECT_TRUE(loaded.Equals(plan));
  EXPECT_EQ(core::builder::GraphBuilder(original, SchemaLookup())
                .ApplyRewritings(loaded)
                .BuildGraph()
                .SerializeAsString(),
            builder.BuildGraph().SerializeAsString());
}

TEST(PatternOptimization, SequentialRebasingPreservesLargeMixedBatch) {
  ScopedAddKernel kernel;
  core::builder::GraphBuilder builder("mixed_batch", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  TensorProto base;
  base.set_name("base");
  base.set_data_type(TensorProto::FLOAT);
  base.add_dims(1);
  base.add_float_data(1.0f);
  builder.MakeInitializer(base);
  constexpr std::size_t count = 64;
  for (std::size_t i = 0; i < count; ++i) {
    const std::string suffix = std::to_string(i);
    builder.MakeNode("Identity", {"x"}, {"identity_" + suffix});
    builder.MakeNode("Relu", {"x"}, {"relu_" + suffix});
    builder.MakeOutput("identity_" + suffix);
    builder.MakeOutput("relu_" + suffix);
  }
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReorderedFoldingPattern>("Relu"));
  patterns.push_back(std::make_unique<ReplaceIdentityWithTwoNodesPattern>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));
  const RewritingProto plan = optimizer.Optimize(1, nullptr, {}, false);
  ASSERT_GE(plan.rewrites().size(), 2 * count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto &folded = plan.rewrites(i);
    EXPECT_EQ(folded.matched_nodes().values(), (std::vector<uint64_t>{2 * i + 1}));
    EXPECT_EQ(folded.added_nodes_positions().values(), (std::vector<uint64_t>{2 * i + 1}));
    EXPECT_EQ(folded.added_initializer_positions().values(), (std::vector<uint64_t>{i + 1}));
    EXPECT_EQ(folded.computed_initializer_positions().values(), (std::vector<uint64_t>{2 * i + 2}));
    const auto &expanded = plan.rewrites(count + i);
    EXPECT_EQ(expanded.matched_nodes().values(), (std::vector<uint64_t>{3 * i}));
    EXPECT_EQ(expanded.added_nodes_positions().values(), (std::vector<uint64_t>{3 * i, 3 * i + 1}));
  }
  EXPECT_EQ(core::builder::GraphBuilder(original, SchemaLookup())
                .ApplyRewritings(plan)
                .BuildGraph()
                .SerializeAsString(),
            builder.BuildGraph().SerializeAsString());
}

TEST(PatternOptimization, ReplayBatchesIndependentEditsWithoutRepeatedGraphInference) {
  core::builder::GraphBuilder builder("independent_replay", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  RewritingProto plan;
  constexpr std::size_t count = 64;
  for (std::size_t i = 0; i < count; ++i) {
    const std::string output = "y" + std::to_string(i);
    builder.MakeNode("Abs", {"x"}, {output});
    builder.MakeOutput(output);
    auto *edit = plan.add_rewrites();
    edit->add_matched_nodes(i);
    edit->add_added_nodes(MakeNode("Neg", {"x"}, {output}));
    edit->add_added_nodes_positions(i);
  }
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);
  ScopedAbsShapeCounter counter;
  const auto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.node().size(), count);
  for (std::size_t i = 0; i < count; ++i)
    EXPECT_TRUE(replayed.node(i).Equals(plan.rewrites(i).added_nodes(0)));
  EXPECT_LE(counter.calls, 4 * count);
}

TEST(PatternOptimization, ReplayBatchStillValidatesLaterSequentialEdits) {
  core::builder::GraphBuilder builder("invalid_batch_replay", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  for (const char *name : {"first", "second"}) {
    builder.MakeNode("Abs", {"x"}, {name});
    builder.MakeOutput(name);
  }
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);
  for (bool duplicate : {false, true}) {
    RewritingProto plan;
    auto *first = plan.add_rewrites();
    first->add_matched_nodes(0);
    first->add_added_nodes(MakeNode("Neg", {"x"}, {"first"}));
    first->add_added_nodes_positions(0);
    auto *invalid = plan.add_rewrites();
    invalid->add_matched_nodes(duplicate ? 1 : 2);
    if (duplicate)
      invalid->add_matched_nodes(1);
    invalid->add_added_nodes(MakeNode("Neg", {"x"}, {"second"}));
    invalid->add_added_nodes_positions(1);
    EXPECT_THROW(
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph(),
        core::builder::BuilderError);
  }
}

TEST(PatternOptimization, ReplayRejectsForwardReferencesAcrossEdits) {
  for (bool independent_prefix : {false, true}) {
    SCOPED_TRACE(independent_prefix);
    core::builder::GraphBuilder builder("forward_reference_replay", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
    RewritingProto plan;
    if (independent_prefix) {
      builder.MakeNode("Abs", {"x"}, {"prefix"});
      auto *edit = plan.add_rewrites();
      edit->add_matched_nodes(0);
      edit->add_added_nodes(MakeNode("Neg", {"x"}, {"prefix"}));
      edit->add_added_nodes_positions(0);
    }
    const std::size_t position = builder.Nodes().size();
    builder.MakeNode("Abs", {"x"}, {"a"});
    builder.MakeNode("Abs", {"x"}, {"b"});
    builder.MakeOutput("a");
    const ModelProto original = builder.ToModel();
    auto *first = plan.add_rewrites();
    first->add_matched_nodes(position);
    first->add_added_nodes(MakeNode("Neg", {"later"}, {"a"}));
    first->add_added_nodes_positions(position);
    EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan),
                 core::builder::BuilderError);
    auto *second = plan.add_rewrites();
    second->add_matched_nodes(position + 1);
    second->add_added_nodes(MakeNode("Neg", {"x"}, {"later"}));
    second->add_added_nodes_positions(position + 1);
    EXPECT_THROW(core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan),
                 core::builder::BuilderError);
  }
}

TEST(PatternOptimization, ReplayPreservesCustomFunctionsNamedLikeNativeOperators) {
  for (const char *domain : {"local.test", "local.other"}) {
    for (std::size_t local_position : {0u, 1u}) {
      SCOPED_TRACE(domain);
      SCOPED_TRACE(local_position);
      core::builder::GraphBuilder builder("local_function_replay", SchemaLookup());
      builder.SetOpsetVersion("", 18);
      builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
      for (const char *output : {"first", "second"}) {
        builder.MakeNode("Abs", {"x"}, {output});
        builder.MakeOutput(output);
      }
      ModelProto original;
      *original.mutable_graph() = builder.BuildGraph();
      AddOpsetImports(original, builder);
      original.add_opset(domain, 1);
      auto *function = original.add_functions();
      function->set_domain(domain);
      function->set_name("Add");
      function->add_input("a");
      function->add_output("b");
      function->add_node(MakeNode("Identity", {"a"}, {"b"}));
      function->add_opset_import()->set_version(18);

      RewritingProto plan;
      for (std::size_t i = 0; i < 2; ++i) {
        auto *edit = plan.add_rewrites();
        edit->add_matched_nodes(i);
        edit->add_added_nodes(MakeNode(i == local_position ? "Add" : "Neg", {"x"},
                                       {i == 0 ? "first" : "second"},
                                       i == local_position ? domain : ""));
        edit->add_added_nodes_positions(i);
      }
      ModelProto sequential = original;
      for (const auto &edit : plan.rewrites()) {
        RewritingProto single;
        *single.add_rewrites() = edit;
        *sequential.mutable_graph() = core::builder::GraphBuilder(sequential, SchemaLookup())
                                          .ApplyRewritings(single)
                                          .BuildGraph();
      }
      EXPECT_TRUE(core::builder::GraphBuilder(original, SchemaLookup())
                      .ApplyRewritings(plan)
                      .BuildGraph()
                      .Equals(sequential.graph()));
    }
  }
}

TEST(PatternOptimization, ReplayReadsPreviouslyProducedInputsFromUpdatedState) {
  const auto constant = [](const std::vector<int64_t> &values) {
    NodeProto node = MakeNode("Constant", {}, {"a"});
    auto *attribute = node.add_attribute();
    attribute->set_name("value_ints");
    attribute->set_type(AttributeProto::INTS);
    for (int64_t value : values)
      attribute->add_ints(value);
    return node;
  };
  for (bool extra_edit : {false, true}) {
    SCOPED_TRACE(extra_edit);
    core::builder::GraphBuilder builder("dependent_replay", SchemaLookup());
    builder.SetOpsetVersion("", 18);
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(2));
    shape.PushBack(core::symbolic::SymDim(3));
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
    const NodeProto original_constant = constant({-1, -1});
    builder.MakeNode("Constant", {}, {"a"}, "", "", original_constant.attribute());
    if (extra_edit) {
      builder.MakeNode("Abs", {"x"}, {"independent"});
      builder.MakeOutput("independent");
    }
    builder.MakeNode("Abs", {"x"}, {"b"});
    builder.MakeOutput("b");
    ModelProto original;
    *original.mutable_graph() = builder.BuildGraph();
    AddOpsetImports(original, builder);

    RewritingProto plan;
    auto *first = plan.add_rewrites();
    first->add_matched_nodes(0);
    first->add_added_nodes(constant({2, 3}));
    first->add_added_nodes_positions(0);
    if (extra_edit) {
      auto *independent = plan.add_rewrites();
      independent->add_matched_nodes(1);
      independent->add_added_nodes(MakeNode("Neg", {"x"}, {"independent"}));
      independent->add_added_nodes_positions(1);
    }
    auto *dependent = plan.add_rewrites();
    dependent->add_matched_nodes(extra_edit ? 2 : 1);
    dependent->add_added_nodes(MakeNode("Reshape", {"x", "a"}, {"b"}));
    dependent->add_added_nodes_positions(extra_edit ? 2 : 1);

    ModelProto sequential = original;
    for (const auto &edit : plan.rewrites()) {
      RewritingProto single;
      *single.add_rewrites() = edit;
      *sequential.mutable_graph() = core::builder::GraphBuilder(sequential, SchemaLookup())
                                        .ApplyRewritings(single)
                                        .BuildGraph();
    }
    EXPECT_TRUE(core::builder::GraphBuilder(original, SchemaLookup())
                    .ApplyRewritings(plan)
                    .BuildGraph()
                    .Equals(sequential.graph()));
  }
}

TEST(PatternOptimization, ReplayRejectsMalformedPositionsAndFutureExecution) {
  core::builder::GraphBuilder builder("invalid", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Neg", {"x"}, {"y"});
  builder.MakeOutput("y");
  TensorProto tensor;
  tensor.set_name("weight");
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.add_dims(1);
  tensor.add_float_data(1);
  builder.MakeInitializer(tensor);
  const ModelProto original = builder.ToModel();
  const auto reject = [&](const auto &mutate) {
    RewritingProto plan;
    LocalRewritingProto &edit = *plan.add_rewrites();
    edit.add_matched_nodes(0);
    *edit.add_added_nodes() = MakeNode("Neg", {"x"}, {"y"});
    edit.add_added_nodes_positions(0);
    mutate(plan, edit);
    EXPECT_THROW(
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph(),
        core::builder::BuilderError);
  };
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  reject([&](auto &, auto &edit) { edit.ref_matched_nodes()[0] = maximum; });
  reject([](auto &, auto &edit) { edit.add_matched_nodes(0); });
  reject([&](auto &, auto &edit) { edit.add_removed_initializers(maximum); });
  reject([](auto &, auto &edit) {
    edit.add_removed_initializers(0);
    edit.add_removed_initializers(0);
  });
  reject([&](auto &, auto &edit) { edit.ref_added_nodes_positions()[0] = maximum; });
  reject([](auto &, auto &edit) { edit.clr_added_nodes_positions(); });
  reject([](auto &, auto &edit) { edit.ref_added_nodes()[0].set_domain("missing.opset"); });
  reject([](auto &, auto &edit) { edit.ref_added_nodes()[0].add_output("extra"); });
  reject([](auto &, auto &edit) {
    *edit.add_added_nodes() = MakeNode("Neg", {"x"}, {"other"});
    edit.add_added_nodes_positions(0);
  });
  for (bool remove : {false, true}) {
    reject([&](auto &, auto &edit) {
      if (remove)
        edit.add_removed_initializers(0);
      *edit.add_added_initializers() = tensor;
      edit.add_added_initializer_positions(maximum);
    });
    reject([&](auto &, auto &edit) {
      if (remove)
        edit.add_removed_initializers(0);
      *edit.add_added_initializers() = tensor;
    });
    reject([&](auto &, auto &edit) {
      if (remove)
        edit.add_removed_initializers(0);
      *edit.add_added_initializers() = tensor;
      *edit.add_added_initializers() = tensor;
      edit.add_added_initializer_positions(0);
      edit.add_added_initializer_positions(0);
    });
  }
  reject([](auto &plan, auto &) { plan.add_added_functions(); });
  reject([](auto &plan, auto &) { plan.add_required_opset_imports(); });
  reject([](auto &, auto &edit) { edit.add_initializer_calls(); });
  reject([](auto &, auto &edit) { edit.add_computed_initializer_positions(0); });
  reject([](auto &, auto &edit) { edit.add_graph_path("missing"); });
  reject([&](auto &, auto &edit) {
    edit.add_removed_initializers(0);
    auto invalid = tensor;
    invalid.set_name("x");
    invalid.set_data_type(TensorProto::INT64);
    *edit.add_added_initializers() = invalid;
    edit.add_added_initializer_positions(0);
  });
}

TEST(PatternOptimization, ReplayRejectsInitializerNameAndInputDefaultConflicts) {
  core::builder::GraphBuilder builder("initializer_validation", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("old", {1}, {1.0f}));
  StructTypeProto record;
  record.set_type_id(1);
  record.set_name("record");
  auto &field = *record.mutable_structure()->add_field();
  field.set_name("value");
  auto &type = *field.mutable_type()->mutable_tensor_type();
  type.set_elem_type(TensorProto::FLOAT);
  type.mutable_shape();
  builder.MakeStructType(record);
  EncodedValueProto encoded;
  encoded.set_name("encoded");
  encoded.mutable_struct_type()->set_type_ref(1);
  encoded.set_raw_data(std::string(4, '\0'));
  builder.MakeEncodedInitializer(encoded);
  builder.MakeNode("Neg", {"x"}, {"y"});
  builder.MakeOutput("y");
  const ModelProto original = builder.ToModel();
  const std::vector<TensorProto> invalid = {
      MakeInitializer<float>("y", {2}, {1.0f, 2.0f}),
      MakeInitializer<float>("encoded", {2}, {1.0f, 2.0f}),
      MakeInitializer<int64_t>("x", {2}, {1, 2}),
      MakeInitializer<float>("x", {}, {1.0f}),
      MakeInitializer<float>("x", {3}, {1.0f, 2.0f, 3.0f}),
  };
  for (const bool remove : {false, true}) {
    SCOPED_TRACE(remove);
    for (const TensorProto &tensor : invalid) {
      SCOPED_TRACE(tensor.name().value());
      RewritingProto plan;
      auto &edit = *plan.add_rewrites();
      if (remove) {
        edit.add_removed_initializers(0);
      }
      *edit.add_added_initializers() = tensor;
      edit.add_added_initializer_positions(remove ? 0 : 1);
      EXPECT_THROW(
          core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph(),
          core::builder::BuilderError);
    }
  }
}

TEST(PatternOptimization, ReplayValidatesReplacementDefaultsAndPreservesRetainedDefaults) {
  core::builder::GraphBuilder builder("replacement_default", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("x", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("old", {1}, {3.0f}));
  builder.MakeNode("Neg", {"x"}, {"y"});
  builder.MakeOutput("y");
  const ModelProto original = builder.ToModel();
  for (const TensorProto &tensor : {
           MakeInitializer<float>("x", {2}, {3.0f, 4.0f}),
           MakeInitializer<int64_t>("x", {2}, {1, 2}),
           MakeInitializer<float>("x", {}, {1.0f}),
           MakeInitializer<float>("x", {3}, {1.0f, 2.0f, 3.0f}),
       }) {
    RewritingProto plan;
    auto &edit = *plan.add_rewrites();
    edit.add_removed_initializers(0);
    *edit.add_added_initializers() = tensor;
    edit.add_added_initializer_positions(0);
    EXPECT_THROW(
        core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph(),
        core::builder::BuilderError);
  }
  RewritingProto plan;
  auto &edit = *plan.add_rewrites();
  edit.add_removed_initializers(1);
  *edit.add_added_initializers() = MakeInitializer<float>("fresh", {1}, {4.0f});
  edit.add_added_initializer_positions(1);
  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.initializer().size(), 2u);
  EXPECT_EQ(replayed.initializer(0).SerializeAsString(),
            original.graph().initializer(0).SerializeAsString());
  EXPECT_EQ(replayed.input(0).SerializeAsString(), original.graph().input(0).SerializeAsString());
}

TEST(PatternOptimization, ReplayRejectsSameNameInitializerReplacement) {
  ScopedAddKernel kernel;
  for (bool input_default : {false, true}) {
    for (bool computed : {false, true}) {
      SCOPED_TRACE(input_default);
      SCOPED_TRACE(computed);
      core::builder::GraphBuilder source("replacement", SchemaLookup(), {{"", 18}});
      core::symbolic::SymShape input_shape;
      input_shape.PushBack(core::symbolic::SymDim(1));
      if (input_default)
        source.MakeInput("p", core::symbolic::TensorType::kFloat, input_shape);
      source.MakeInitializer(MakeInitializer<float>("p", {1}, {1.0f}));
      source.MakeInitializer(MakeInitializer<float>("seed", {1}, {3.0f}));
      source.MakeNode("Identity", {"seed"}, {"y"});
      source.MakeOutput("p");
      source.MakeOutput("y");
      const auto original = source.ToModel().SerializeAsString();
      const auto tags = source.Compute().ValueTags();
      const auto node_tags = source.Compute().NodeTags();
      const auto memory = source.Compute().Memory();
      const auto shape = source.GetShape("p");
      RewritingProto plan;
      auto &edit = *plan.add_rewrites();
      edit.add_removed_initializers(0);
      if (computed) {
        plan.ref_added_functions().push_back(PreparationAdd());
        auto &opset = *plan.add_required_opset_imports();
        opset.set_domain("test.prepare");
        opset.set_version(1);
        edit.ref_initializer_calls().push_back(
            MakeNode("Prepare", {"seed", "seed"}, {"p", "other"}, "test.prepare"));
        edit.add_computed_initializer_positions(0);
        edit.add_computed_initializer_positions(2);
      } else {
        edit.ref_added_initializers().push_back(MakeInitializer<float>("p", {1}, {5.0f}));
        edit.add_added_initializer_positions(0);
      }
      RewritingProto loaded;
      ASSERT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
      EXPECT_THROW(source.ApplyRewritings(loaded), core::builder::BuilderError);
      EXPECT_EQ(source.ToModel().SerializeAsString(), original);
      EXPECT_EQ(source.Compute().ValueTags(), tags);
      EXPECT_EQ(source.Compute().NodeTags(), node_tags);
      EXPECT_EQ(source.Compute().Memory(), memory);
      EXPECT_EQ(source.GetShape("p"), shape);
      EXPECT_EQ(source.Nodes()[0].input(0), "seed");
    }
  }
}

TEST(PatternOptimization, ReplayAllowsFreshInitializerWithRewrittenReferences) {
  core::builder::GraphBuilder source("fresh_initializer", SchemaLookup(), {{"", 18}});
  source.MakeInitializer(MakeInitializer<float>("p", {2}, {1.0f, 2.0f}));
  source.MakeNode("Identity", {"p"}, {"y"});
  source.MakeOutput("y");
  RewritingProto plan;
  auto &edit = *plan.add_rewrites();
  edit.add_matched_nodes(0);
  edit.ref_added_nodes().push_back(MakeNode("Identity", {"fresh"}, {"y"}));
  edit.add_added_nodes_positions(0);
  edit.add_removed_initializers(0);
  edit.ref_added_initializers().push_back(MakeInitializer<float>("fresh", {2}, {3.0f, 4.0f}));
  edit.add_added_initializer_positions(0);
  const auto result = source.ApplyRewritings(plan);
  ASSERT_EQ(result.Initializers().size(), 1u);
  EXPECT_EQ(result.Initializers()[0].name(), "fresh");
  EXPECT_EQ(result.Initializers()[0].SerializeAsString(),
            edit.added_initializers(0).SerializeAsString());
  EXPECT_EQ(result.Nodes()[0].input(0), "fresh");
  EXPECT_FALSE(result.HasShape("p"));
  EXPECT_EQ(result.GetShape("fresh").Shape(), source.GetShape("p").Shape());
  EXPECT_EQ(result.GetShape("fresh").Dtype(), source.GetShape("p").Dtype());
}

TEST(PatternOptimization, ReplayAllowsFirstInputDefault) {
  core::builder::GraphBuilder source("first_default", SchemaLookup(), {{"", 18}});
  source.MakeInput("p", core::symbolic::TensorType::kFloat, Shape());
  source.MakeOutput("p");
  RewritingProto plan;
  auto &edit = *plan.add_rewrites();
  edit.ref_added_initializers().push_back(MakeInitializer<float>("p", {2}, {1.0f, 2.0f}));
  edit.add_added_initializer_positions(0);
  const auto result = source.ApplyRewritings(plan);
  ASSERT_EQ(result.Initializers().size(), 1u);
  EXPECT_EQ(result.Initializers()[0].name(), "p");
  EXPECT_EQ(result.GetShape("p"), source.GetShape("p"));
  EXPECT_FALSE(result.Compute().IsConstantValue("p"));
  EXPECT_EQ(result.BuildGraph().input(0).name(), "p");
}

TEST(PatternOptimization, ReplayPositionsInitializersAmongRetainedValues) {
  core::builder::GraphBuilder builder("initializer_positions", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  TensorProto tensor;
  tensor.set_name("old");
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.add_dims(1);
  tensor.add_float_data(1);
  builder.MakeInitializer(tensor);
  tensor.set_name("retained");
  builder.MakeInitializer(tensor);
  builder.MakeNode("Add", {"x", "retained"}, {"y"});
  builder.MakeOutput("y");
  const ModelProto original = builder.ToModel();
  RewritingProto plan;
  auto &edit = *plan.add_rewrites();
  edit.add_removed_initializers(0);
  tensor.set_name("new");
  *edit.add_added_initializers() = tensor;
  edit.add_added_initializer_positions(0);
  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.initializer().size(), 2u);
  EXPECT_EQ(replayed.initializer(0).name(), "new");
  EXPECT_EQ(replayed.initializer(1).name(), "retained");
}

TEST(PatternOptimization, IgnoresNullPositionalPlaceholders) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  auto pattern = std::make_shared<NullPlaceholderPattern>();
  core::builder::GraphGraph graph(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{pattern});
  const RewritingProto plan = graph.Optimize(1);
  const auto &rewrites = plan.rewrites();

  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(rewrites[0].matched_nodes().values(), std::vector<uint64_t>({0}));
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Relu");
}

TEST(PatternOptimization, FastOpTypeIsCachedAcrossOptimizationIterations) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Add", {"x", "x"}, {"y"});
  builder.MakeOutput("y");
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  auto pattern = std::make_shared<CountingFastOpTypePattern>();
  std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns = {pattern};
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const RewritingProto plan = graph.Optimize(4);
  const auto &rewrites = plan.rewrites();

  EXPECT_EQ(rewrites.size(), 4u);
  EXPECT_EQ(pattern->fast_op_type_calls, 1u);
  std::string bytes;
  ASSERT_TRUE(plan.SerializeToString(bytes));
  RewritingProto loaded;
  ASSERT_TRUE(loaded.ParseFromString(bytes));
  EXPECT_EQ(core::builder::GraphBuilder(original, SchemaLookup())
                .ApplyRewritings(loaded)
                .BuildGraph()
                .SerializeAsString(),
            builder.BuildGraph().SerializeAsString());
}

TEST(PatternOptimization, CastCastCollapsesRedundantOuterCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  EXPECT_EQ(pattern.priority, 1);
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Cast"}));
  EXPECT_EQ(pattern.ToString(),
            "PatternOptimization(name=CastCast, priority=1, fast_op_types=[Cast])");

  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);
  EXPECT_EQ(match.insert_at, match.nodes[1]);
  EXPECT_EQ(match.ToString(), "MatchResult(pattern=CastCast, nodes=[Cast(outputs=[middle]), "
                              "Cast(outputs=[y])], insert_at=Cast(outputs=[y]))");

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  ASSERT_NE(FindAttribute(replacements[0], "to"), nullptr);
  EXPECT_EQ(FindAttribute(replacements[0], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
}

TEST(PatternOptimization, CastReplacesRedundantConversionWithIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "y", TensorProto::DataType::FLOAT, "redundant");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastPattern pattern;
  EXPECT_EQ(pattern.priority, 0);
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Cast"}));

  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.nodes, std::vector<const NodeProto *>({&builder.Nodes()[0]}));
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(replacements[0].name().value(), "CastPattern--redundant");
  EXPECT_TRUE(replacements[0].attribute().empty());
}

TEST(PatternOptimization, CastRejectsTypeChangingConversion) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "y", TensorProto::DataType::DOUBLE);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  EXPECT_TRUE(match.nodes.empty());
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_TRUE(match.no_match->source_file.ends_with("cast_pattern.cc"));
  EXPECT_GT(match.no_match->source_line, 0u);
  EXPECT_EQ(match.no_match->reason, "the input and target element types differ");
  EXPECT_NE(match.ToString().find("cast_pattern.cc:"), std::string::npos);
  EXPECT_NE(match.ToString().find("the input and target element types differ"), std::string::npos);

  core::builder::OptimizationReport report;
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastPattern>());
  core::builder::GraphGraph diagnostic_graph(builder, std::move(patterns));
  const RewritingProto plan = diagnostic_graph.Optimize(-1, &report);
  const auto &rewrites = plan.rewrites();
  EXPECT_TRUE(rewrites.empty());
  ASSERT_EQ(report.patterns.size(), 1u);
  ASSERT_EQ(report.patterns[0].no_matches.size(), 1u);
  EXPECT_TRUE(report.patterns[0].no_matches[0].source_file.ends_with("cast_pattern.cc"));
  EXPECT_EQ(report.patterns[0].no_matches[0].source_line, match.no_match->source_line);
  EXPECT_EQ(report.patterns[0].no_matches[0].reason, "the input and target element types differ");
  EXPECT_EQ(report.patterns[0].no_matches[0].occurrences, 1u);
}

TEST(PatternOptimization, ApplyContractErrorIncludesSourceLocation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastPattern pattern;

  try {
    pattern.Apply(graph, {});
    FAIL() << "Expected CastPattern::Apply to reject an invalid direct call.";
  } catch (const core::builder::BuilderError &error) {
    EXPECT_TRUE(error.SourceFile().ends_with("cast_pattern.cc"));
    EXPECT_GT(error.SourceLine(), 0u);
    EXPECT_NE(std::string(error.what()).find("cast_pattern.cc:"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("expects one default-domain Cast node"),
              std::string::npos);
  }
}

TEST(PatternOptimization, OptimizeCastProducesExpectedGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"middle"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "middle");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "middle");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "y");
}

TEST(PatternOptimization, AcceptsCustomPriority) {
  onnx_patterns::CastCastPattern pattern(3);
  EXPECT_EQ(pattern.priority, 3);
}

TEST(PatternOptimization, CastCastUsesIdentityForSafeRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_TRUE(replacements[0].attribute().empty());
}

TEST(PatternOptimization, CastCastKeepsSharedInnerCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"middle"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, nullptr);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].output()[0].value(), "middle");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
}

TEST(PatternOptimization, CastCastRejectsLossyRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::INT32);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  EXPECT_TRUE(match.nodes.empty());
}

TEST(PatternOptimization, OptimizeAppliesPatternAndCleanupPasses) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"x"}, {"unused"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const RewritingProto plan = graph.Optimize();
  const auto &rewrites = plan.rewrites();
  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(rewrites[0].pattern_name(), "CastCast");
  EXPECT_EQ(rewrites[0].matched_nodes().values(), std::vector<uint64_t>({0, 1}));
  EXPECT_EQ(rewrites[0].added_nodes().size(), 1u);
  EXPECT_EQ(rewrites[0].added_nodes_positions().values(), std::vector<uint64_t>({0}));
  EXPECT_TRUE(rewrites[0].added_initializers().empty());
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Cast");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(PatternOptimization, CleanupPassesProduceReplayableRewritingSequence) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("c1", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("c2", {1}, {1.0f}));
  builder.MakeNode("Neg", {"x"}, {"a"});
  builder.MakeNode("Neg", {"x"}, {"b"});
  builder.MakeNode("Identity", {"b"}, {"forwarded"});
  builder.MakeNode("Add", {"a", "forwarded"}, {"sum"});
  builder.MakeNode("Add", {"sum", "c1"}, {"biased"});
  builder.MakeNode("Add", {"biased", "c2"}, {"y"});
  builder.MakeNode("Relu", {"x"}, {"dead"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  EXPECT_EQ(graph.Canonicalize(), 3u);
  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);
  const RewritingProto plan = graph.Optimize();
  const auto &rewrites = plan.rewrites();

  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(rewrites[0].pattern_name(), "RemoveUnusedNodes");

  const GraphProto optimized = builder.BuildGraph();
  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
  EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
            original.graph().SerializeAsString());
}

TEST(PatternOptimization, OptimizeReportsPhaseAndPatternTimings) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;

  const RewritingProto plan = graph.Optimize(-1, &report);
  const auto &rewrites = plan.rewrites();
  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(report.rewrites, 1u);
  EXPECT_EQ(report.iterations, 2u);
  EXPECT_GE(report.matching_time_ns, 0);
  EXPECT_GE(report.rewriting_time_ns, 0);
  EXPECT_GE(report.cleanup_time_ns, 0);
  EXPECT_GE(report.constant_folding_time_ns, 0);
  EXPECT_EQ(report.subgraph_optimization_time_ns, 0);
  ASSERT_EQ(report.patterns.size(), 1u);
  EXPECT_EQ(report.patterns[0].pattern_name, "CastCast");
  EXPECT_EQ(report.patterns[0].attempts, 3u);
  EXPECT_EQ(report.patterns[0].matches, 1u);
  EXPECT_GE(report.patterns[0].match_time_ns, 0);
  EXPECT_GE(report.patterns[0].apply_time_ns, 0);
  EXPECT_EQ(report.TotalTimeNs(), report.matching_time_ns + report.rewriting_time_ns +
                                      report.cleanup_time_ns + report.constant_folding_time_ns);
  EXPECT_NE(report.ToString().find("phases={matching:"), std::string::npos);
  EXPECT_NE(report.ToString().find("CastCast(attempts=3, matches=1"), std::string::npos);
}

TEST(PatternOptimization, ParallelMatchingPreservesCandidateOrderAndStatistics) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  for (std::size_t i = 0; i < 64; ++i) {
    const std::string input = "x" + std::to_string(i);
    builder.MakeInput(input, core::symbolic::TensorType::kFloat, Shape());
    const std::string output = "y" + std::to_string(i);
    builder.MakeNode("Add", {input, input}, {output});
    builder.MakeOutput(output);
  }

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<CountingFastOpTypePattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;

  const RewritingProto plan = graph.Optimize(1, &report, {}, false, {}, true);

  ASSERT_EQ(report.patterns.size(), 1u);
  EXPECT_EQ(report.patterns[0].attempts, 64u);
  EXPECT_EQ(report.patterns[0].matches, 64u);
  ASSERT_EQ(plan.rewrites_size(), 64);
  for (int i = 0; i < plan.rewrites_size(); ++i) {
    ASSERT_EQ(plan.rewrites(i).matched_nodes_size(), 1);
    EXPECT_EQ(plan.rewrites(i).matched_nodes(0), static_cast<uint64_t>(i));
  }
}

TEST(PatternOptimization, ParallelMatchingRespectsEachPatternsThreadSafety) {
  for (bool parallel : {false, true}) {
    SCOPED_TRACE(parallel);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    for (std::size_t i = 0; i < 64; ++i) {
      const std::string input = "x" + std::to_string(i);
      const std::string output = "y" + std::to_string(i);
      builder.MakeInput(input, core::symbolic::TensorType::kFloat, Shape());
      builder.MakeNode("Add", {input, input}, {output});
      builder.MakeOutput(output);
    }
    auto sequential_pattern = std::make_shared<ThreadTrackingPattern>();
    auto parallel_pattern = std::make_shared<ParallelThreadTrackingPattern>();
    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                     sequential_pattern, parallel_pattern});
    core::builder::OptimizationReport report;
    core::runtime::ParallelRegionCollector collector(2);
    core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
    const RewritingProto plan = graph.Optimize(1, &report, {}, false, {}, parallel);

    EXPECT_EQ(plan.rewrites_size(), 0);
    ASSERT_EQ(report.patterns.size(), 2u);
    EXPECT_EQ(report.patterns[0].attempts, 64u);
    EXPECT_EQ(report.patterns[1].attempts, 64u);
    EXPECT_EQ(sequential_pattern->worker_calls.load(), 0u);
    EXPECT_EQ(collector.dropped_events(), 0u);
    ASSERT_EQ(collector.events().size(), parallel ? 1u : 0u);
    if (parallel) {
      EXPECT_EQ(collector.events()[0].total_iterations, 64);
    } else {
      EXPECT_EQ(parallel_pattern->worker_calls.load(), 0u);
    }
  }
}

TEST(PatternOptimization, OptimizesNestedSubgraphsAndReplaysByGraphPath) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  builder.MakeNode("Neg", {"x"}, {"captured"});

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  AddCast(body, "captured", "body_middle", TensorProto::DataType::FLOAT);
  AddCast(body, "body_middle", "body_y", TensorProto::DataType::FLOAT);
  body.MakeOutput("body_y");

  core::builder::GraphBuilder &leaf = body.MakeSubgraph("leaf");
  AddCast(leaf, "captured", "leaf_middle", TensorProto::DataType::FLOAT);
  AddCast(leaf, "leaf_middle", "leaf_y", TensorProto::DataType::FLOAT);
  leaf.MakeOutput("leaf_y");
  AddSubgraphReference(body, "body", "leaf", "leaf_result");
  body.MakeOutput("leaf_result");

  AddSubgraphReference(builder, "body", "body", "body_result");
  builder.MakeOutput("body_result");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;
  const RewritingProto plan = graph.Optimize(-1, &report);
  const auto &rewrites = plan.rewrites();

  ASSERT_EQ(rewrites.size(), 2u);
  ASSERT_EQ(rewrites[0].graph_path().size(), 2u);
  EXPECT_EQ(rewrites[0].graph_path(0), "body");
  EXPECT_EQ(rewrites[0].graph_path(1), "leaf");
  ASSERT_EQ(rewrites[1].graph_path().size(), 1u);
  EXPECT_EQ(rewrites[1].graph_path(0), "body");
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 2u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Cast");
  ASSERT_EQ(builder.Subgraph("body").Subgraph("leaf").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Subgraph("leaf").Nodes()[0].op_type().value(), "Cast");

  EXPECT_EQ(report.rewrites, 2u);
  EXPECT_GT(report.subgraph_optimization_time_ns, 0);
  ASSERT_EQ(report.patterns.size(), 1u);
  EXPECT_EQ(report.patterns[0].matches, 2u);
  ASSERT_EQ(report.subgraphs.size(), 2u);
  EXPECT_EQ(report.subgraphs[0].graph_path, std::vector<std::string>({"body"}));
  EXPECT_EQ(report.subgraphs[0].rewrites, 2u);
  EXPECT_EQ(report.subgraphs[1].graph_path, std::vector<std::string>({"body", "leaf"}));
  EXPECT_EQ(report.subgraphs[1].rewrites, 1u);
  EXPECT_NE(report.ToString().find("Subgraph(path=body/leaf"), std::string::npos);
  EXPECT_EQ(report.TotalTimeNs(), report.matching_time_ns + report.rewriting_time_ns +
                                      report.cleanup_time_ns + report.constant_folding_time_ns +
                                      report.subgraph_optimization_time_ns);

  const GraphProto optimized = builder.BuildGraph();
  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
  EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
            original.graph().SerializeAsString());
  auto invalid = plan;
  invalid.ref_rewrites()[0].ref_graph_path()[1] = "missing";
  EXPECT_THROW(builder.UndoRewritings(invalid), core::builder::BuilderError);
  EXPECT_EQ(builder.BuildGraph().SerializeAsString(), optimized.SerializeAsString());
}

TEST(PatternOptimization, ReportsMixedGraphAndFunctionRewritesExactlyOnce) {
  for (const int max_iter : {0, 1, -1}) {
    SCOPED_TRACE(max_iter);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
    AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
    AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
    builder.MakeOutput("y");
    auto &body = builder.MakeSubgraph("body");
    AddCast(body, "x", "body_middle", TensorProto::DataType::FLOAT);
    AddCast(body, "body_middle", "out", TensorProto::DataType::FLOAT);
    body.MakeOutput("out");
    AddSubgraphReference(builder, "body", "body", "body_out");
    builder.MakeOutput("body_out");
    for (auto *scope : {&builder, &body}) {
      auto &function = scope->MakeLocalFunction(
          scope == &builder ? "RootFunction" : "NestedFunction", "test.pattern");
      function.MakeInput("input", core::symbolic::TensorType::kFloat16, Shape());
      AddCast(function, "input", "middle", TensorProto::DataType::FLOAT);
      AddCast(function, "middle", "output", TensorProto::DataType::FLOAT);
      function.MakeOutput("output");
    }
    core::builder::GraphGraph graph(
        builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{
                     std::make_shared<onnx_patterns::CastCastPattern>()});
    core::builder::OptimizationReport report;
    core::builder::ConstantFoldingOptions folding;
    folding.enabled = false;
    const auto plan = graph.Optimize(max_iter, &report, folding, false);
    EXPECT_EQ(report.rewrites, max_iter == 0 ? 0u : 4u);
    EXPECT_EQ(plan.rewrites().size(), max_iter == 0 ? 0u : 2u);
    ASSERT_EQ(report.subgraphs.size(), 3u);
    EXPECT_EQ(report.subgraphs[0].graph_path, std::vector<std::string>({"body"}));
    EXPECT_EQ(report.subgraphs[0].rewrites, max_iter == 0 ? 0u : 2u);
    EXPECT_EQ(report.subgraphs[1].graph_path,
              std::vector<std::string>({"body", "function:NestedFunction"}));
    EXPECT_EQ(report.subgraphs[1].rewrites, max_iter == 0 ? 0u : 1u);
    EXPECT_EQ(report.subgraphs[2].graph_path, std::vector<std::string>({"function:RootFunction"}));
    EXPECT_EQ(report.subgraphs[2].rewrites, max_iter == 0 ? 0u : 1u);
  }
}

TEST(PatternOptimization, OptimizesAndReplaysLocalFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.SetOpsetVersion("test.pattern", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());

  core::builder::GraphBuilder &function = builder.MakeLocalFunction("CastPair", "test.pattern");
  function.MakeInput("input", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(function, "input", "middle", TensorProto::DataType::FLOAT);
  AddCast(function, "middle", "output", TensorProto::DataType::FLOAT);
  function.MakeOutput("output");
  builder.MakeNode("CastPair", {"x"}, {"y"}, "test.pattern");
  builder.MakeOutput("y");

  const ModelProto original = builder.ToModel();
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;
  const RewritingProto plan = graph.Optimize(-1, &report);

  EXPECT_TRUE(plan.rewrites().empty());
  ASSERT_EQ(plan.added_functions().size(), 1u);
  ASSERT_EQ(plan.replaced_function_ids().size(), 1u);
  EXPECT_EQ(plan.replaced_function_ids(0), "test.pattern:CastPair");
  ASSERT_EQ(plan.added_functions(0).node().size(), 1u);
  EXPECT_EQ(plan.added_functions(0).node(0).op_type().value(), "Cast");
  ASSERT_EQ(builder.LocalFunction("CastPair").Nodes().size(), 1u);
  EXPECT_EQ(builder.LocalFunction("CastPair").Nodes()[0].op_type().value(), "Cast");
  EXPECT_EQ(report.patterns[0].matches, 1u);
  EXPECT_EQ(report.rewrites, 1u);

  core::builder::GraphBuilder replayed(original, SchemaLookup());
  RewritingProto serialized;
  ASSERT_TRUE(serialized.ParseFromString(plan.SerializeAsString()));
  const core::builder::GraphBuilder replayed_optimized = replayed.ApplyRewritings(serialized);
  ASSERT_EQ(replayed_optimized.LocalFunction("CastPair").Nodes().size(), 1u);
  EXPECT_EQ(replayed_optimized.LocalFunction("CastPair").Nodes()[0].op_type().value(), "Cast");

  auto invalid = serialized;
  invalid.ref_replaced_function_ids().clear();
  EXPECT_THROW(replayed.ApplyRewritings(invalid), core::builder::BuilderError);
  invalid = serialized;
  invalid.add_replaced_function_ids("test.pattern:Missing");
  EXPECT_THROW(replayed.ApplyRewritings(invalid), core::builder::BuilderError);
  invalid = serialized;
  invalid.add_replaced_function_ids("test.pattern:CastPair");
  EXPECT_THROW(replayed.ApplyRewritings(invalid), core::builder::BuilderError);
  invalid = serialized;
  invalid.ref_added_functions().clear();
  EXPECT_THROW(replayed.ApplyRewritings(invalid), core::builder::BuilderError);
  invalid = serialized;
  auto conflict = serialized.added_functions(0);
  conflict.ref_node()[0].set_op_type("Neg");
  invalid.ref_added_functions().push_back(std::move(conflict));
  EXPECT_THROW(replayed.ApplyRewritings(invalid), core::builder::BuilderError);

  const core::builder::GraphBuilder undone = builder.UndoRewritings(plan);
  ASSERT_EQ(undone.LocalFunction("CastPair").Nodes().size(), 2u);
}

TEST(PatternOptimization, NormalizesFunctionReplacementsInOneBatch) {
  std::size_t lookups = 0;
  const auto schemas = SchemaLookup();
  const auto lookup = [&](const std::string &name) {
    ++lookups;
    return schemas(name);
  };
  core::builder::GraphBuilder builder("g", lookup, {{"", 18}, {"test.pattern", 1}});
  constexpr std::size_t count = 8;
  for (std::size_t index = 0; index < count; ++index) {
    auto &function = builder.MakeLocalFunction("F" + std::to_string(index), "test.pattern");
    function.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
    function.MakeNode("Neg", {"X"}, {"middle"});
    function.MakeNode("Neg", {"middle"}, {"Y"});
    function.MakeOutput("Y");
  }
  const auto original = builder.ToModel();
  core::builder::GraphBuilder source(original, lookup);
  RewritingProto plan;
  for (const auto &function : original.functions()) {
    auto replacement = function;
    replacement.clr_node();
    replacement.add_node("Identity", {"X"}, {"Y"});
    auto &values = replacement.ref_value_info();
    for (auto it = values.begin(); it != values.end();) {
      if (it->name() == "middle")
        it = values.erase(it);
      else
        ++it;
    }
    plan.add_replaced_function_ids("test.pattern:" + function.name().value());
    plan.ref_added_functions().push_back(std::move(replacement));
  }
  lookups = 0;
  const auto result = source.ApplyRewritings(plan);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &function = result.LocalFunction("F" + std::to_string(index));
    ASSERT_EQ(function.Nodes().size(), 1u);
    EXPECT_EQ(function.Nodes()[0].op_type(), "Identity");
  }
  EXPECT_LE(lookups, 4 * count);
}

TEST(PatternOptimization, FunctionMatchersAllowCustomDomainOperatorNames) {
  for (bool secondary : {false, true}) {
    SCOPED_TRACE(secondary);
    core::builder::GraphBuilder definition(secondary ? "Relu" : "Cast", SchemaLookup(), {{"", 18}});
    definition.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
    definition.MakeNode(secondary ? "Neg" : "Relu", {"X"}, {"Y"});
    definition.MakeOutput("Y");
    FunctionProto override = definition.ToFunction("test.operators");
    if (!secondary)
      override.add_attribute("to");

    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"test.pattern", 1}});
    auto &function = builder.MakeLocalFunction("Wrapper", "test.pattern");
    function.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
    if (secondary) {
      function.MakeInitializer(MakeInitializer<float>("minimum", {}, {0.0f}));
      function.MakeNode("Relu", {"X"}, {"middle"});
      function.MakeNode("Clip", {"middle", "minimum"}, {"Y"});
    } else {
      AddCast(function, "X", "middle", TensorProto::FLOAT);
      AddCast(function, "middle", "Y", TensorProto::FLOAT);
    }
    function.MakeOutput("Y");
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
    builder.MakeNode("Wrapper", {"x"}, {"y"}, "test.pattern");
    builder.MakeOutput("y");
    ModelProto model = builder.ToModel();
    model.ref_functions().push_back(std::move(override));
    std::swap(model.ref_functions()[0], model.ref_functions()[1]);
    core::builder::GraphBuilder imported(model, SchemaLookup());
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    if (secondary)
      patterns.push_back(std::make_unique<onnx_patterns::ReluClipFusionPattern>());
    else
      patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
    core::builder::GraphGraph optimizer(imported, std::move(patterns));
    const RewritingProto plan = optimizer.Optimize();
    EXPECT_TRUE(plan.rewrites().empty());
    EXPECT_FALSE(plan.added_functions().empty());
    const auto &optimized_function = imported.LocalFunction("Wrapper");
    const auto &nodes = optimized_function.Nodes();
    ASSERT_EQ(nodes.size(), secondary ? 2u : 1u);
    EXPECT_EQ(nodes[nodes.size() - 1].op_type(), secondary ? "Clip" : "Identity");
  }
}

TEST(PatternOptimization, ConstantQueriesDoNotInterpretUnresolvedAttributeReferences) {
  for (const bool scalar_attribute : {false, true}) {
    SCOPED_TRACE(scalar_attribute);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
    utils::RepeatedProtoField<AttributeProto> attributes;
    auto &value = attributes.add();
    value.set_name(scalar_attribute ? "value_float" : "value");
    value.set_type(scalar_attribute ? AttributeProto::FLOAT : AttributeProto::TENSOR);
    if (scalar_attribute)
      value.set_f(1.0f);
    else
      *value.mutable_t() = MakeInitializer<float>("", {}, {1.0f});
    builder.MakeNode("Constant", {}, {"minimum"}, "", "", attributes);
    builder.MakeOutput("minimum");
    core::builder::GraphGraph graph(builder);
    EXPECT_TRUE(graph.IsConstantScalar("minimum", 1.0, false));
    EXPECT_EQ(graph.GetConstantScope("minimum") != nullptr, !scalar_attribute);
    auto &node = const_cast<NodeProto &>(builder.Nodes()[0]);
    node.mutable_attribute(0)->set_ref_attr_name("parameter");
    EXPECT_EQ(graph.GetComputedConstant("minimum"), nullptr);
    EXPECT_EQ(graph.GetConstantScope("minimum"), nullptr);
    EXPECT_FALSE(graph.IsConstantScalar("minimum"));
  }
}

TEST(PatternOptimization, ConstantFunctionOutputAllowsCustomDomainIdentity) {
  ScopedConstantKernel kernel;
  for (const std::string domain : {"local.test", "local.other"}) {
    SCOPED_TRACE(domain);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"test.pattern", 1}});
    auto &identity = builder.MakeLocalFunction("Identity", domain);
    identity.MakeInput("X", core::symbolic::TensorType::kFloat, Shape());
    identity.MakeNode("Neg", {"X"}, {"Y"});
    identity.MakeOutput("Y");
    auto &function = builder.MakeLocalFunction("Wrapper", "test.pattern");
    utils::RepeatedProtoField<AttributeProto> attributes;
    auto &value = attributes.add();
    value.set_name("value");
    value.set_type(AttributeProto::TENSOR);
    *value.mutable_t() = MakeInitializer<float>("", {2}, {1, 2});
    function.MakeNode("Constant", {}, {"Y"}, "", "", attributes);
    function.MakeOutput("Y");
    builder.MakeNode("Wrapper", {}, {"y"}, "test.pattern");
    builder.MakeOutput("y");
    const ModelProto original = builder.ToModel();
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::ConstantToInitializerPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    core::builder::ConstantFoldingOptions folding;
    folding.enabled = false;
    const auto plan = graph.Optimize(-1, nullptr, folding);
    EXPECT_FALSE(plan.added_functions().empty());
    EXPECT_FALSE(function.Initializers().empty());
    for (const auto &model : {original, builder.ToModel()}) {
      const auto outputs = core::runtime::RunModel(model, {});
      ASSERT_EQ(outputs.size(), 1u);
      ASSERT_EQ(outputs[0].element_count(), 2);
      EXPECT_FLOAT_EQ(outputs[0].AsFloat()[0], 1);
      EXPECT_FLOAT_EQ(outputs[0].AsFloat()[1], 2);
    }
  }
}

TEST(PatternOptimization, FunctionMatchersRejectUnresolvedAttributeReferences) {
  ModelProto model;
  model.add_opset("", 18);
  model.add_opset("test.pattern", 1);
  FunctionProto &function = *model.add_functions();
  function.set_name("TransposePair");
  function.set_domain("test.pattern");
  function.add_input("X");
  function.add_output("Y");
  function.add_attribute("perm");
  function.add_opset("", 18);
  for (const auto &[input, output] :
       {std::pair<const char *, const char *>{"X", "middle"}, {"middle", "Y"}}) {
    NodeProto &transpose = function.add_node("Transpose", {input}, {output});
    AttributeProto &perm = *transpose.add_attribute();
    perm.set_name("perm");
    perm.set_type(AttributeProto::AttributeType::INTS);
    perm.set_ref_attr_name("perm");
  }
  ValueInfoProto &input = *function.add_value_info();
  input.set_name("X");
  input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  for (int64_t dimension : {2, 3, 4}) {
    input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
        dimension);
  }
  model.mutable_graph()->set_name("g");
  NodeProto &call = model.mutable_graph()->add_node("TransposePair", {"x"}, {"y"}, "test.pattern");
  AttributeProto &permutation = *call.add_attribute();
  permutation.set_name("perm");
  permutation.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : {1, 2, 0}) {
    permutation.add_ints(value);
  }
  ValueInfoProto &graph_input = *model.mutable_graph()->add_input();
  graph_input.set_name("x");
  graph_input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  for (int64_t dimension : {2, 3, 4}) {
    graph_input.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
        dimension);
  }
  model.mutable_graph()->add_output()->set_name("y");
  core::builder::GraphBuilder builder(model, SchemaLookup());
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::TransposeTransposePattern>());
  core::builder::GraphGraph optimizer(builder, std::move(patterns));

  const RewritingProto plan = optimizer.Optimize();
  EXPECT_TRUE(plan.rewrites().empty());
  EXPECT_TRUE(plan.added_functions().empty());
  EXPECT_EQ(builder.LocalFunction("TransposePair").Nodes().size(), 2u);
}

TEST(PatternOptimization, ExportsAndReplaysHelpersCreatedInsideFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.SetOpsetVersion("test.pattern", 1);
  const auto integer_attributes =
      [](std::initializer_list<std::pair<std::string, int64_t>> values) {
        utils::RepeatedProtoField<AttributeProto> attributes;
        for (const auto &[name, value] : values) {
          AttributeProto &attribute = attributes.add();
          attribute.set_name(name);
          attribute.set_type(AttributeProto::AttributeType::INT);
          attribute.set_i(value);
        }
        return attributes;
      };
  for (const auto &name : {"Parent", "OtherParent"}) {
    auto &function = builder.MakeLocalFunction(name, "test.pattern");
    function.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
    function.MakeInput("cos", core::symbolic::TensorType::kFloat, Shape({6, 8}));
    function.MakeInput("sin", core::symbolic::TensorType::kFloat, Shape({6, 8}));
    function.MakeNode("Split", {"X"}, {"first", "second"}, "", "",
                      integer_attributes({{"axis", -1}, {"num_outputs", 2}}));
    function.MakeNode("Neg", {"second"}, {"negative"});
    function.MakeNode("Concat", {"negative", "first"}, {"rotated"}, "", "",
                      integer_attributes({{"axis", -1}}));
    function.MakeNode("Mul", {"rotated", "sin"}, {"rotated_sin"});
    function.MakeNode("Mul", {"X", "cos"}, {"scaled"});
    function.MakeNode("Add", {"rotated_sin", "scaled"}, {"Y"});
    function.MakeOutput("Y");
  }
  const ModelProto original = builder.ToModel();

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::FunctionHalfRotaryEmbeddingPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;
  const RewritingProto plan = graph.Optimize(-1, &report);

  ASSERT_EQ(report.rewrites, 2u);
  ASSERT_EQ(plan.added_functions().size(), 3u);
  EXPECT_EQ(plan.added_functions(0).name().value(), "HalfRotaryEmbedding");
  EXPECT_EQ(plan.added_functions(1).name().value(), "Parent");
  const ModelProto exported = builder.ToModel();
  ASSERT_EQ(exported.functions().size(), 3u);
  EXPECT_EQ(exported.functions(0).name().value(), "HalfRotaryEmbedding");
  EXPECT_EQ(exported.functions(1).name().value(), "Parent");
  ASSERT_EQ(exported.functions(1).node().size(), 1u);
  EXPECT_EQ(exported.functions(1).node(0).op_type().value(), "HalfRotaryEmbedding");
  EXPECT_EQ(exported.functions(2).name().value(), "OtherParent");
  ASSERT_EQ(exported.functions(2).node().size(), 1u);
  EXPECT_EQ(exported.functions(2).node(0).op_type().value(), "HalfRotaryEmbedding");
  EXPECT_FALSE(builder.LocalFunction("OtherParent").HasLocalFunction("HalfRotaryEmbedding"));

  core::builder::GraphBuilder replayed(original, SchemaLookup());
  const auto replayed_optimized = replayed.ApplyRewritings(plan).ToModel();
  ASSERT_EQ(replayed_optimized.functions().size(), 3u);
  EXPECT_EQ(replayed_optimized.functions(0).name().value(), "HalfRotaryEmbedding");
  EXPECT_EQ(replayed_optimized.functions(1).name().value(), "Parent");
  EXPECT_EQ(replayed_optimized.functions(1).node(0).op_type().value(), "HalfRotaryEmbedding");
  EXPECT_EQ(replayed_optimized.functions(2).node(0).op_type().value(), "HalfRotaryEmbedding");

  const auto undone = builder.UndoRewritings(plan);
  EXPECT_EQ(undone.LocalFunction("Parent").Nodes().size(), 6u);
  EXPECT_EQ(undone.LocalFunction("OtherParent").Nodes().size(), 6u);
  EXPECT_FALSE(undone.LocalFunction("Parent").HasLocalFunction("HalfRotaryEmbedding"));

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> repeated_patterns;
  repeated_patterns.push_back(
      std::make_unique<onnx_patterns::FunctionHalfRotaryEmbeddingPattern>());
  core::builder::GraphGraph repeated(builder, std::move(repeated_patterns));
  const RewritingProto repeated_plan = repeated.Optimize();
  EXPECT_TRUE(repeated_plan.added_functions().empty());
  const auto &helper = builder.LocalFunction("HalfRotaryEmbedding");
  EXPECT_EQ(helper.Nodes().size(), 6u);
  EXPECT_FALSE(helper.HasLocalFunction("HalfRotaryEmbedding"));

  for (const auto &[domain, overload] : {std::pair<std::string, std::string>{"intermediate", ""},
                                         {"other", ""},
                                         {"intermediate", "variant"}}) {
    for (const bool conflicting : {false, true}) {
      SCOPED_TRACE(domain);
      SCOPED_TRACE(overload);
      SCOPED_TRACE(conflicting);
      const bool same_identity = domain == "intermediate" && overload.empty();
      ModelProto with_helper = original;
      FunctionProto definition = exported.functions(0);
      definition.ref_value_info().clear();
      for (std::size_t index = 0; index < definition.input().size(); ++index) {
        const auto &parent = original.functions(0);
        const auto found = std::find_if(
            parent.value_info().begin(), parent.value_info().end(),
            [&](const auto &info) { return info.name().value() == parent.input(index).value(); });
        ASSERT_NE(found, parent.value_info().end());
        auto input = *found;
        input.set_name(definition.input(index));
        definition.ref_value_info().push_back(std::move(input));
      }
      definition.set_domain(domain);
      definition.set_overload(overload);
      if (conflicting)
        definition.ref_node().back().set_op_type("Sub");
      utils::RepeatedProtoField<FunctionProto> functions;
      functions.push_back(std::move(definition));
      functions.extend(std::move(with_helper.ref_functions()));
      with_helper.ref_functions() = std::move(functions);
      core::builder::GraphBuilder reused(with_helper, SchemaLookup());
      std::vector<std::unique_ptr<core::builder::PatternOptimization>> reuse_patterns;
      reuse_patterns.push_back(
          std::make_unique<onnx_patterns::FunctionHalfRotaryEmbeddingPattern>());
      core::builder::GraphGraph reuse_graph(reused, std::move(reuse_patterns));
      if (conflicting && same_identity) {
        EXPECT_THROW(reuse_graph.Optimize(), core::builder::BuilderError);
        EXPECT_EQ(reused.LocalFunction("Parent").Nodes().size(), 6u);
      } else {
        const auto reuse_plan = reuse_graph.Optimize();
        EXPECT_EQ(reuse_plan.added_functions().size(),
                  same_identity ? 2u : (conflicting ? 3u : 4u));
        const auto reused_model = reused.ToModel();
        EXPECT_EQ(reused_model.functions().size(), same_identity ? 3u : 4u);
        EXPECT_NO_THROW(core::builder::GraphBuilder(reused_model, SchemaLookup()));
      }
    }
  }
}

void CheckSharedHelperWhenUnusedBranchIsPruned(bool preexisting) {
  SCOPED_TRACE(preexisting);
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.SetOpsetVersion("other", 1);
  builder.SetOpsetVersion("test.pattern", 1);
  auto &name_collision = builder.MakeLocalFunction("HalfRotaryEmbedding", "other");
  name_collision.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
  name_collision.MakeNode("Identity", {"X"}, {"Y"});
  name_collision.MakeOutput("Y");
  auto &parent = builder.MakeLocalFunction("Parent", "test.pattern");
  parent.MakeInput("condition", core::symbolic::TensorType::kBool, Shape({}));
  parent.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
  parent.MakeInput("cos", core::symbolic::TensorType::kFloat, Shape({6, 8}));
  parent.MakeInput("sin", core::symbolic::TensorType::kFloat, Shape({6, 8}));

  const auto add_decomposition = [](core::builder::GraphBuilder &scope, const std::string &prefix,
                                    const std::string &output) {
    utils::RepeatedProtoField<AttributeProto> split_attributes;
    AttributeProto &num_outputs = split_attributes.add();
    num_outputs.set_name("num_outputs");
    num_outputs.set_type(AttributeProto::AttributeType::INT);
    num_outputs.set_i(2);
    AttributeProto &split_axis = split_attributes.add();
    split_axis.set_name("axis");
    split_axis.set_type(AttributeProto::AttributeType::INT);
    split_axis.set_i(-1);
    scope.MakeNode("Split", {"X"}, {prefix + "first", prefix + "second"}, "", "", split_attributes);
    scope.MakeNode("Neg", {prefix + "second"}, {prefix + "negative"});
    utils::RepeatedProtoField<AttributeProto> concat_attributes;
    AttributeProto &concat_axis = concat_attributes.add();
    concat_axis.set_name("axis");
    concat_axis.set_type(AttributeProto::AttributeType::INT);
    concat_axis.set_i(-1);
    scope.MakeNode("Concat", {prefix + "negative", prefix + "first"}, {prefix + "rotated"}, "", "",
                   concat_attributes);
    scope.MakeNode("Mul", {prefix + "rotated", "sin"}, {prefix + "rotated_sin"});
    scope.MakeNode("Mul", {"X", "cos"}, {prefix + "scaled"});
    scope.MakeNode("Add", {prefix + "rotated_sin", prefix + "scaled"}, {output});
  };

  auto &then_branch = parent.MakeSubgraph("then_branch");
  add_decomposition(then_branch, "branch_", "branch_output");
  then_branch.MakeOutput("branch_output");
  auto &else_branch = parent.MakeSubgraph("else_branch");
  else_branch.MakeNode("Identity", {"X"}, {"else_output"});
  else_branch.MakeOutput("else_output");
  if (preexisting) {
    auto &helper = then_branch.MakeLocalFunction("HalfRotaryEmbedding_0", "intermediate");
    helper.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
    helper.MakeInput("cos_cache", core::symbolic::TensorType::kFloat, Shape({6, 8}));
    helper.MakeInput("sin_cache", core::symbolic::TensorType::kFloat, Shape({6, 8}));
    auto split = MakeNode("Split", {"X"}, {"left", "right"});
    AddAttribute<int64_t>(split, "num_outputs", 2);
    AddAttribute<int64_t>(split, "axis", -1);
    helper.ImportNode(split);
    helper.MakeNode("Neg", {"right"}, {"right_neg"});
    auto concat = MakeNode("Concat", {"right_neg", "left"}, {"rotated"});
    AddAttribute<int64_t>(concat, "axis", -1);
    helper.ImportNode(concat);
    helper.MakeNode("Mul", {"X", "cos_cache"}, {"x_cos"});
    helper.MakeNode("Mul", {"rotated", "sin_cache"}, {"rotated_sin"});
    helper.MakeNode("Add", {"x_cos", "rotated_sin"}, {"Y"});
    helper.MakeOutput("Y");
    helper.set_device(core::symbolic::Device::kGPU0);
  }
  utils::RepeatedProtoField<AttributeProto> branch_attributes;
  for (const auto &[name, graph] :
       {std::pair<const char *, const char *>{"then_branch_ref", "then_branch"},
        {"else_branch_ref", "else_branch"}}) {
    AttributeProto &attribute = branch_attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::STRING);
    attribute.set_s(graph);
  }
  parent.MakeNode("If", {"condition"}, {"unused_if_output"}, "", "", branch_attributes);
  add_decomposition(parent, "", "Y");
  parent.MakeOutput("Y");

  builder.MakeInput("condition", core::symbolic::TensorType::kBool, Shape({}));
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 4, 6, 8}));
  builder.MakeInput("cos", core::symbolic::TensorType::kFloat, Shape({6, 8}));
  builder.MakeInput("sin", core::symbolic::TensorType::kFloat, Shape({6, 8}));
  builder.MakeNode("Parent", {"condition", "X", "cos", "sin"}, {"Y"}, "test.pattern");
  builder.MakeOutput("Y");

  const ModelProto original = builder.ToModel();
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::FunctionHalfRotaryEmbeddingPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  const RewritingProto plan = graph.Optimize();

  if (!preexisting) {
    EXPECT_TRUE(std::any_of(plan.added_functions().begin(), plan.added_functions().end(),
                            [](const auto &function) {
                              return function.name().value() == "HalfRotaryEmbedding_0" &&
                                     function.domain().value() == "intermediate";
                            }));
  }
  EXPECT_FALSE(std::any_of(plan.added_functions().begin(), plan.added_functions().end(),
                           [](const auto &function) {
                             return function.name().value() == "HalfRotaryEmbedding_1" &&
                                    function.domain().value() == "intermediate";
                           }));
  const ModelProto exported = builder.ToModel();
  if (preexisting) {
    ASSERT_TRUE(builder.HasLocalFunction("HalfRotaryEmbedding_0"));
    EXPECT_EQ(builder.LocalFunction("HalfRotaryEmbedding_0").device(),
              core::symbolic::Device::kGPU0);
  }
  ASSERT_EQ(exported.graph().node().size(), 1u);
  EXPECT_EQ(exported.graph().node(0).op_type().value(), "Parent");
  ASSERT_EQ(exported.functions().size(), 3u);
  const auto helper = std::find_if(exported.functions().begin(), exported.functions().end(),
                                   [](const auto &function) {
                                     return function.name().value() == "HalfRotaryEmbedding_0" &&
                                            function.domain().value() == "intermediate";
                                   });
  ASSERT_NE(helper, exported.functions().end());
  const auto exported_parent =
      std::find_if(exported.functions().begin(), exported.functions().end(),
                   [](const auto &function) { return function.name().value() == "Parent"; });
  ASSERT_NE(exported_parent, exported.functions().end());
  ASSERT_EQ(exported_parent->node().size(), 1u);
  EXPECT_EQ(exported_parent->node(0).op_type().value(), "HalfRotaryEmbedding_0");
  EXPECT_EQ(exported_parent->node(0).domain().value(), "intermediate");
  EXPECT_TRUE(std::any_of(exported.functions().begin(), exported.functions().end(),
                          [](const auto &function) {
                            return function.name().value() == "HalfRotaryEmbedding" &&
                                   function.domain().value() == "other";
                          }));

  const ModelProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).ToModel();
  ASSERT_EQ(replayed.graph().node().size(), 1u);
  EXPECT_EQ(replayed.graph().node(0).op_type().value(), "Parent");
  ASSERT_EQ(replayed.functions().size(), 3u);
  const auto replayed_parent =
      std::find_if(replayed.functions().begin(), replayed.functions().end(),
                   [](const auto &function) { return function.name().value() == "Parent"; });
  ASSERT_NE(replayed_parent, replayed.functions().end());
  ASSERT_EQ(replayed_parent->node().size(), 1u);
  EXPECT_EQ(replayed_parent->node(0).op_type().value(), "HalfRotaryEmbedding_0");
  EXPECT_EQ(replayed_parent->node(0).domain().value(), "intermediate");
  EXPECT_NO_THROW(core::builder::GraphBuilder(replayed, SchemaLookup()));
}

TEST(PatternOptimization, KeepsSharedHelperWhenUnusedBranchIsPruned) {
  CheckSharedHelperWhenUnusedBranchIsPruned(false);
}

TEST(PatternOptimization, KeepsPreexistingNestedHelperWhenUnusedBranchIsPruned) {
  CheckSharedHelperWhenUnusedBranchIsPruned(true);
}

TEST(PatternOptimization, SkipsFunctionPatternsWithUnresolvedAttributeReferences) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  auto &function = builder.MakeLocalFunction("Parameterized", "test.pattern");
  function.SetOpsetVersion("", 18);
  function.MakeInput("X", core::symbolic::TensorType::kFloat, Shape({2, 3, 4}));
  const auto reference_attribute = []() {
    utils::RepeatedProtoField<AttributeProto> attributes;
    AttributeProto &attribute = attributes.add();
    attribute.set_name("perm");
    attribute.set_ref_attr_name("perm");
    attribute.set_type(AttributeProto::AttributeType::INTS);
    return attributes;
  };
  function.MakeNode("Transpose", {"X"}, {"middle"}, "", "", reference_attribute());
  function.MakeNode("Transpose", {"middle"}, {"Y"}, "", "", reference_attribute());
  function.MakeOutput("Y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::TransposeTransposePattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  const RewritingProto plan = graph.Optimize();

  EXPECT_TRUE(plan.added_functions().empty());
  ASSERT_EQ(builder.LocalFunction("Parameterized").Nodes().size(), 2u);
  EXPECT_EQ(builder.LocalFunction("Parameterized").Nodes()[0].op_type().value(), "Transpose");
  EXPECT_EQ(builder.LocalFunction("Parameterized").Nodes()[1].op_type().value(), "Transpose");
}

TEST(PatternOptimization, SubgraphPatternsSeeCapturedParentInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion(kDefaultOnnxDomain, 21);
  builder.MakeInitializer(MakeInitializer<float>("a", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {2.0f}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"a"}, {"body_y"});
  body.MakeOutput("body_y");
  AddSubgraphReference(builder, "body", "body", "body_result");
  builder.MakeOutput("body_result");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReplaceIdentityWithConstantAddPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  const RewritingProto plan = graph.Optimize();
  const auto &rewrites = plan.rewrites();

  ASSERT_EQ(rewrites.size(), 1u);
  ASSERT_EQ(rewrites[0].graph_path().size(), 1u);
  EXPECT_EQ(rewrites[0].graph_path(0), "body");
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Add");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input()[0].value(), "a");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input()[1].value(), "b");

  const GraphProto optimized = builder.BuildGraph();
  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
}

TEST(PatternOptimization, OptimizeFoldsConstantReplacementAndReplaysInitializer) {
  ScopedAddKernel kernel;
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInitializer(MakeInitializer<float>("a", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {2.0f}));
  builder.MakeNode("Add", {"a", "b"}, {"untouched"});
  builder.MakeNode("Identity", {"a"}, {"y"});
  builder.MakeOutput("untouched");
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReplaceIdentityWithConstantAddPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;

  const RewritingProto plan = graph.Optimize(-1, &report);
  const auto &rewrites = plan.rewrites();
  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_TRUE(rewrites[0].added_nodes().empty());
  EXPECT_TRUE(rewrites[0].added_initializers().empty());
  ASSERT_EQ(rewrites[0].initializer_calls().size(), 1u);
  EXPECT_EQ(rewrites[0].initializer_calls(0).output(0), "y");
  EXPECT_EQ(rewrites[0].computed_initializer_positions().values(), (std::vector<uint64_t>{2}));
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "untouched");
  EXPECT_TRUE(graph.IsConstant("y"));
  const TensorProto *folded = graph.GetComputedConstant("y");
  ASSERT_NE(folded, nullptr);
  EXPECT_FLOAT_EQ(core::runtime::TensorFromProto(*folded).AsFloat()[0], 3.0f);
  EXPECT_GT(report.constant_folding_time_ns, 0);

  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.node_size(), 1);
  EXPECT_EQ(replayed.node(0).output()[0].value(), "untouched");
  ASSERT_EQ(replayed.initializer_size(), 3);
  EXPECT_EQ(replayed.initializer(2).name().value(), "y");
  EXPECT_FLOAT_EQ(core::runtime::TensorFromProto(replayed.initializer(2)).AsFloat()[0], 3.0f);
  EXPECT_EQ(builder.UndoRewritings(plan).BuildGraph().SerializeAsString(),
            original.graph().SerializeAsString());
  auto invalid = plan;
  invalid.ref_rewrites()[0].ref_initializer_calls()[0].ref_output()[0] = "mismatch";
  EXPECT_THROW(builder.UndoRewritings(invalid), core::builder::BuilderError);
}

TEST(PatternOptimization, OptimizeAppliesDisjointMatchesInOneIteration) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x1", core::symbolic::TensorType::kFloat16, Shape());
  builder.MakeInput("x2", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x1", "middle1", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle1", "y1", TensorProto::DataType::FLOAT);
  AddCast(builder, "x2", "middle2", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle2", "y2", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y1");
  builder.MakeOutput("y2");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const RewritingProto plan = graph.Optimize(1);
  const auto &rewrites = plan.rewrites();
  ASSERT_EQ(rewrites.size(), 2u);
  EXPECT_EQ(rewrites[0].added_nodes_positions().values(), std::vector<uint64_t>({0}));
  EXPECT_EQ(rewrites[1].added_nodes_positions().values(), std::vector<uint64_t>({1}));
  EXPECT_EQ(rewrites[1].matched_nodes().values(), std::vector<uint64_t>({1, 2}));
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x1");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "x2");

  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.node_size(), 2);
  EXPECT_EQ(replayed.node(0).input()[0].value(), "x1");
  EXPECT_EQ(replayed.node(0).output()[0].value(), "y1");
  EXPECT_EQ(replayed.node(1).input()[0].value(), "x2");
  EXPECT_EQ(replayed.node(1).output()[0].value(), "y2");
}

TEST(PatternOptimization, RecordsFinalPositionsForMultipleAddedNodes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReplaceIdentityWithTwoNodesPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  const RewritingProto plan = graph.Optimize(1);
  const auto &rewrites = plan.rewrites();

  ASSERT_EQ(rewrites.size(), 1u);
  ASSERT_EQ(rewrites[0].added_nodes().size(), 2u);
  EXPECT_EQ(rewrites[0].added_nodes_positions().values(), std::vector<uint64_t>({0, 1}));
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].name().value(), "replacement_neg");
  EXPECT_EQ(builder.Nodes()[1].name().value(), "replacement_relu");

  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.node_size(), 2);
  EXPECT_EQ(replayed.node(0).name().value(), "replacement_neg");
  EXPECT_EQ(replayed.node(1).name().value(), "replacement_relu");
}

TEST(PatternOptimization, ReplayRestoresAddedInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  AddOpsetImports(original, builder);

  RewritingProto plan;
  LocalRewritingProto &rewrite = *plan.add_rewrites();
  rewrite.set_pattern_name("CastCast");
  rewrite.add_matched_nodes(0);
  TensorProto &weight = *rewrite.add_added_initializers();
  weight.set_name("weight");
  weight.set_data_type(TensorProto::DataType::FLOAT);
  weight.add_dims(2);
  weight.add_float_data(1.0f);
  weight.add_float_data(2.0f);
  *rewrite.add_added_nodes() = MakeNode("Add", {"x", "weight"}, {"y"});
  rewrite.add_added_nodes_positions(0);
  rewrite.add_added_initializer_positions(0);

  const GraphProto replayed =
      core::builder::GraphBuilder(original, SchemaLookup()).ApplyRewritings(plan).BuildGraph();
  ASSERT_EQ(replayed.initializer_size(), 1);
  EXPECT_EQ(replayed.initializer(0).name().value(), "weight");
  ASSERT_EQ(replayed.node_size(), 1);
  EXPECT_EQ(replayed.node(0).op_type().value(), "Add");
  EXPECT_EQ(replayed.node(0).input()[1].value(), "weight");
}

TEST(PatternOptimization, ProtoRetainsNameAfterOptimizerDestruction) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  RewritingProto plan;
  {
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    plan = graph.Optimize();
  }

  ASSERT_EQ(plan.rewrites().size(), 1u);
  EXPECT_EQ(plan.rewrites(0).pattern_name(), "CastCast");
}

TEST(PatternOptimization, OptimizeKeepsSharedInnerCastInTopologicalOrder) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"middle"}, {"other"});
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  EXPECT_EQ(graph.Optimize().rewrites().size(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "middle");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "y");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Neg");
}

TEST(PatternOptimization, OptimizeHonorsDoNotRemovePredicate) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns), [](const NodeProto &node) {
    return node.output()[0].value() == "middle";
  });

  EXPECT_TRUE(graph.Optimize().rewrites().empty());
  EXPECT_EQ(builder.Nodes().size(), 2u);
}

TEST(PatternOptimization, RegistersBuiltInPatternsOnce) {
  onnx_patterns::RegisterPatterns();
  const std::vector<std::string> names = core::builder::RegisteredPatternNames();
  EXPECT_EQ(std::count(names.begin(), names.end(), "Cast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastCast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastCastBinary"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastOpCast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ClipClip"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConstantToInitializer"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConvAddFusion"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConvBatchNormalizationFusion"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConvBiasNull"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConvMulFusion"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "Dropout"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "Identity"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "InitializerUnsqueezeCast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "NotNot"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "PadConv"), 1);
  const std::vector<std::string> new_pattern_names = {
      "ConcatSliceElimination",
      "GatherSliceToSplit",
      "GatherToSlice",
      "LabelEncoderFusion",
      "SliceConcatToSpaceToDepth",
      "ConcatReshape",
      "Reshape",
      "ReduceReshape",
      "Reshape2Of3",
      "ReshapeReshapeBinary",
      "ReshapeReshape",
      "ReshapeSqueeze",
      "ShapeBasedEditDistanceReshape",
      "ShapeBasedReshapeIsSqueeze",
      "ShapedBasedReshape",
      "StaticConcatReshape",
      "UnsqueezeOrSqueezeReshape",
      "UnsqueezeReshape",
      "MulUnsqueezeUnsqueeze",
      "SqueezeAdd",
      "SqueezeBinaryUnsqueeze",
      "SwapUnsqueezeTranspose",
      "TransposeEqualReshape",
      "TransposeReshapeTranspose",
      "MulMulMulScalar",
      "SwitchOrderBinary",
      "SwapRangeAddScalar",
      "ReduceArgTopK",
      "ReduceSumNormalize",
      "Sub1Mul",
      "SwapUnary",
      "SameChildren",
      "SameChildrenFromInput",
      "ShapeBasedIdentity",
      "ShapeBasedSameChildren",
      "ShapeBasedShapeShapeAdd",
      "GemmSumFusion",
      "GemmTranspose",
      "GroupedMatMul",
      "MatMulAdd",
      "MatMulBatchNormalizationFusion",
      "MatMulReshape2Of3",
      "MatMulScaleFusion",
      "MulMulMatMul",
      "ReshapeMatMulReshape",
      "ShapeBasedMatMulToMul",
      "SwitchReshapeActivation",
      "TransposeMatMul",
      "TransposeReshapeMatMul",
      "BatchNormalization",
      "BatchNormalizationTraining",
      "CastLayerNormalizationCast",
      "LayerNormalization",
      "LayerNormalizationScale",
      "RMSNormalization",
      "RMSNormalizationMul",
      "Gelu",
      "LeakyRelu",
      "MaxRelu",
      "SoftmaxCrossEntropyLossCast",
      "RotaryEmbedding",
      "RotaryConcatPart",
      "FunctionCausalMask",
      "FunctionCausalMaskMulAdd",
      "FunctionCosSinCache",
      "FunctionHalfRotaryEmbedding",
      "FunctionAttention",
      "FunctionAttentionGQA",
      "AttentionGQA",
  };
  for (const std::string &name : new_pattern_names) {
    EXPECT_EQ(std::count(names.begin(), names.end(), name), 1) << name;
  }
  EXPECT_EQ(std::count(names.begin(), names.end(), "ShapeBasedExpandCastWhereSwap"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ShapeBasedExpandSwap"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ShapeBasedStaticExpand"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "SwapExpandReshape"), 1);

  onnx_patterns::RegisterPatterns();
  EXPECT_EQ(core::builder::RegisteredPatternNames(), names);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns =
      core::builder::CreateRegisteredPatterns();
  const bool found_cast = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::CastPattern *>(pattern.get()) != nullptr;
  });
  const bool found_cast_cast =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::CastCastPattern *>(pattern.get()) != nullptr;
      });
  const bool found_cast_cast_binary =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::CastCastBinaryPattern *>(pattern.get()) != nullptr;
      });
  const bool found_cast_op_cast =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::CastOpCastPattern *>(pattern.get()) != nullptr;
      });
  const bool found_clip_clip =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ClipClipPattern *>(pattern.get()) != nullptr;
      });
  const bool found_constant_to_initializer =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConstantToInitializerPattern *>(pattern.get()) !=
               nullptr;
      });
  const bool found_conv_bias_null =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConvBiasNullPattern *>(pattern.get()) != nullptr;
      });
  const bool found_conv_add_fusion =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConvAddFusionPattern *>(pattern.get()) != nullptr;
      });
  const bool found_conv_mul_fusion =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConvMulFusionPattern *>(pattern.get()) != nullptr;
      });
  const bool found_conv_bn_fusion =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConvBatchNormalizationFusionPattern *>(pattern.get()) !=
               nullptr;
      });
  const bool found_dropout = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::DropoutPattern *>(pattern.get()) != nullptr;
  });
  const bool found_identity =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::IdentityPattern *>(pattern.get()) != nullptr;
      });
  const bool found_initializer_unsqueeze_cast =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::InitializerUnsqueezeCastPattern *>(pattern.get()) !=
               nullptr;
      });
  const bool found_not_not = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::NotNotPattern *>(pattern.get()) != nullptr;
  });
  const bool found_pad_conv =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::PadConvPattern *>(pattern.get()) != nullptr;
      });
  EXPECT_TRUE(found_cast);
  EXPECT_TRUE(found_cast_cast);
  EXPECT_TRUE(found_cast_cast_binary);
  EXPECT_TRUE(found_cast_op_cast);
  EXPECT_TRUE(found_clip_clip);
  EXPECT_TRUE(found_constant_to_initializer);
  EXPECT_TRUE(found_conv_add_fusion);
  EXPECT_TRUE(found_conv_bn_fusion);
  EXPECT_TRUE(found_conv_bias_null);
  EXPECT_TRUE(found_conv_mul_fusion);
  EXPECT_TRUE(found_dropout);
  EXPECT_TRUE(found_identity);
  EXPECT_TRUE(found_initializer_unsqueeze_cast);
  EXPECT_TRUE(found_not_not);
  EXPECT_TRUE(found_pad_conv);
}

TEST(PatternOptimization, CreatesBuiltInPatternByName) {
  onnx_patterns::RegisterPatterns();

  std::unique_ptr<core::builder::PatternOptimization> cast = onnx_patterns::CreatePattern("Cast");
  ASSERT_NE(cast, nullptr);
  EXPECT_EQ(cast->Name(), "Cast");

  std::unique_ptr<core::builder::PatternOptimization> cast_with_priority =
      onnx_patterns::CreatePattern("Cast", 9);
  ASSERT_NE(cast_with_priority, nullptr);
  EXPECT_EQ(cast_with_priority->Name(), "Cast");
  EXPECT_EQ(cast_with_priority->priority, 9);

  std::unique_ptr<core::builder::PatternOptimization> initializer_unsqueeze_cast =
      onnx_patterns::CreatePattern("InitializerUnsqueezeCast");
  ASSERT_NE(initializer_unsqueeze_cast, nullptr);
  EXPECT_NE(dynamic_cast<onnx_patterns::InitializerUnsqueezeCastPattern *>(
                initializer_unsqueeze_cast.get()),
            nullptr);

  EXPECT_THROW(onnx_patterns::CreatePattern("TransposeToInitializer"),
               core::builder::PatternRegistrationError);

  EXPECT_THROW(onnx_patterns::CreatePattern("UnknownPattern"),
               core::builder::PatternRegistrationError);
}

} // namespace Test
