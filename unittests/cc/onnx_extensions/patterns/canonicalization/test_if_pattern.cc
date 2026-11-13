// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/patterns/canonicalization/if_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

using core::builder::GraphBuilder;
using core::builder::GraphGraph;
using core::runtime::Tensor;
using onnx_patterns::IfIdenticalBranchesPattern;

GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

ValueInfoProto Value(const std::string &name, int type = TensorProto::FLOAT, bool scalar = false) {
  ValueInfoProto value;
  value.set_name(name);
  auto &tensor = *value.mutable_type()->mutable_tensor_type();
  tensor.set_elem_type(type);
  auto &shape = *tensor.mutable_shape();
  if (!scalar)
    shape.add_dim()->set_dim_value(2);
  return value;
}

void GraphAttribute(NodeProto &node, const char *name, const GraphProto &graph) {
  auto &attribute = *node.add_attribute();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::GRAPH);
  *attribute.mutable_g() = graph;
}

GraphProto Branch(const std::string &suffix, bool multiple = false) {
  GraphProto graph;
  graph.set_name("branch" + suffix);
  graph.add_node(MakeNode("Neg", {"x"}, {"middle" + suffix}));
  graph.add_node(MakeNode("Neg", {"middle" + suffix}, {"result" + suffix}));
  *graph.add_output() = Value("result" + suffix);
  if (multiple)
    *graph.add_output() = Value("middle" + suffix);
  return graph;
}

NodeProto If(const GraphProto &first, const GraphProto &second,
             const std::string &condition = "condition",
             const std::vector<std::string> &outputs = {"y"}) {
  auto node = MakeNode("If", {condition}, outputs);
  GraphAttribute(node, "then_branch", first);
  GraphAttribute(node, "else_branch", second);
  return node;
}

ModelProto Model(const GraphProto &first, const GraphProto &second) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 18);
  auto &graph = *model.mutable_graph();
  graph.set_name("main");
  *graph.add_input() = Value("x");
  *graph.add_input() = Value("z");
  *graph.add_input() = Value("condition", TensorProto::BOOL, true);
  *graph.add_input() = Value("inner_condition", TensorProto::BOOL, true);
  std::vector<std::string> outputs;
  for (int i = 0; i < first.output_size(); ++i) {
    outputs.push_back("y" + std::to_string(i));
    auto info = first.output(i);
    info.set_name(outputs.back());
    *graph.add_output() = info;
  }
  graph.add_node(If(first, second, "condition", outputs));
  return model;
}

std::vector<std::vector<float>> Evaluate(const ModelProto &model, bool condition) {
  core::runtime::RuntimeContext runtime;
  runtime.RegisterCustomKernel("", "Neg", [](const NodeProto &node, auto &rt) {
    const auto &input = rt.Get(node.input(0));
    std::vector<float> result(input.element_count());
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = -input.AsFloat()[i];
    rt.Put(node.output(0), Tensor::From<float>(node.output(0), input.shape, result));
  });
  runtime.RegisterCustomKernel("", "Add", [](const NodeProto &node, auto &rt) {
    const auto &a = rt.Get(node.input(0));
    const auto &b = rt.Get(node.input(1));
    std::vector<float> result(a.element_count());
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = a.AsFloat()[i] + b.AsFloat()[i % b.element_count()];
    rt.Put(node.output(0), Tensor::From<float>(node.output(0), a.shape, result));
  });
  runtime.RegisterCustomKernel("", "Identity", [](const NodeProto &node, auto &rt) {
    auto value = rt.Get(node.input(0));
    value.name = node.output(0);
    rt.Put(node.output(0), std::move(value));
  });
  runtime.RegisterCustomKernel("", "Constant", [](const NodeProto &node, auto &rt) {
    auto value = core::runtime::TensorFromProto(FindAttribute(node, "value")->t());
    value.name = node.output(0);
    rt.Put(node.output(0), std::move(value));
  });
  runtime.Put("x", Tensor::From<float>("x", {2}, {1.5f, -3.0f}));
  runtime.Put("z", Tensor::From<float>("z", {2}, {-4.0f, 9.0f}));
  runtime.Put("condition", Tensor::FromBool("condition", {}, {condition}));
  runtime.Put("inner_condition", Tensor::FromBool("inner_condition", {}, {!condition}));
  core::runtime::RuntimeSession session(model);
  session.Run(runtime);
  std::vector<std::vector<float>> outputs;
  for (const auto &output : model.graph().output()) {
    const auto &value = runtime.Get(output.name());
    outputs.emplace_back(value.AsFloat(), value.AsFloat() + value.element_count());
  }
  return outputs;
}

ModelProto OptimizeAndCheck(const ModelProto &original) {
  GraphBuilder builder(original, SchemaLookup());
  const auto reference = builder.BuildGraph();
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<IfIdenticalBranchesPattern>());
  GraphGraph graph(builder, std::move(patterns));
  core::builder::ConstantFoldingOptions folding;
  folding.enabled = false;
  const auto plan = graph.Optimize(-1, nullptr, folding);
  EXPECT_FALSE(plan.rewrites().empty());
  const auto optimized = builder.ToModel(original.ir_version());
  EXPECT_NO_THROW(VerifyModel(optimized));
  RewritingProto loaded;
  EXPECT_TRUE(loaded.ParseFromString(plan.SerializeAsString()));
  const auto replayed = GraphBuilder(original, SchemaLookup()).ApplyRewritings(loaded).ToModel(10);
  EXPECT_EQ(replayed.graph().SerializeAsString(), optimized.graph().SerializeAsString());
  const auto restored = builder.UndoRewritings(loaded).BuildGraph();
  EXPECT_EQ(restored.SerializeAsString(), reference.SerializeAsString());
  for (bool condition : {false, true}) {
    EXPECT_EQ(Evaluate(original, condition), Evaluate(optimized, condition));
    EXPECT_EQ(Evaluate(original, condition), Evaluate(replayed, condition));
  }
  return optimized;
}

TEST(IfIdenticalBranchesPattern, AlphaEquivalentAndMultipleOutputs) {
  for (bool multiple : {false, true}) {
    const auto original = Model(Branch("_then", multiple), Branch("_else", multiple));
    const auto optimized = OptimizeAndCheck(original);
    ASSERT_EQ(optimized.graph().node_size(), 2);
    EXPECT_EQ(optimized.graph().node(0).op_type(), "Neg");
    EXPECT_EQ(optimized.graph().node(0).input(0), "x");
    EXPECT_EQ(optimized.graph().node(1).op_type(), "Neg");
    EXPECT_EQ(optimized.graph().output(0).name(), "y0");
    EXPECT_EQ(optimized.graph().node(1).output(0), "y0");
    if (multiple) {
      EXPECT_EQ(optimized.graph().node(0).output(0), "y1");
    }
  }
}

TEST(IfIdenticalBranchesPattern, MatchesNativeBuildersWithoutDeclaredOutputTypes) {
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput(Value("x"));
  builder.MakeInput(Value("condition", TensorProto::BOOL, true));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto *name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    const auto output = std::string(name) + "_result";
    branch.MakeNode("Neg", {"x"}, {output});
    branch.MakeOutput(output);
    auto &attribute = attributes.add();
    attribute.set_name(std::string(name) + "_ref");
    attribute.set_type(AttributeProto::STRING);
    attribute.set_s(name);
  }
  builder.MakeNode("If", {"condition"}, {"y"}, "", "", attributes);
  builder.MakeOutput("y");
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<IfIdenticalBranchesPattern>());
  GraphGraph graph(builder, std::move(patterns));
  EXPECT_NE(graph.Patterns()[0]->Match(graph, builder.Nodes()[0]).pattern, nullptr);
  graph.Optimize();
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].output(0), "y");
}

TEST(IfIdenticalBranchesPattern, ForwardsCapturedValueAndDuplicateOutputs) {
  GraphProto branch;
  branch.set_name("forward");
  *branch.add_output() = Value("x");
  *branch.add_output() = Value("x");
  const auto original = Model(branch, branch);
  GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput(Value("x"));
  builder.MakeInput(Value("condition", TensorProto::BOOL, true));
  GraphGraph graph(builder);
  IfIdenticalBranchesPattern pattern;
  const auto &candidate = original.graph().node(0);
  const auto match = pattern.Match(graph, candidate);
  ASSERT_EQ(match.pattern, &pattern);
  auto optimized = original;
  optimized.mutable_graph()->ref_node() = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(optimized.graph().node_size(), 2);
  for (const auto &node : optimized.graph().node()) {
    EXPECT_EQ(node.op_type(), "Identity");
    EXPECT_EQ(node.input(0), "x");
  }
  EXPECT_NO_THROW(VerifyModel(optimized));
  // The runtime requires local output producers, even for forwarded captures.
  auto reference = original;
  for (auto &attribute : reference.mutable_graph()->ref_node()[0].ref_attribute()) {
    auto &body = attribute.ref_g();
    for (int i = 0; i < body.output_size(); ++i) {
      const auto local = "forwarded_" + std::to_string(i);
      body.add_node(MakeNode("Identity", {"x"}, {local}));
      body.ref_output()[i].set_name(local);
    }
  }
  for (bool condition : {false, true})
    EXPECT_EQ(Evaluate(reference, condition), Evaluate(optimized, condition));
}

TEST(IfIdenticalBranchesPattern, PreservesOutputsCapturedByAnotherSubgraph) {
  auto original = Model(Branch("_then"), Branch("_else"));
  GraphProto forwarded;
  forwarded.set_name("forwarded");
  forwarded.add_node(MakeNode("Identity", {"y0"}, {"local_capture"}));
  *forwarded.add_output() = Value("local_capture");
  auto other = Branch("_other");
  original.mutable_graph()->add_node(If(forwarded, other, "inner_condition", {"observed"}));
  *original.mutable_graph()->add_output() = Value("observed");
  const auto optimized = OptimizeAndCheck(original);
  EXPECT_EQ(optimized.graph().output(0).name(), "y0");
  EXPECT_EQ(optimized.graph().output(1).name(), "observed");
  EXPECT_EQ(optimized.graph().node(1).output(0), "y0");
}

TEST(IfIdenticalBranchesPattern, AvoidsNamesDefinedInOtherSubgraphs) {
  auto original = Model(Branch("_then"), Branch("_else"));
  auto first = Branch("_observer"), second = Branch("_other");
  const std::string reserved = "IfIdenticalBranches_0_value_0";
  first.ref_node()[0].ref_output()[0] = reserved;
  first.ref_node()[1].ref_input()[0] = reserved;
  second.ref_node()[0].ref_input()[0] = "z";
  original.mutable_graph()->add_node(If(first, second, "inner_condition", {"observed"}));
  *original.mutable_graph()->add_output() = Value("observed");
  const auto optimized = OptimizeAndCheck(original);
  for (const auto &node : optimized.graph().node())
    for (const auto &output : node.output()) {
      EXPECT_NE(output, reserved);
    }
}

TEST(IfIdenticalBranchesPattern, InitializersAndConstantsCompareByValue) {
  for (bool constant : {false, true}) {
    auto first = Branch("_then");
    auto second = Branch("_else");
    for (auto *branch : {&first, &second}) {
      const auto name = branch == &first ? "a" : "b";
      auto tensor = MakeInitializer<float>(name, {2}, {2.0f, -5.0f});
      if (branch == &second) {
        const float values[] = {2.0f, -5.0f};
        tensor.clr_float_data();
        tensor.set_raw_data(values, sizeof(values));
      }
      if (constant) {
        auto node = MakeNode("Constant", {}, {name});
        auto &attribute = *node.add_attribute();
        attribute.set_name("value");
        attribute.set_type(AttributeProto::TENSOR);
        *attribute.mutable_t() = tensor;
        utils::RepeatedProtoField<NodeProto> nodes;
        nodes.push_back(std::move(node));
        nodes.extend(std::move(branch->ref_node()));
        branch->ref_node() = std::move(nodes);
      } else {
        *branch->add_initializer() = tensor;
      }
      auto &last = branch->ref_node().back();
      last.set_op_type("Add");
      last.add_input(name);
    }
    OptimizeAndCheck(Model(first, second));
    second.ref_node().back().ref_input()[1] = "x";
    GraphBuilder builder(Model(first, second), SchemaLookup());
    GraphGraph graph(builder);
    IfIdenticalBranchesPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  }
}

TEST(IfIdenticalBranchesPattern, ForwardsBranchLocalInitializer) {
  GraphProto first, second;
  for (auto *branch : {&first, &second}) {
    const auto name = branch == &first ? "first_weight" : "second_weight";
    branch->set_name(name);
    *branch->add_initializer() = MakeInitializer<float>(name, {2}, {2.0f, -5.0f});
    *branch->add_output() = Value(name);
  }
  const auto optimized = OptimizeAndCheck(Model(first, second));
  ASSERT_EQ(optimized.graph().node_size(), 1);
  EXPECT_EQ(optimized.graph().node(0).op_type(), "Constant");
  EXPECT_EQ(optimized.graph().node(0).output(0), "y0");
}

TEST(IfIdenticalBranchesPattern, PreservesNestedGraphAttributeNamesDuringInlining) {
  for (const std::string attribute_name : {"body", "body_ref__onnx_light_graph_reference_0"}) {
    GraphProto first, second;
    for (auto *branch : {&first, &second}) {
      const std::string suffix = branch == &first ? "_then" : "_else";
      GraphProto nested;
      nested.set_name("nested" + suffix);
      nested.add_node(MakeNode("Identity", {"x"}, {"nested_result" + suffix}));
      *nested.add_output() = Value("nested_result" + suffix);
      auto call = MakeNode("Custom", {"x"}, {"result" + suffix}, "custom");
      auto &attribute = *call.add_attribute();
      attribute.set_name(attribute_name);
      attribute.set_type(AttributeProto::GRAPH);
      *attribute.mutable_g() = nested;
      branch->add_node(call);
      *branch->add_output() = Value("result" + suffix);
    }

    auto original = Model(first, second);
    original.add_opset("custom", 1);
    FunctionProto function;
    function.set_name("Custom");
    function.set_domain("custom");
    function.add_input("arg");
    function.add_output("result");
    function.add_opset("", 18);
    function.add_node(MakeNode("Identity", {"arg"}, {"result"}));
    function.add_attribute(attribute_name);
    AttributeProto optional_reference;
    optional_reference.set_name("body_ref");
    optional_reference.set_type(AttributeProto::INT);
    optional_reference.set_i(1);
    *function.add_attribute_proto() = optional_reference;
    *original.add_functions() = function;

    GraphBuilder builder(original, SchemaLookup());
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<IfIdenticalBranchesPattern>());
    GraphGraph graph(builder, std::move(patterns));
    core::builder::ConstantFoldingOptions folding;
    folding.enabled = false;
    const auto plan = graph.Optimize(-1, nullptr, folding);
    ASSERT_FALSE(plan.rewrites().empty());

    const auto optimized = builder.ToModel(original.ir_version());
    ASSERT_EQ(optimized.graph().node_size(), 1);
    const auto *attribute = FindAttribute(optimized.graph().node(0), attribute_name.c_str());
    ASSERT_NE(attribute, nullptr);
    ASSERT_TRUE(attribute->has_g());
    EXPECT_EQ(attribute->name(), attribute_name);
    EXPECT_NO_THROW(VerifyModel(optimized));

    const auto round_trip = GraphBuilder(optimized, SchemaLookup()).ToModel();
    const auto *round_trip_attribute =
        FindAttribute(round_trip.graph().node(0), attribute_name.c_str());
    ASSERT_NE(round_trip_attribute, nullptr);
    EXPECT_TRUE(round_trip_attribute->has_g());
    EXPECT_EQ(round_trip_attribute->name(), attribute_name);
  }
}

GraphProto NestedBranch(const std::string &suffix, const char *op = "If") {
  GraphProto branch;
  branch.set_name("outer" + suffix);
  branch.add_node(MakeNode("Neg", {"x"}, {"local" + suffix}));
  auto inner = Branch(suffix);
  inner.ref_node()[0].ref_input()[0] = "local" + suffix;
  if (std::string(op) == "If") {
    auto other = inner;
    other.ref_node()[1].set_op_type("Identity");
    // The inner branches intentionally differ so only the outer If is removed.
    branch.add_node(If(inner, other, "inner_condition", {"nested" + suffix}));
  } else {
    inner.clr_node();
    inner.clr_output();
    if (std::string(op) == "Loop") {
      *inner.add_input() = Value("iteration" + suffix, TensorProto::INT64, true);
      *inner.add_input() = Value("keep" + suffix, TensorProto::BOOL, true);
      *inner.add_input() = Value("state" + suffix);
      inner.add_node(MakeNode("Identity", {"keep" + suffix}, {"continue" + suffix}));
      *inner.add_output() = Value("continue" + suffix, TensorProto::BOOL, true);
    } else {
      *inner.add_input() = Value("state" + suffix, TensorProto::FLOAT, true);
    }
    inner.add_node(MakeNode("Add", {"local" + suffix, "state" + suffix}, {"next" + suffix}));
    *inner.add_output() = Value("next" + suffix);
    auto node =
        MakeNode(op,
                 std::string(op) == "Loop" ? std::vector<std::string>{"trip", "condition", "x"}
                                           : std::vector<std::string>{"x"},
                 {"nested" + suffix});
    GraphAttribute(node, "body", inner);
    if (std::string(op) == "Scan") {
      auto &attribute = *node.add_attribute();
      attribute.set_name("num_scan_inputs");
      attribute.set_type(AttributeProto::INT);
      attribute.set_i(1);
    }
    branch.add_node(node);
  }
  *branch.add_output() = Value("nested" + suffix);
  if (std::string(op) == "Scan")
    branch.ref_output()[0]
        .mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->add_dim()
        ->set_dim_value(2);
  return branch;
}

TEST(IfIdenticalBranchesPattern, RejectsUnsupportedInitializerConstantsAtOlderOpsets) {
  GraphProto first, second;
  for (auto *branch : {&first, &second}) {
    const auto name = branch == &first ? "then_weight" : "else_weight";
    *branch->add_initializer() = MakeInitializer<int64_t>(name, {2}, {1, 2});
    *branch->add_output() = Value(name, TensorProto::INT64);
  }
  auto original = Model(first, second);
  original.ref_opset_import()[0].set_version(8);
  EXPECT_NO_THROW(VerifyModel(original));

  GraphBuilder builder(original, SchemaLookup());
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<IfIdenticalBranchesPattern>());
  GraphGraph graph(builder, std::move(patterns));
  core::builder::ConstantFoldingOptions folding;
  folding.enabled = false;
  EXPECT_TRUE(graph.Optimize(-1, nullptr, folding).rewrites().empty());
  const auto unchanged = builder.ToModel(original.ir_version());
  ASSERT_EQ(unchanged.graph().node_size(), 1);
  EXPECT_EQ(unchanged.graph().node(0).op_type(), "If");
  EXPECT_NO_THROW(VerifyModel(unchanged));
}

TEST(IfIdenticalBranchesPattern, NestedSubgraphsKeepLexicalCapturesAndReplay) {
  auto original = Model(NestedBranch("_then"), NestedBranch("_else"));
  const auto optimized = OptimizeAndCheck(original);
  ASSERT_EQ(optimized.graph().node_size(), 2);
  EXPECT_EQ(optimized.graph().node(0).op_type(), "Neg");
  EXPECT_EQ(optimized.graph().node(1).op_type(), "If");
  const auto *body = FindAttribute(optimized.graph().node(1), "then_branch");
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body->g().node(0).input(0), optimized.graph().node(0).output(0));
}

TEST(IfIdenticalBranchesPattern, RecursivelyComparesLoopAndScanBodies) {
  for (const auto *op : {"Loop", "Scan"}) {
    auto first = NestedBranch("_then", op), second = NestedBranch("_else", op);
    auto original = Model(first, second);
    *original.mutable_graph()->add_initializer() = MakeInitializer<int64_t>("trip", {}, {2});
    const auto optimized = OptimizeAndCheck(original);
    ASSERT_EQ(optimized.graph().node_size(), 2);
    EXPECT_EQ(optimized.graph().node(1).op_type(), op);
    second.ref_node()[1].ref_attribute()[0].ref_g().ref_node().back().ref_input()[0] = "z";
    auto different = Model(first, second);
    *different.mutable_graph()->add_initializer() = MakeInitializer<int64_t>("trip", {}, {2});
    GraphBuilder builder(different, SchemaLookup());
    GraphGraph graph(builder);
    IfIdenticalBranchesPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  }
}

TEST(IfIdenticalBranchesPattern, OptimizesIfInsideLoop) {
  auto original = Model(Branch("_then"), Branch("_else"));
  auto body = original.graph();
  body.set_name("loop_body");
  body.clr_input();
  *body.add_input() = Value("iteration", TensorProto::INT64, true);
  *body.add_input() = Value("keep", TensorProto::BOOL, true);
  *body.add_input() = Value("state");
  body.ref_node()[0].ref_input()[0] = "keep";
  auto first = Branch("_then"), second = Branch("_else");
  first.ref_node()[0].ref_input()[0] = "state";
  second.ref_node()[0].ref_input()[0] = "state";
  body.ref_node()[0] = If(first, second, "keep", {"next_state"});
  body.clr_output();
  body.add_node(MakeNode("Identity", {"keep"}, {"continue"}));
  *body.add_output() = Value("continue", TensorProto::BOOL, true);
  *body.add_output() = Value("next_state");
  auto &root = *original.mutable_graph();
  root.clr_node();
  auto loop = MakeNode("Loop", {"trip", "condition", "x"}, {"y0"});
  GraphAttribute(loop, "body", body);
  root.add_node(loop);
  *root.add_initializer() = MakeInitializer<int64_t>("trip", {}, {3});
  const auto optimized = OptimizeAndCheck(original);
  const auto &rewritten_body = FindAttribute(optimized.graph().node(0), "body")->g();
  for (const auto &node : rewritten_body.node())
    EXPECT_NE(node.op_type(), "If");
}

TEST(IfIdenticalBranchesPattern, RejectsDifferentCapturesAttributesConstantsAndOutputOrder) {
  for (int variation = 0; variation < 6; ++variation) {
    SCOPED_TRACE(variation);
    auto first = Branch("_then", true);
    auto second = Branch("_else", true);
    if (variation == 0)
      second.ref_node()[0].ref_input()[0] = "z";
    if (variation == 1) {
      first.ref_node()[0].set_op_type("LeakyRelu");
      second.ref_node()[0].set_op_type("LeakyRelu");
      for (auto *branch : {&first, &second}) {
        auto &attribute = *branch->ref_node()[0].add_attribute();
        attribute.set_name("alpha");
        attribute.set_type(AttributeProto::FLOAT);
        attribute.set_f(branch == &first ? 0.1f : 0.2f);
      }
    }
    if (variation == 2 || variation == 3) {
      for (auto *branch : {&first, &second}) {
        auto tensor = MakeInitializer<float>("weight", {2}, {branch == &first ? 1.0f : 2.0f, 3.0f});
        if (variation == 2) {
          *branch->add_initializer() = tensor;
          branch->ref_node()[0].ref_input()[0] = "weight";
        } else {
          auto &node = branch->ref_node()[0];
          node.set_op_type("Constant");
          node.clr_input();
          auto &attribute = *node.add_attribute();
          attribute.set_name("value");
          attribute.set_type(AttributeProto::TENSOR);
          *attribute.mutable_t() = tensor;
        }
      }
    }
    if (variation == 4)
      std::swap(second.ref_output()[0], second.ref_output()[1]);
    if (variation == 5)
      second.ref_node()[1].set_op_type("Identity");
    GraphBuilder builder(Model(first, second), SchemaLookup());
    GraphGraph graph(builder);
    IfIdenticalBranchesPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {&builder.Nodes()[0]}), core::builder::BuilderError);
  }
}

TEST(IfIdenticalBranchesPattern, RejectsDifferentOutputTypesAndOperatorIdentity) {
  for (int variation = 0; variation < 5; ++variation) {
    GraphBuilder builder("g", {}, {{"", 18}, {"custom", 1}, {"other", 1}});
    builder.MakeInput(Value("x"));
    builder.MakeInput(Value("condition", TensorProto::BOOL, true));
    auto first = Branch("_then"), second = Branch("_else");
    for (auto *branch : {&first, &second})
      for (auto &node : branch->ref_node())
        node.set_domain("custom");
    if (variation == 0)
      second.ref_output()[0] = Value("result_else", TensorProto::INT64);
    if (variation == 1)
      second.ref_node()[0].set_domain("other");
    if (variation == 2)
      second.ref_node()[0].set_overload("different");
    builder.MakeSubgraph("first", first);
    builder.MakeSubgraph("second", second);
    auto candidate = MakeNode("If", {"condition"}, {"y"});
    for (const auto &name : {"then_branch", "else_branch"}) {
      auto &attribute = *candidate.add_attribute();
      attribute.set_name(std::string(name) + "_ref");
      attribute.set_type(AttributeProto::STRING);
      attribute.set_s(std::string(name) == "then_branch" ? "first" : "second");
    }
    if (variation == 3)
      candidate.set_domain("custom");
    if (variation == 4)
      candidate.set_overload("custom");
    GraphGraph graph(builder);
    IfIdenticalBranchesPattern pattern;
    EXPECT_EQ(pattern.Match(graph, candidate).pattern, nullptr);
    EXPECT_THROW(pattern.Apply(graph, {}), core::builder::BuilderError);
    EXPECT_THROW(pattern.Apply(graph, {nullptr}), core::builder::BuilderError);
  }
}

TEST(IfIdenticalBranchesPattern, RespectsConfiguredNodesAndDoNotRemove) {
  for (int variation = 0; variation < 3; ++variation) {
    auto model = Model(Branch("_then"), Branch("_else"));
    if (variation == 0)
      model.mutable_graph()->ref_node()[0].add_device_configurations()->set_configuration_id("gpu");
    if (variation == 1)
      model.mutable_graph()
          ->ref_node()[0]
          .ref_attribute()[0]
          .ref_g()
          .ref_node()[0]
          .add_device_configurations()
          ->set_configuration_id("gpu");
    GraphBuilder builder(model, SchemaLookup());
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<IfIdenticalBranchesPattern>());
    GraphGraph graph(builder, std::move(patterns), [variation](const NodeProto &node) {
      return variation == 2 && node.op_type() == "If";
    });
    const auto plan = graph.Optimize();
    EXPECT_TRUE(plan.rewrites().empty());
    ASSERT_EQ(builder.Nodes().size(), 1u);
    EXPECT_EQ(builder.Nodes()[0].op_type(), "If");
  }
}

TEST(IfIdenticalBranchesPattern, RejectsConflictingBranchDeviceTargets) {
  GraphBuilder builder(Model(Branch("_then"), Branch("_else")), SchemaLookup());
  const auto name = FindAttribute(builder.Nodes()[0], "then_branch_ref")->s().value();
  builder.Subgraph(name).set_device(core::symbolic::Device::kCPU);
  GraphGraph graph(builder);
  IfIdenticalBranchesPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(IfIdenticalBranchesPattern, RejectsValueMetadataThatCannotBeInlined) {
  auto first = Branch("_then"), second = Branch("_else");
  for (auto *branch : {&first, &second}) {
    auto info = Value(branch == &first ? "middle_then" : "middle_else");
    auto &metadata = *info.add_metadata_props();
    metadata.set_key("onnx_light.value_tag");
    metadata.set_value("weight");
    *branch->add_value_info() = info;
  }
  GraphBuilder builder(Model(first, second), SchemaLookup());
  GraphGraph graph(builder);
  IfIdenticalBranchesPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(IfIdenticalBranchesPattern, RegistersBuiltInPattern) {
  onnx_patterns::RegisterPatterns();
  auto pattern = core::builder::CreateRegisteredPattern("IfIdenticalBranches");
  ASSERT_NE(pattern, nullptr);
  EXPECT_EQ(pattern->Name(), "IfIdenticalBranches");
  EXPECT_EQ(pattern->FastOpType(), std::set<std::string>{"If"});
}

} // namespace
} // namespace Test
