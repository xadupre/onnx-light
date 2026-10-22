// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "onnx_core/graph/layer.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_lib/checker.h"
#include "onnx_op/operator_sets_rt.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::Tensor;
using core::shapes::ShapesContext;
using core::symbolic::SymShape;
using core::symbolic::SymTensor;
using core::symbolic::TensorType;

NodeProto MakeLayer() {
  NodeProto node;
  node.set_domain("ai.rt");
  node.set_op_type("Layer");
  node.add_input("X");
  node.add_input("W");
  node.add_output("Y");
  node.add_output("Z");
  auto *attr = node.add_attribute();
  attr->set_name("body");
  attr->set_type(AttributeProto::GRAPH);
  GraphProto *body = attr->mutable_g();
  body->set_name("layer_body");
  body->add_input()->set_name("a");
  body->add_input()->set_name("w");
  body->add_output()->set_name("b");
  body->add_output()->set_name("w");
  NodeProto *inner = body->add_node();
  inner->set_domain("test.layer");
  inner->set_op_type("Add");
  inner->add_input("a");
  inner->add_input("w");
  inner->add_output("b");
  return node;
}

GraphProto &Body(NodeProto &node) { return node.mutable_attribute(0)->ref_g(); }

void SetTensorType(ValueInfoProto &value, int64_t dim) {
  auto *tensor = value.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(dim);
}

ShapesContext MakeShapes() {
  ShapesContext ctx;
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, SymShape({"batch", 4})));
  ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape({"batch", 4})));
  ctx.SetCustomShapeInferenceFunction("test.layer", "Add",
                                      [](ShapesContext &child, const NodeProto &node) {
                                        child.CopyValueFrom(node.output(0), child, node.input(0));
                                      });
  return ctx;
}

void RegisterAdd(RuntimeContext &rt, int &calls) {
  rt.RegisterCustomKernel(
      "test.layer", "Add", [&calls](const NodeProto &node, RuntimeContext &context) {
        ++calls;
        const size_t count = context.Get(node.input(0)).element_count();
        const auto a = context.Get(node.input(0)).AsFloat();
        const auto w = context.Get(node.input(1)).AsFloat();
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i)
          result[i] = a[i] + w[i];
        context.Set(node.output(0), Tensor::FromFloat(node.output(0), {int64_t(count)}, result));
      });
}

} // namespace

TEST(Layer, InfersActualSymbolicDescriptorsAndIsolatesFormals) {
  ShapesContext ctx = MakeShapes();
  NodeProto node = MakeLayer();
  for (auto &formal : Body(node).input()) {
    auto *type = formal.mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT);
    type->mutable_shape()->add_dim()->set_dim_param("formal_batch");
    type->mutable_shape()->add_dim()->set_dim_value(4);
  }
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("Y").Shape(), ctx.Get("X").Shape());
  EXPECT_EQ(ctx.Get("Z").Shape(), ctx.Get("W").Shape());
  EXPECT_FALSE(ctx.Has("a"));
  EXPECT_FALSE(ctx.Has("b"));
}

TEST(Layer, RunsPositionalBodyWithExplicitWeightsRepeatedly) {
  GraphProto graph;
  graph.add_input()->set_name("X");
  graph.add_input()->set_name("W");
  graph.add_output()->set_name("Y");
  graph.add_output()->set_name("Z");
  *graph.add_node() = MakeLayer();
  core::runtime::ExecutionPlan plan(graph);
  RuntimeSession session(plan);
  RuntimeContext rt;
  int calls = 0;
  RegisterAdd(rt, calls);
  rt.Set("a", Tensor::FromFloat("a", {1}, {99}));
  rt.Set("X", Tensor::FromFloat("X", {2}, {1, 2}));
  rt.Set("W", Tensor::FromFloat("W", {2}, {10, 20}));
  session.Run(rt);
  EXPECT_EQ(rt.Get("Y").AsFloat()[0], 11);
  EXPECT_EQ(rt.Get("Y").AsFloat()[1], 22);
  EXPECT_EQ(rt.Get("Z").AsFloat()[1], 20);
  EXPECT_EQ(rt.Get("a").AsFloat()[0], 99);
  EXPECT_FALSE(rt.Has("b"));
  rt.Put("X", Tensor::FromFloat("X", {2}, {3, 4}));
  session.Run(rt);
  EXPECT_EQ(rt.Get("Y").AsFloat()[0], 13);
  EXPECT_EQ(calls, 2);
}

TEST(Layer, RunsWithLocalInitializerAndReturnsOwnedForwardedInput) {
  NodeProto node = MakeLayer();
  node.clear_input();
  node.add_input("X");
  GraphProto &body = Body(node);
  body.clear_input();
  body.add_input()->set_name("a");
  TensorProto *weight = body.add_initializer();
  weight->set_name("w");
  weight->set_data_type(TensorProto::FLOAT);
  weight->add_dims(2);
  weight->add_float_data(10);
  weight->add_float_data(20);
  RuntimeContext rt;
  int calls = 0;
  RegisterAdd(rt, calls);
  rt.Set("X", Tensor::FromFloat("X", {2}, {1, 2}));
  core::runtime::RunNode(node, rt);
  EXPECT_EQ(rt.Get("Y").AsFloat()[1], 22);
  EXPECT_EQ(rt.Get("Z").AsFloat()[0], 10);
  EXPECT_FALSE(rt.Get("Z").is_borrowed());
  EXPECT_FALSE(rt.Has("w"));
  EXPECT_FALSE(rt.Has("a"));
}

TEST(Layer, PreservesPositionalOutputsWhenCallerAndBodyNamesCollide) {
  NodeProto node = MakeLayer();
  *node.mutable_output(0) = "w";
  *node.mutable_output(1) = "b";
  RuntimeContext rt;
  int calls = 0;
  RegisterAdd(rt, calls);
  rt.Set("X", Tensor::FromFloat("X", {1}, {1}));
  rt.Set("W", Tensor::FromFloat("W", {1}, {10}));
  core::runtime::RunNode(node, rt);
  EXPECT_EQ(rt.Get("w").AsFloat()[0], 11);
  EXPECT_EQ(rt.Get("b").AsFloat()[0], 10);
}

TEST(Layer, ExecutesZeroInputBodyAndDoesNotLeakUnusedInitializers) {
  NodeProto node;
  node.set_domain("ai.rt");
  node.set_op_type("Layer");
  node.add_output("result");
  auto *attr = node.add_attribute();
  attr->set_name("body");
  attr->set_type(AttributeProto::GRAPH);
  auto *body = attr->mutable_g();
  body->add_output()->set_name("constant");
  auto *initializer = body->add_initializer();
  initializer->set_name("constant");
  initializer->set_data_type(TensorProto::FLOAT);
  initializer->add_dims(1);
  initializer->add_float_data(42);
  const TensorProto constant = *initializer;
  *body->add_initializer() = constant;
  body->mutable_initializer(1)->set_name("unused");
  ShapesContext shapes;
  shapes.ComputeShapeNode(node);
  EXPECT_EQ(shapes.Get("result").Shape(), SymShape({1}));
  RuntimeContext rt;
  core::runtime::RunNode(node, rt);
  EXPECT_EQ(rt.Get("result").AsFloat()[0], 42);
  EXPECT_FALSE(rt.Has("constant"));
  EXPECT_FALSE(rt.Has("unused"));
}

TEST(Layer, ForwardsHeterogeneousTensorAndSequenceValues) {
  NodeProto node = MakeLayer();
  Body(node).clear_node();
  Body(node).mutable_output(0)->set_name("a");
  RuntimeContext rt;
  rt.Set("X", Tensor::FromInt64("X", {1}, {17}));
  core::runtime::Tensors elements;
  elements.push_back(Tensor::FromFloat("", {1}, {23}));
  rt.PutSequence("W", core::runtime::Sequence("W", TensorProto::FLOAT, std::move(elements)));
  core::runtime::RunNode(node, rt);
  EXPECT_EQ(rt.Get("Y").AsInt64()[0], 17);
  ASSERT_TRUE(rt.HasSequence("Z"));
  EXPECT_EQ(rt.GetSequence("Z").values[0].AsFloat()[0], 23);
  EXPECT_FALSE(rt.GetSequence("Z").values[0].is_borrowed());
}

TEST(Layer, RejectsMissingMalformedBodyAndBoundaryArity) {
  NodeProto missing = MakeLayer();
  missing.clear_attribute();
  EXPECT_THROW(core::graph::ValidateLayer(missing), std::invalid_argument);
  NodeProto malformed = MakeLayer();
  malformed.mutable_attribute(0)->set_type(AttributeProto::STRING);
  EXPECT_THROW(core::graph::ValidateLayer(malformed), std::invalid_argument);
  NodeProto no_graph = MakeLayer();
  no_graph.mutable_attribute(0)->clear_g();
  EXPECT_THROW(core::graph::ValidateLayer(no_graph), std::invalid_argument);
  NodeProto duplicate = MakeLayer();
  const AttributeProto copied_body = duplicate.attribute(0);
  *duplicate.add_attribute() = copied_body;
  EXPECT_THROW(core::graph::ValidateLayer(duplicate), std::invalid_argument);
  NodeProto arity = MakeLayer();
  arity.add_input("extra");
  EXPECT_THROW(core::graph::ValidateLayer(arity), std::invalid_argument);
  arity = MakeLayer();
  arity.add_output("extra");
  EXPECT_THROW(core::graph::ValidateLayer(arity), std::invalid_argument);
  NodeProto omitted = MakeLayer();
  *omitted.mutable_input(0) = "";
  EXPECT_THROW(core::graph::ValidateLayer(omitted), std::invalid_argument);
}

TEST(Layer, RejectsCapturesEvenWhenPresentInCaller) {
  NodeProto node = MakeLayer();
  *Body(node).mutable_node(0)->mutable_input(1) = "hidden";
  ShapesContext ctx = MakeShapes();
  ctx.Set("hidden", SymTensor(nullptr, TensorType::kFloat, SymShape({"batch", 4})));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  RuntimeContext rt;
  rt.Set("X", Tensor::FromFloat("X", {1}, {1}));
  rt.Set("W", Tensor::FromFloat("W", {1}, {2}));
  rt.Set("hidden", Tensor::FromFloat("hidden", {1}, {3}));
  EXPECT_THROW(core::runtime::RunNode(node, rt), std::invalid_argument);
}

TEST(Layer, RejectsRecursivelyNestedGraphCapturesAndUndefinedOutputs) {
  NodeProto node = MakeLayer();
  AttributeProto *attr = Body(node).mutable_node(0)->add_attribute();
  attr->set_name("nested");
  attr->set_type(AttributeProto::GRAPH);
  GraphProto *nested = attr->mutable_g();
  nested->add_output()->set_name("deep_result");
  NodeProto *inner = nested->add_node();
  inner->set_op_type("Identity");
  inner->add_input("hidden");
  inner->add_output("deep_result");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
  *inner->mutable_input(0) = "a";
  EXPECT_NO_THROW(core::graph::ValidateLayer(node));
  nested->mutable_output(0)->set_name("hidden_output");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
  node = MakeLayer();
  Body(node).mutable_output(0)->set_name("undefined");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
}

TEST(Layer, RejectsGraphListCapturesAndNestedLayersWithImplicitInputs) {
  NodeProto node = MakeLayer();
  AttributeProto *attr = Body(node).mutable_node(0)->add_attribute();
  attr->set_name("nested_list");
  attr->set_type(AttributeProto::GRAPHS);
  attr->add_graphs()->add_output()->set_name("hidden");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
  node = MakeLayer();
  NodeProto *nested = Body(node).add_node();
  *nested = MakeLayer();
  *nested->mutable_input(0) = "a";
  *nested->mutable_input(1) = "w";
  *nested->mutable_output(0) = "nested_Y";
  *nested->mutable_output(1) = "nested_Z";
  *Body(*nested).mutable_node(0)->mutable_input(0) = "a";
  // The nested Layer's "a" is declared; capturing its parent's "b" is not.
  EXPECT_NO_THROW(core::graph::ValidateLayer(node));
  *Body(*nested).mutable_node(0)->mutable_input(0) = "b";
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
}

TEST(Layer, RejectsDuplicateValuesAndForwardReferences) {
  NodeProto node = MakeLayer();
  Body(node).add_input()->set_name("a");
  node.add_input("X");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
  node = MakeLayer();
  *Body(node).mutable_node(0)->mutable_output(0) = "a";
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
  node = MakeLayer();
  *Body(node).mutable_node(0)->mutable_input(0) = "future";
  auto *later = Body(node).add_node();
  later->add_input("a");
  later->add_output("future");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
}

TEST(Layer, RejectsDirectCapturedGraphOutputs) {
  NodeProto node = MakeLayer();
  AttributeProto *attr = Body(node).mutable_node(0)->add_attribute();
  attr->set_name("nested");
  attr->set_type(AttributeProto::GRAPH);
  attr->mutable_g()->add_output()->set_name("a");
  EXPECT_THROW(core::graph::ValidateLayer(node), std::invalid_argument);
}

TEST(Layer, CallerInputsOverrideInitializerDefaultsIncludingNestedGraphs) {
  NodeProto node = MakeLayer();
  TensorProto *initializer = Body(node).add_initializer();
  initializer->set_name("a");
  initializer->set_data_type(TensorProto::FLOAT);
  initializer->add_dims(2);
  initializer->add_float_data(100);
  initializer->add_float_data(100);
  const TensorProto default_value = *initializer;
  ShapesContext ctx = MakeShapes();
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, SymShape({2})));
  ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape({2})));
  EXPECT_NO_THROW(ctx.ComputeShapeNode(node));
  EXPECT_EQ(ctx.Get("Y").Shape(), SymShape({2}));
  RuntimeContext rt;
  int calls = 0;
  RegisterAdd(rt, calls);
  rt.Set("X", Tensor::FromFloat("X", {2}, {1, 2}));
  rt.Set("W", Tensor::FromFloat("W", {2}, {10, 20}));
  core::runtime::RunNode(node, rt);
  EXPECT_EQ(rt.Get("Y").AsFloat()[0], 11);
  EXPECT_EQ(rt.Get("Y").AsFloat()[1], 22);
  AttributeProto *attr = Body(node).mutable_node(0)->add_attribute();
  attr->set_name("nested");
  attr->set_type(AttributeProto::GRAPH);
  GraphProto *nested = attr->mutable_g();
  nested->add_input()->set_name("a");
  nested->add_output()->set_name("a");
  *nested->add_initializer() = default_value;
  EXPECT_NO_THROW(core::graph::ValidateLayer(node));
}

TEST(Layer, RejectsIncompatibleBoundaryAnnotations) {
  NodeProto node = MakeLayer();
  SetTensorType(*Body(node).mutable_input(0), 3);
  ShapesContext ctx = MakeShapes();
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  RuntimeContext rt;
  rt.Set("X", Tensor::FromFloat("X", {2}, {1, 2}));
  rt.Set("W", Tensor::FromFloat("W", {2}, {10, 20}));
  EXPECT_THROW(core::runtime::RunNode(node, rt), std::invalid_argument);
  node = MakeLayer();
  SetTensorType(*Body(node).mutable_output(0), 3);
  ctx = MakeShapes();
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(Layer, RejectsInconsistentRepeatedSymbolicBoundaryDimensions) {
  NodeProto node = MakeLayer();
  for (auto &formal : Body(node).input()) {
    auto *type = formal.mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT);
    type->mutable_shape()->add_dim()->set_dim_param("N");
  }
  ShapesContext ctx = MakeShapes();
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, SymShape({2})));
  ctx.Set("W", SymTensor(nullptr, TensorType::kFloat, SymShape({3})));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  RuntimeContext rt;
  rt.Set("X", Tensor::FromFloat("X", {2}, {1, 2}));
  rt.Set("W", Tensor::FromFloat("W", {3}, {1, 2, 3}));
  EXPECT_THROW(core::runtime::RunNode(node, rt), std::invalid_argument);
}

TEST(Layer, SchemaAcceptsVariadicOutputsAndValidatesClosedBoundary) {
  const auto schemas = onnx_op::rt::GetAllOnnxOpRtSchemasWithHistory("Layer", true);
  ASSERT_EQ(schemas.size(), 1u);
  EXPECT_EQ(schemas[0].since_version(), 1);
  NodeProto node = MakeLayer();
  EXPECT_NO_THROW(schemas[0].Verify(node));
  *Body(node).mutable_node(0)->mutable_input(0) = "hidden";
  EXPECT_THROW(schemas[0].Verify(node), std::invalid_argument);
}

TEST(Layer, CheckerValidatesClosedBoundaryEvenWhenCustomDomainCheckingIsDisabled) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);
  opset = model.add_opset_import();
  opset->set_domain("ai.rt");
  opset->set_version(1);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("layer_checker");
  for (const auto &name : {"X", "W"}) {
    auto *input = graph->add_input();
    input->set_name(name);
    SetTensorType(*input, 1);
  }
  for (const auto &name : {"Y", "Z"}) {
    auto *output = graph->add_output();
    output->set_name(name);
    SetTensorType(*output, 1);
  }
  NodeProto *node = graph->add_node();
  *node = MakeLayer();
  for (auto &input : Body(*node).input())
    SetTensorType(input, 1);
  for (auto &output : Body(*node).output())
    SetTensorType(output, 1);
  NodeProto *inner = Body(*node).mutable_node(0);
  inner->set_domain("");
  inner->set_op_type("Identity");
  inner->clear_input();
  inner->add_input("a");
  EXPECT_NO_THROW(checker::check_model(model));
  EXPECT_NO_THROW(checker::check_model(model, false, false, true));
  *inner->mutable_input(0) = "X";
  EXPECT_THROW(checker::check_model(model), checker::ValidationError);
  EXPECT_THROW(checker::check_model(model, false, false, true), checker::ValidationError);
  *inner->mutable_input(0) = "a";
  node->clear_attribute();
  EXPECT_THROW(checker::check_model(model), checker::ValidationError);
}
