// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_runtime/local_function/include_local_function_cases.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr int64_t kIrVersion = 10;
constexpr int64_t kOnnxOpsetVersion = 18;

// Appends a node to ``func`` of type ``op_type``/``domain`` with the
// given input and output names.
void AddFuncNode(FunctionProto &func, const std::string &op_type, const std::string &domain,
                 const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
  NodeProto *n = func.add_node();
  n->set_op_type(op_type);
  if (!domain.empty()) {
    n->set_domain(domain);
  }
  for (const auto &in : inputs) {
    n->add_input(in);
  }
  for (const auto &out : outputs) {
    n->add_output(out);
  }
}

// Adds an opset import (domain, version) to ``model``.
void AddOpsetImport(ModelProto &model, const std::string &domain, int64_t version) {
  OperatorSetIdProto *os = model.add_opset_import();
  if (!domain.empty()) {
    os->set_domain(domain);
  }
  os->set_version(version);
}

} // namespace

// ---------------------------------------------------------------------------
// Cross-domain function-to-function dispatch
// ---------------------------------------------------------------------------
//
//   inner::Square(x)         -> Mul(x, x)
//   outer::SquareThenAdd(a, b) -> Add(inner::Square(a), b)
//
// The top-level graph applies ``outer::SquareThenAdd(x, y)``. Exercises that
// the model-local function registry built by ``RunModel`` is consulted by
// (domain, name, overload) from inside a callee's body.
// ---------------------------------------------------------------------------
void RegisterFunctionCallsFunctionAcrossDomainsCase(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_local_function_calls_function_across_domains";

  TestCase tc(name, name, "node", "local_function");
  ModelProto &model = tc.model;
  model.set_ir_version(kIrVersion);
  model.set_producer_name("backend-test");
  AddOpsetImport(model, "", kOnnxOpsetVersion);
  AddOpsetImport(model, "inner", 1);
  AddOpsetImport(model, "outer", 1);

  // inner::Square(x) -> y = Mul(x, x)
  FunctionProto *square = model.add_functions();
  square->set_name("Square");
  square->set_domain("inner");
  square->add_input("x");
  square->add_output("y");
  AddFuncNode(*square, "Mul", "", {"x", "x"}, {"y"});

  // outer::SquareThenAdd(a, b) -> a2 = inner::Square(a); z = Add(a2, b)
  FunctionProto *sqadd = model.add_functions();
  sqadd->set_name("SquareThenAdd");
  sqadd->set_domain("outer");
  sqadd->add_input("a");
  sqadd->add_input("b");
  sqadd->add_output("z");
  AddFuncNode(*sqadd, "Square", "inner", {"a"}, {"a2"});
  AddFuncNode(*sqadd, "Add", "", {"a2", "b"}, {"z"});

  // Top-level graph: z = outer::SquareThenAdd(x, y)
  GraphProto *g = model.add_graph();
  g->set_name(name);
  NodeProto *call = g->add_node();
  call->set_op_type("SquareThenAdd");
  call->set_domain("outer");
  call->add_input("x");
  call->add_input("y");
  call->add_output("z");

  const std::vector<int64_t> shape = {3};
  Tensor x = Tensor::FromFloat("x", shape, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("y", shape, {10.0f, 20.0f, 30.0f});
  Tensor z = Tensor::FromFloat("z", shape,
                               {1.0f * 1.0f + 10.0f, 2.0f * 2.0f + 20.0f, 3.0f * 3.0f + 30.0f});

  FillValueInfo(x, *g->add_input());
  FillValueInfo(y, *g->add_input());
  FillValueInfo(z, *g->add_output());

  AppendDataSet(tc, {x, y}, {z});
  registry.emplace_back(std::move(tc));
}

// ---------------------------------------------------------------------------
// Three-level nested model-local function calls (same domain)
// ---------------------------------------------------------------------------
//
//   Inner(x)  -> y = Add(x, x)         => 2*x
//   Middle(x) -> t = Inner(x); y = Inner(t)   => 4*x
//   Outer(x)  -> t = Inner(x); y = Middle(t)  => 8*x
//
// The top-level graph applies ``custom::Outer(x)``. Exercises that the
// function registry propagated to nested ``RuntimeContext`` instances
// remains complete at every nesting depth.
// ---------------------------------------------------------------------------
void RegisterFunctionThreeLevelNestedCallsCase(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_local_function_three_level_nested_calls";

  TestCase tc(name, name, "node", "local_function");
  ModelProto &model = tc.model;
  model.set_ir_version(kIrVersion);
  model.set_producer_name("backend-test");
  AddOpsetImport(model, "", kOnnxOpsetVersion);
  AddOpsetImport(model, "custom", 1);

  // Inner(x) -> y = Add(x, x)
  FunctionProto *inner = model.add_functions();
  inner->set_name("Inner");
  inner->set_domain("custom");
  inner->add_input("x");
  inner->add_output("y");
  AddFuncNode(*inner, "Add", "", {"x", "x"}, {"y"});

  // Middle(x) -> t = Inner(x); y = Inner(t)
  FunctionProto *middle = model.add_functions();
  middle->set_name("Middle");
  middle->set_domain("custom");
  middle->add_input("x");
  middle->add_output("y");
  AddFuncNode(*middle, "Inner", "custom", {"x"}, {"t"});
  AddFuncNode(*middle, "Inner", "custom", {"t"}, {"y"});

  // Outer(x) -> t = Inner(x); y = Middle(t)
  FunctionProto *outer = model.add_functions();
  outer->set_name("Outer");
  outer->set_domain("custom");
  outer->add_input("x");
  outer->add_output("y");
  AddFuncNode(*outer, "Inner", "custom", {"x"}, {"t"});
  AddFuncNode(*outer, "Middle", "custom", {"t"}, {"y"});

  // Top-level graph: y = custom::Outer(x)
  GraphProto *g = model.add_graph();
  g->set_name(name);
  NodeProto *call = g->add_node();
  call->set_op_type("Outer");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");

  const std::vector<int64_t> shape = {3};
  Tensor x = Tensor::FromFloat("x", shape, {1.0f, 2.5f, -3.0f});
  Tensor y = Tensor::FromFloat("y", shape, {8.0f, 20.0f, -24.0f});

  FillValueInfo(x, *g->add_input());
  FillValueInfo(y, *g->add_output());

  AppendDataSet(tc, {x}, {y});
  registry.emplace_back(std::move(tc));
}

// ---------------------------------------------------------------------------
// Model-local function with linked (``ref_attr_name``) attributes
// ---------------------------------------------------------------------------
//
//   Pick(cond) -> If(cond) { then_branch = ref(then_branch),
//                            else_branch = ref(else_branch) }
//
// The call-site supplies concrete branch sub-graphs as GRAPH attributes
// of the call node, and ``CallModelLocalFunction`` must resolve the
// ``ref_attr_name`` references in the function body to those sub-graphs
// before executing the ``If``. Each branch is a constant-only graph
// that emits a single FLOAT scalar; the test asserts the value selected
// by the condition matches the corresponding branch.
// ---------------------------------------------------------------------------
void RegisterFunctionLinkedAttributeCase(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_local_function_linked_attribute";

  TestCase tc(name, name, "node", "local_function");
  ModelProto &model = tc.model;
  model.set_ir_version(kIrVersion);
  model.set_producer_name("backend-test");
  AddOpsetImport(model, "", kOnnxOpsetVersion);
  AddOpsetImport(model, "custom", 1);

  // custom::Pick(cond) with formal attributes ``then_branch`` and
  // ``else_branch`` referenced from the body's If node.
  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  func->add_attribute("then_branch");
  func->add_attribute("else_branch");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  // Top-level graph: out = custom::Pick(cond) with concrete branches.
  GraphProto *g = model.add_graph();
  g->set_name(name);
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");

  auto fill_branch = [](GraphProto &gp, const std::string &branch_name,
                        const std::string &init_name, float value) {
    gp.set_name(branch_name);
    TensorProto *init = gp.add_initializer();
    init->set_name(init_name);
    init->set_data_type(TensorProto::DataType::FLOAT);
    init->add_float_data(value);
    NodeProto *add = gp.add_node();
    add->set_op_type("Add");
    add->add_input(init_name);
    add->add_input(init_name);
    add->add_output("z");
    gp.add_output()->set_name("z");
  };

  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  fill_branch(*tattr->add_g(), "then_g", "t", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  fill_branch(*eattr->add_g(), "else_g", "e", 1.0f);

  Tensor cond_true = Tensor::FromBool("cond", {}, {1});
  Tensor cond_false = Tensor::FromBool("cond", {}, {0});
  Tensor out_true = Tensor::FromFloat("out", {}, {20.0f});
  Tensor out_false = Tensor::FromFloat("out", {}, {2.0f});

  FillValueInfo(cond_true, *g->add_input());
  FillValueInfo(out_true, *g->add_output());

  AppendDataSet(tc, {cond_true}, {out_true});
  AppendDataSet(tc, {cond_false}, {out_false});
  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
