// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::KernelDispatchTable;
using onnx_kernels::RunFunction;
using onnx_kernels::RunGraph;
using onnx_kernels::RunModel;
using onnx_kernels::RunNode;
using onnx_kernels::RunNodes;
using onnx_kernels::RuntimeContext;
using onnx_kernels::Tensor;
using onnx_kernels::TensorFromProto;
using onnx_kernels::TensorMap;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

// Builds a single-node ``NodeProto`` of type ``op_type`` with the
// requested input and output names.
NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const std::string &domain = "") {
  NodeProto node;
  node.set_op_type(op_type);
  if (!domain.empty()) {
    node.set_domain(domain);
  }
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

TEST(RunNodes, DispatchTableContainsRegisteredOps) {
  const auto &table = KernelDispatchTable();
  // Spot-check the initial registered baseline of element-wise math ops.
  EXPECT_NE(table.find("ai.onnx:Add"), table.end());
  EXPECT_NE(table.find("ai.onnx:Sub"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mul"), table.end());
  EXPECT_NE(table.find("ai.onnx:Div"), table.end());
  EXPECT_NE(table.find("ai.onnx:Neg"), table.end());
  EXPECT_NE(table.find("ai.onnx:Abs"), table.end());

  // Spot-check the extended registration set covering the rest of the
  // unary / binary / variadic / attribute-driven math + logical
  // kernels (see ``onnx_kernels/kernel_dispatch_table.cc``).
  // Unary math, no attributes.
  EXPECT_NE(table.find("ai.onnx:Cos"), table.end());
  EXPECT_NE(table.find("ai.onnx:Erf"), table.end());
  EXPECT_NE(table.find("ai.onnx:Sigmoid"), table.end());
  EXPECT_NE(table.find("ai.onnx:Tanh"), table.end());
  // Binary math, no attributes.
  EXPECT_NE(table.find("ai.onnx:MatMul"), table.end());
  EXPECT_NE(table.find("ai.onnx:Pow"), table.end());
  // Variadic reducers.
  EXPECT_NE(table.find("ai.onnx:Sum"), table.end());
  EXPECT_NE(table.find("ai.onnx:Max"), table.end());
  EXPECT_NE(table.find("ai.onnx:Min"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mean"), table.end());
  // Reduction operators.
  EXPECT_NE(table.find("ai.onnx:ArgMax"), table.end());
  EXPECT_NE(table.find("ai.onnx:ArgMin"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceL1"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceL2"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceLogSum"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceLogSumExp"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMax"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMean"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMin"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceProd"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceSum"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceSumSquare"), table.end());
  // Attribute-driven kernels.
  EXPECT_NE(table.find("ai.onnx:Softmax"), table.end());
  EXPECT_NE(table.find("ai.onnx:LeakyRelu"), table.end());
  EXPECT_NE(table.find("ai.onnx:HardSigmoid"), table.end());
  EXPECT_NE(table.find("ai.onnx:Selu"), table.end());
  EXPECT_NE(table.find("ai.onnx:Gelu"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mod"), table.end());
  EXPECT_NE(table.find("ai.onnx:Clip"), table.end());
  EXPECT_NE(table.find("ai.onnx:IsInf"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitShift"), table.end());
  // Generator kernels.
  EXPECT_NE(table.find("ai.onnx:EyeLike"), table.end());
  // Logical / bitwise kernels.
  EXPECT_NE(table.find("ai.onnx:And"), table.end());
  EXPECT_NE(table.find("ai.onnx:Or"), table.end());
  EXPECT_NE(table.find("ai.onnx:Xor"), table.end());
  EXPECT_NE(table.find("ai.onnx:Not"), table.end());
  EXPECT_NE(table.find("ai.onnx:Equal"), table.end());
  EXPECT_NE(table.find("ai.onnx:Greater"), table.end());
  EXPECT_NE(table.find("ai.onnx:Less"), table.end());
  EXPECT_NE(table.find("ai.onnx:Where"), table.end());
  EXPECT_NE(table.find("ai.onnx:IsNaN"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitwiseAnd"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitwiseNot"), table.end());
}

TEST(RunNodes, RunNodeSingleAdd) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  RunNode(node, rt);
  ASSERT_NE(rt.tensors().find("z"), rt.tensors().end());
  const Tensor &z = rt.tensors()["z"];
  EXPECT_EQ(z.name, "z");
  EXPECT_EQ(z.shape, std::vector<int64_t>({3}));
  ASSERT_EQ(z.element_count(), 3);
  const float *got = z.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 11.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 33.0f);
}

TEST(RunNodes, RunNodeNormalisesDefaultDomain) {
  // The default ONNX domain is the empty string. The dispatcher must
  // normalise it to ``ai.onnx`` before looking up the kernel.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {-1.5f, 2.5f});
  NodeProto node = MakeNode("Abs", {"x"}, {"y"}); // empty domain
  EXPECT_TRUE(node.domain().as_string().empty());
  RunNode(node, rt);
  const float *got = rt.tensors()["y"].AsFloat();
  ASSERT_EQ(rt.tensors()["y"].element_count(), 2);
  EXPECT_FLOAT_EQ(got[0], 1.5f);
  EXPECT_FLOAT_EQ(got[1], 2.5f);
}

TEST(RunNodes, RunNodesOnRepeatedProtoFieldChain) {
  // Builds the small graph:  t = Mul(x, y);  out = Sub(t, z)
  // and runs it through the iterator overload that drives a
  // RepeatedProtoField<NodeProto> directly (mirroring how a caller
  // would feed ``graph.node()``).
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {2}, {3.0f, 4.0f});
  rt.tensors()["z"] = Tensor::FromFloat("z", {2}, {0.5f, 0.25f});

  utils::RepeatedProtoField<NodeProto> nodes;
  *nodes.Add() = MakeNode("Mul", {"x", "y"}, {"t"});
  *nodes.Add() = MakeNode("Sub", {"t", "z"}, {"out"});

  RunNodes(nodes, rt);

  ASSERT_NE(rt.tensors().find("t"), rt.tensors().end());
  ASSERT_NE(rt.tensors().find("out"), rt.tensors().end());
  const float *out = rt.tensors()["out"].AsFloat();
  ASSERT_EQ(rt.tensors()["out"].element_count(), 2);
  EXPECT_FLOAT_EQ(out[0], 1.0f * 3.0f - 0.5f);
  EXPECT_FLOAT_EQ(out[1], 2.0f * 4.0f - 0.25f);
  EXPECT_EQ(rt.tensors()["out"].name, "out");
}

TEST(RunNodes, RunNodesOnIteratorRangeFromVector) {
  // Same graph, but driven through the generic iterator overload so
  // any container whose elements dereference to ``NodeProto`` works.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {1}, {6.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {1}, {2.0f});

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Div", {"a", "b"}, {"q"})); // q = 3
  nodes.push_back(MakeNode("Neg", {"q"}, {"out"}));    // out = -3

  RunNodes(nodes.begin(), nodes.end(), rt);

  const float *out = rt.tensors()["out"].AsFloat();
  ASSERT_EQ(rt.tensors()["out"].element_count(), 1);
  EXPECT_FLOAT_EQ(out[0], -3.0f);
}

TEST(RunNodes, RunNodeEyeLikeUsesAttributes) {
  // Verify the EyeLike trampoline forwards both the ``k`` and ``dtype``
  // attributes to ``kernel::EyeLike``. Input shape is copied (2x3) and
  // the output uses dtype=INT64 (=7) with ones on the super-diagonal.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {0, 0, 0, 0, 0, 0});
  NodeProto node = MakeNode("EyeLike", {"x"}, {"y"});
  AttributeProto *attr_k = node.add_attribute();
  attr_k->set_name("k");
  attr_k->set_type(AttributeProto::AttributeType::INT);
  attr_k->set_i(1);
  AttributeProto *attr_dtype = node.add_attribute();
  attr_dtype->set_name("dtype");
  attr_dtype->set_type(AttributeProto::AttributeType::INT);
  attr_dtype->set_i(7); // INT64
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, std::vector<int64_t>({2, 3}));
  EXPECT_EQ(y.data_type, 7);
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 0);
  EXPECT_EQ(got[1], 1);
  EXPECT_EQ(got[2], 0);
  EXPECT_EQ(got[3], 0);
  EXPECT_EQ(got[4], 0);
  EXPECT_EQ(got[5], 1);
}

TEST(RunNodes, RunNodeUnsupportedOpTypeThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  NodeProto node = MakeNode("ThisOpDoesNotExist", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeMissingInputThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18))); // empty: "x" is not present
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeWrongInputCountThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  // Add expects two inputs but we only declare one.
  NodeProto node = MakeNode("Add", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RuntimeContextSetGetHas) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_FALSE(rt.Has("x"));
  rt.Set("x", Tensor::FromFloat("x", {1}, {7.0f}));
  EXPECT_TRUE(rt.Has("x"));
  EXPECT_FLOAT_EQ(rt.Get("x").AsFloat()[0], 7.0f);
  EXPECT_THROW(rt.Get("missing"), std::out_of_range);
}

TEST(RunNodes, RuntimeContextRemove) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {7.0f}));
  EXPECT_TRUE(rt.Remove("x"));
  EXPECT_FALSE(rt.Has("x"));
  EXPECT_THROW(rt.Get("x"), std::out_of_range);
  EXPECT_FALSE(rt.Remove("x"));
}

// ---------------------------------------------------------------------------
// TensorFromProto tests
// ---------------------------------------------------------------------------

TEST(TensorFromProto, FloatTypedField) {
  TensorProto tp;
  tp.set_name("w");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(1.0f);
  tp.add_float_data(2.0f);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.name, "w");
  EXPECT_EQ(t.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 2.0f);
}

TEST(TensorFromProto, FloatRawData) {
  TensorProto tp;
  tp.set_name("r");
  tp.ref_dims().push_back(3);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  const float vals[] = {3.0f, 4.0f, 5.0f};
  const auto *raw_ptr = reinterpret_cast<const uint8_t *>(vals);
  tp.ref_raw_data() = std::vector<uint8_t>(raw_ptr, raw_ptr + sizeof(vals));

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.name, "r");
  ASSERT_EQ(t.element_count(), 3);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[2], 5.0f);
}

TEST(TensorFromProto, Int64TypedField) {
  TensorProto tp;
  tp.set_name("i64");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT64);
  tp.add_int64_data(100);
  tp.add_int64_data(-200);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_EQ(t.AsInt64()[0], 100);
  EXPECT_EQ(t.AsInt64()[1], -200);
}

TEST(TensorFromProto, Int32TypedField) {
  TensorProto tp;
  tp.set_name("i32");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT32);
  tp.add_int32_data(7);
  tp.add_int32_data(-3);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_EQ(t.AsInt32()[0], 7);
  EXPECT_EQ(t.AsInt32()[1], -3);
}

TEST(TensorFromProto, ScalarNoShape) {
  // Scalar TensorProto has no dims entry.
  TensorProto tp;
  tp.set_name("s");
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(42.0f);

  Tensor t = TensorFromProto(tp);
  EXPECT_TRUE(t.shape.empty());
  EXPECT_EQ(t.element_count(), 1);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 42.0f);
}

// ---------------------------------------------------------------------------
// RunGraph tests
// ---------------------------------------------------------------------------

TEST(RunGraph, InitializersLoadedAndNodesRun) {
  // Graph:  out = Add(x_input, w_init)
  //   x_input is provided by the caller; w_init comes from the initializer.
  TensorProto init_tp;
  init_tp.set_name("w_init");
  init_tp.ref_dims().push_back(2);
  init_tp.set_data_type(TensorProto::DataType::FLOAT);
  init_tp.add_float_data(10.0f);
  init_tp.add_float_data(20.0f);

  GraphProto graph;
  *graph.add_initializer() = init_tp;
  NodeProto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("x_input");
  node->add_input("w_init");
  node->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x_input", Tensor::FromFloat("x_input", {2}, {1.0f, 2.0f}));

  RunGraph(graph, rt);

  ASSERT_TRUE(rt.Has("out"));
  const float *res = rt.Get("out").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 11.0f);
  EXPECT_FLOAT_EQ(res[1], 22.0f);
}

TEST(RunGraph, CallerInputOverridesInitializer) {
  // When the caller has already seeded a name that is also an initializer,
  // the caller's value must win.
  TensorProto init_tp;
  init_tp.set_name("w");
  init_tp.ref_dims().push_back(1);
  init_tp.set_data_type(TensorProto::DataType::FLOAT);
  init_tp.add_float_data(999.0f); // should be ignored

  GraphProto graph;
  *graph.add_initializer() = init_tp;
  NodeProto *node = graph.add_node();
  node->set_op_type("Abs");
  node->add_input("w");
  node->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  // Override the initializer with the caller's value.
  rt.Set("w", Tensor::FromFloat("w", {1}, {-5.0f}));

  RunGraph(graph, rt);

  ASSERT_TRUE(rt.Has("out"));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 5.0f);
}

// ---------------------------------------------------------------------------
// RunFunction tests
// ---------------------------------------------------------------------------

TEST(RunFunction, NodesRun) {
  FunctionProto func;
  func.set_name("f");
  func.add_input("a");
  func.add_input("b");
  func.add_output("result");
  NodeProto *node = func.add_node();
  node->set_op_type("Mul");
  node->add_input("a");
  node->add_input("b");
  node->add_output("result");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("a", Tensor::FromFloat("a", {2}, {3.0f, 4.0f}));
  rt.Set("b", Tensor::FromFloat("b", {2}, {2.0f, 5.0f}));

  RunFunction(func, rt);

  ASSERT_TRUE(rt.Has("result"));
  const float *res = rt.Get("result").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 6.0f);
  EXPECT_FLOAT_EQ(res[1], 20.0f);
}

// ---------------------------------------------------------------------------
// RunModel tests
// ---------------------------------------------------------------------------

TEST(RunModel, GraphRun) {
  // Build a minimal ModelProto with a single Add node.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Add");
  node->add_input("p");
  node->add_input("q");
  node->add_output("r");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("p", Tensor::FromFloat("p", {3}, {1.0f, 2.0f, 3.0f}));
  rt.Set("q", Tensor::FromFloat("q", {3}, {4.0f, 5.0f, 6.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("r"));
  const float *res = rt.Get("r").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 5.0f);
  EXPECT_FLOAT_EQ(res[1], 7.0f);
  EXPECT_FLOAT_EQ(res[2], 9.0f);
}

TEST(RunModel, NoGraphThrows) {
  ModelProto model;
  model.set_ir_version(10);
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(RunModel(model, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Model-local function dispatch tests
// ---------------------------------------------------------------------------

TEST(RunModel, NodeDispatchedToModelLocalFunction) {
  // Define a model-local function "MyAddMul" in domain "custom":
  //   inputs:  a, b, c
  //   output:  out  = (a + b) * c
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("MyAddMul");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b");
  func->add_input("c");
  func->add_output("out");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Add");
    n->add_input("a");
    n->add_input("b");
    n->add_output("tmp");
  }
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Mul");
    n->add_input("tmp");
    n->add_input("c");
    n->add_output("out");
  }

  // Main graph: y = MyAddMul(x, w, k) where (x, w, k) are graph inputs.
  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("MyAddMul");
  node->set_domain("custom");
  node->add_input("x");
  node->add_input("w");
  node->add_input("k");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  rt.Set("w", Tensor::FromFloat("w", {2}, {3.0f, 4.0f}));
  rt.Set("k", Tensor::FromFloat("k", {2}, {10.0f, 100.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 2);
  EXPECT_FLOAT_EQ(res[0], (1.0f + 3.0f) * 10.0f);
  EXPECT_FLOAT_EQ(res[1], (2.0f + 4.0f) * 100.0f);
  // The function's internal "tmp" value must NOT leak into the caller's
  // tensor map: the child context is isolated.
  EXPECT_FALSE(rt.Has("tmp"));
}

TEST(RunModel, ModelLocalFunctionCanCallAnotherFunction) {
  // Define two model-local functions in "custom":
  //   AddOne(x) -> Add(x, one_init_passed_as_input)  -- here we use x+x
  //   Quad(x)   -> AddOne(AddOne(x))                 -- so result = x*4? No: just chained calls
  // To keep it simple and only exercise nesting, use:
  //   Twice(x)  -> Add(x, x)
  //   Quad(x)   -> Twice(Twice(x))   => 4*x
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *twice = model.add_functions();
  twice->set_name("Twice");
  twice->set_domain("custom");
  twice->add_input("x");
  twice->add_output("y");
  {
    NodeProto *n = twice->add_node();
    n->set_op_type("Add");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *quad = model.add_functions();
  quad->set_name("Quad");
  quad->set_domain("custom");
  quad->add_input("x");
  quad->add_output("y");
  {
    NodeProto *n = quad->add_node();
    n->set_op_type("Twice");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = quad->add_node();
    n->set_op_type("Twice");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Quad");
  node->set_domain("custom");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 4.0f);
  EXPECT_FLOAT_EQ(res[1], 10.0f);
  EXPECT_FLOAT_EQ(res[2], -12.0f);
}

TEST(RunModel, ModelLocalFunctionCallsAnotherFunctionAcrossDomains) {
  // Define two model-local functions in different domains; the function
  // in domain "outer" calls into the function in domain "inner". This
  // exercises that the function registry propagated to the child
  // RuntimeContext is keyed by (domain, name, overload) and that
  // cross-domain function-to-function dispatch works.
  //
  //   inner::Square(x) -> Mul(x, x)
  //   outer::SquareThenAdd(x, y) -> Add(inner::Square(x), y)
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *square = model.add_functions();
  square->set_name("Square");
  square->set_domain("inner");
  square->add_input("x");
  square->add_output("y");
  {
    NodeProto *n = square->add_node();
    n->set_op_type("Mul");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *sqadd = model.add_functions();
  sqadd->set_name("SquareThenAdd");
  sqadd->set_domain("outer");
  sqadd->add_input("a");
  sqadd->add_input("b");
  sqadd->add_output("z");
  {
    NodeProto *n = sqadd->add_node();
    n->set_op_type("Square");
    n->set_domain("inner");
    n->add_input("a");
    n->add_output("a2");
  }
  {
    NodeProto *n = sqadd->add_node();
    n->set_op_type("Add");
    n->add_input("a2");
    n->add_input("b");
    n->add_output("z");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("SquareThenAdd");
  node->set_domain("outer");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  rt.Set("y", Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("z"));
  const float *res = rt.Get("z").AsFloat();
  ASSERT_EQ(rt.Get("z").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 1.0f * 1.0f + 10.0f);
  EXPECT_FLOAT_EQ(res[1], 2.0f * 2.0f + 20.0f);
  EXPECT_FLOAT_EQ(res[2], 3.0f * 3.0f + 30.0f);
  // The intermediate name produced inside the SquareThenAdd function
  // body must not leak into the caller's tensor map.
  EXPECT_FALSE(rt.Has("a2"));
}

TEST(RunModel, ModelLocalFunctionThreeLevelNestedCalls) {
  // Demonstrates that the function registry is propagated through
  // arbitrary nesting depth: the top-level graph calls Outer, Outer
  // calls Middle, and Middle calls Inner. Each level is a separate
  // FunctionProto in the model.
  //
  //   Inner(x)  -> Add(x, x)        => 2*x
  //   Middle(x) -> Inner(Inner(x))  => 4*x
  //   Outer(x)  -> Middle(Inner(x)) => 8*x
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *inner = model.add_functions();
  inner->set_name("Inner");
  inner->set_domain("custom");
  inner->add_input("x");
  inner->add_output("y");
  {
    NodeProto *n = inner->add_node();
    n->set_op_type("Add");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *middle = model.add_functions();
  middle->set_name("Middle");
  middle->set_domain("custom");
  middle->add_input("x");
  middle->add_output("y");
  {
    NodeProto *n = middle->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = middle->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  FunctionProto *outer = model.add_functions();
  outer->set_name("Outer");
  outer->set_domain("custom");
  outer->add_input("x");
  outer->add_output("y");
  {
    NodeProto *n = outer->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = outer->add_node();
    n->set_op_type("Middle");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Outer");
  node->set_domain("custom");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 8.0f);
  EXPECT_FLOAT_EQ(res[1], 20.0f);
  EXPECT_FLOAT_EQ(res[2], -24.0f);
}

TEST(RunModel, ModelLocalFunctionOverloadDisambiguation) {
  // Two functions share the same (domain, name) but differ by overload.
  // The dispatcher must pick the one matching node.overload.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *f_sum = model.add_functions();
  f_sum->set_name("Combine");
  f_sum->set_domain("custom");
  f_sum->set_overload("sum");
  f_sum->add_input("a");
  f_sum->add_input("b");
  f_sum->add_output("out");
  {
    NodeProto *n = f_sum->add_node();
    n->set_op_type("Add");
    n->add_input("a");
    n->add_input("b");
    n->add_output("out");
  }

  FunctionProto *f_diff = model.add_functions();
  f_diff->set_name("Combine");
  f_diff->set_domain("custom");
  f_diff->set_overload("diff");
  f_diff->add_input("a");
  f_diff->add_input("b");
  f_diff->add_output("out");
  {
    NodeProto *n = f_diff->add_node();
    n->set_op_type("Sub");
    n->add_input("a");
    n->add_input("b");
    n->add_output("out");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *n1 = g->add_node();
  n1->set_op_type("Combine");
  n1->set_domain("custom");
  n1->set_overload("sum");
  n1->add_input("x");
  n1->add_input("y");
  n1->add_output("s");
  NodeProto *n2 = g->add_node();
  n2->set_op_type("Combine");
  n2->set_domain("custom");
  n2->set_overload("diff");
  n2->add_input("x");
  n2->add_input("y");
  n2->add_output("d");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {10.0f}));
  rt.Set("y", Tensor::FromFloat("y", {1}, {3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("s"));
  ASSERT_TRUE(rt.Has("d"));
  EXPECT_FLOAT_EQ(rt.Get("s").AsFloat()[0], 13.0f);
  EXPECT_FLOAT_EQ(rt.Get("d").AsFloat()[0], 7.0f);
}

TEST(RunModel, ModelLocalFunctionWrongInputCountThrows) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *func = model.add_functions();
  func->set_name("F");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("a");
  fn->add_input("b");
  fn->add_output("out");

  GraphProto *g = model.add_graph();
  NodeProto *node = g->add_node();
  node->set_op_type("F");
  node->set_domain("custom");
  node->add_input("x"); // only 1, function expects 2
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {1.0f}));

  EXPECT_THROW(RunModel(model, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Model-local function with linked (ref_attr_name) attributes
// ---------------------------------------------------------------------------

namespace {

// Builds a constant-only sub-graph that emits a single FLOAT scalar
// named ``out_name`` set to ``value``. Used as a GRAPH attribute below.
void FillConstantBranch(GraphProto &g, const std::string &branch_name, const std::string &init_name,
                        const std::string &out_name, float value) {
  g.set_name(branch_name);
  TensorProto *init = g.add_initializer();
  init->set_name(init_name);
  init->set_data_type(TensorProto::DataType::FLOAT);
  init->add_float_data(value);
  // The If implementation expects each branch sub-graph to produce its
  // output via at least one node. Use Add(init, init) so the value is
  // doubled, mirroring the existing ``IfNodeWithBranchSubgraphs`` test.
  NodeProto *add = g.add_node();
  add->set_op_type("Add");
  add->add_input(init_name);
  add->add_input(init_name);
  add->add_output(out_name);
  g.add_output()->set_name(out_name);
}

} // namespace

TEST(RunModel, ModelLocalFunctionLinkedAttributeFromCallSite) {
  // Define a model-local function "Pick(cond)" whose body delegates to
  // an ``If`` node where both ``then_branch`` and ``else_branch`` are
  // attribute references (``ref_attr_name``) to call-site attributes of
  // the same name. Two distinct call-sites supply different branch
  // sub-graphs, proving that the attribute is resolved per call rather
  // than baked in at function-definition time.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

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

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tattr->add_g(), "then_g", "t", "z", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*eattr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt_true(KernelContext(DefaultOpset(18)));
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt_true);
  ASSERT_TRUE(rt_true.Has("out"));
  EXPECT_FLOAT_EQ(rt_true.Get("out").AsFloat()[0], 20.0f);

  RuntimeContext rt_false(KernelContext(DefaultOpset(18)));
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  RunModel(model, rt_false);
  ASSERT_TRUE(rt_false.Has("out"));
  EXPECT_FLOAT_EQ(rt_false.Get("out").AsFloat()[0], 2.0f);
}

TEST(RunModel, ModelLocalFunctionLinkedAttributeUsesDefault) {
  // The function declares typed defaults for ``then_branch`` and
  // ``else_branch`` via ``attribute_proto``. The call-site omits both
  // and the defaults must be used instead.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  AttributeProto *tdef = func->add_attribute_proto();
  tdef->set_name("then_branch");
  tdef->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tdef->add_g(), "then_default_g", "t_def", "z", 5.0f);
  AttributeProto *edef = func->add_attribute_proto();
  edef->set_name("else_branch");
  edef->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*edef->add_g(), "else_default_g", "e_def", "z", 7.0f);
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

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt);
  ASSERT_TRUE(rt.Has("out"));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 10.0f);
}

TEST(RunModel, ModelLocalFunctionDoesNotMutateModel) {
  // Verify that resolving ``ref_attr_name`` references operates on a
  // copy of the FunctionProto so the source ModelProto's function body
  // is unchanged after RunModel.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

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

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tattr->add_g(), "then_g", "t", "z", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*eattr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt);

  // The function body's attribute must still be a reference (no graph
  // baked in) so the same model can be executed again with a different
  // call-site attribute.
  const FunctionProto &saved = model.functions()[0];
  const AttributeProto &a = saved.node()[0].attribute()[0];
  EXPECT_EQ(a.ref_attr_name().as_string(), "then_branch");
  EXPECT_FALSE(a.has_g());
}

TEST(RunModel, IfNodeWithBranchSubgraphs) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("out");

  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  then_g->set_name("then_graph");
  TensorProto *then_init = then_g->add_initializer();
  then_init->set_name("t");
  then_init->set_data_type(TensorProto::DataType::FLOAT);
  then_init->add_float_data(10.0f);
  NodeProto *then_add = then_g->add_node();
  then_add->set_op_type("Add");
  then_add->add_input("t");
  then_add->add_input("t");
  then_add->add_output("z");
  then_g->add_output()->set_name("z");

  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *else_g = else_attr->add_g();
  else_g->set_name("else_graph");
  TensorProto *else_init = else_g->add_initializer();
  else_init->set_name("e");
  else_init->set_data_type(TensorProto::DataType::FLOAT);
  else_init->add_float_data(1.0f);
  NodeProto *else_add = else_g->add_node();
  else_add->set_op_type("Add");
  else_add->add_input("e");
  else_add->add_input("e");
  else_add->add_output("z");
  else_g->add_output()->set_name("z");

  RuntimeContext rt_true(KernelContext(DefaultOpset(18)));
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt_true);
  ASSERT_TRUE(rt_true.Has("out"));
  EXPECT_FLOAT_EQ(rt_true.Get("out").AsFloat()[0], 20.0f);

  RuntimeContext rt_false(KernelContext(DefaultOpset(18)));
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  RunModel(model, rt_false);
  ASSERT_TRUE(rt_false.Has("out"));
  EXPECT_FLOAT_EQ(rt_false.Get("out").AsFloat()[0], 2.0f);
}

TEST(RunModel, LoopNodeRunsBodySubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *loop = g->add_node();
  loop->set_op_type("Loop");
  loop->add_input("M");
  loop->add_input("cond");
  loop->add_input("s_init");
  loop->add_output("s_final");
  loop->add_output("scan");

  AttributeProto *body_attr = loop->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("loop_body");
  body->add_input()->set_name("iter");
  body->add_input()->set_name("cond_in");
  body->add_input()->set_name("s_in");
  TensorProto *one = body->add_initializer();
  one->set_name("one");
  one->set_data_type(TensorProto::DataType::FLOAT);
  one->add_float_data(1.0f);
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("s_in");
  add->add_input("one");
  add->add_output("s_out");
  body->add_output()->set_name("cond_in");
  body->add_output()->set_name("s_out");
  body->add_output()->set_name("s_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("s_final"));
  ASSERT_TRUE(rt.Has("scan"));
  EXPECT_FLOAT_EQ(rt.Get("s_final").AsFloat()[0], 3.0f);
  ASSERT_EQ(rt.Get("scan").shape, (std::vector<int64_t>{3}));
  const float *scan = rt.Get("scan").AsFloat();
  EXPECT_FLOAT_EQ(scan[0], 1.0f);
  EXPECT_FLOAT_EQ(scan[1], 2.0f);
  EXPECT_FLOAT_EQ(scan[2], 3.0f);
}

TEST(RunModel, ScanNodeRunsBodySubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  scan->add_input("state0");
  scan->add_input("x");
  scan->add_output("state_final");
  scan->add_output("y");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("state_in");
  body->add_input()->set_name("x_in");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("state_in");
  add->add_input("x_in");
  add->add_output("state_out");
  body->add_output()->set_name("state_out");
  body->add_output()->set_name("state_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("state0", Tensor::FromFloat("state0", {}, {0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("state_final"));
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("state_final").AsFloat()[0], 6.0f);
  ASSERT_EQ(rt.Get("y").shape, (std::vector<int64_t>{3}));
  const float *y = rt.Get("y").AsFloat();
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 3.0f);
  EXPECT_FLOAT_EQ(y[2], 6.0f);
}

} // namespace Test
