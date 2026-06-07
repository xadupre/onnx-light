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

} // namespace Test
