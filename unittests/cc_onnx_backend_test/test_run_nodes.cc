// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/run_nodes.h"
#include "onnx_backend_test/simple_tensor.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::KernelDispatchTable;
using onnx_backend_test::RunNode;
using onnx_backend_test::RunNodes;
using onnx_backend_test::Tensor;
using onnx_backend_test::TensorMap;
using onnx_backend_test::kernel::KernelContext;

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
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  tensors["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  KernelContext ctx(DefaultOpset(18));
  RunNode(node, tensors, ctx);
  ASSERT_NE(tensors.find("z"), tensors.end());
  const Tensor &z = tensors["z"];
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
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {2}, {-1.5f, 2.5f});
  NodeProto node = MakeNode("Abs", {"x"}, {"y"}); // empty domain
  EXPECT_TRUE(node.domain().as_string().empty());
  KernelContext ctx(DefaultOpset(18));
  RunNode(node, tensors, ctx);
  const float *got = tensors["y"].AsFloat();
  ASSERT_EQ(tensors["y"].element_count(), 2);
  EXPECT_FLOAT_EQ(got[0], 1.5f);
  EXPECT_FLOAT_EQ(got[1], 2.5f);
}

TEST(RunNodes, RunNodesOnRepeatedProtoFieldChain) {
  // Builds the small graph:  t = Mul(x, y);  out = Sub(t, z)
  // and runs it through the iterator overload that drives a
  // RepeatedProtoField<NodeProto> directly (mirroring how a caller
  // would feed ``graph.node()``).
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  tensors["y"] = Tensor::FromFloat("y", {2}, {3.0f, 4.0f});
  tensors["z"] = Tensor::FromFloat("z", {2}, {0.5f, 0.25f});

  utils::RepeatedProtoField<NodeProto> nodes;
  *nodes.Add() = MakeNode("Mul", {"x", "y"}, {"t"});
  *nodes.Add() = MakeNode("Sub", {"t", "z"}, {"out"});

  KernelContext ctx(DefaultOpset(18));
  RunNodes(nodes, tensors, ctx);

  ASSERT_NE(tensors.find("t"), tensors.end());
  ASSERT_NE(tensors.find("out"), tensors.end());
  const float *out = tensors["out"].AsFloat();
  ASSERT_EQ(tensors["out"].element_count(), 2);
  EXPECT_FLOAT_EQ(out[0], 1.0f * 3.0f - 0.5f);
  EXPECT_FLOAT_EQ(out[1], 2.0f * 4.0f - 0.25f);
  EXPECT_EQ(tensors["out"].name, "out");
}

TEST(RunNodes, RunNodesOnIteratorRangeFromVector) {
  // Same graph, but driven through the generic iterator overload so
  // any container whose elements dereference to ``NodeProto`` works.
  TensorMap tensors;
  tensors["a"] = Tensor::FromFloat("a", {1}, {6.0f});
  tensors["b"] = Tensor::FromFloat("b", {1}, {2.0f});

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Div", {"a", "b"}, {"q"})); // q = 3
  nodes.push_back(MakeNode("Neg", {"q"}, {"out"}));    // out = -3

  KernelContext ctx(DefaultOpset(18));
  RunNodes(nodes.begin(), nodes.end(), tensors, ctx);

  const float *out = tensors["out"].AsFloat();
  ASSERT_EQ(tensors["out"].element_count(), 1);
  EXPECT_FLOAT_EQ(out[0], -3.0f);
}

TEST(RunNodes, RunNodeUnsupportedOpTypeThrows) {
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  NodeProto node = MakeNode("ThisOpDoesNotExist", {"x"}, {"y"});
  KernelContext ctx(DefaultOpset(18));
  EXPECT_THROW(RunNode(node, tensors, ctx), std::invalid_argument);
}

TEST(RunNodes, RunNodeMissingInputThrows) {
  TensorMap tensors; // empty: "x" is not present
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  KernelContext ctx(DefaultOpset(18));
  EXPECT_THROW(RunNode(node, tensors, ctx), std::invalid_argument);
}

TEST(RunNodes, RunNodeWrongInputCountThrows) {
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  // Add expects two inputs but we only declare one.
  NodeProto node = MakeNode("Add", {"x"}, {"y"});
  KernelContext ctx(DefaultOpset(18));
  EXPECT_THROW(RunNode(node, tensors, ctx), std::invalid_argument);
}

} // namespace Test
