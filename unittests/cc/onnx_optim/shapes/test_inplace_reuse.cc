// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the structural in-place reuse guess
// (:cpp:func:`onnx_optim::shapes::ComputeInPlaceReuse`). The analysis is
// driven entirely by the shapes inferred into a ``ShapesContext`` and by
// value lifetimes, so each test runs shape inference on a small graph and
// then checks which (output, input) reuse opportunities are reported.

#include "onnx_optim/shapes/inplace_reuse.h"

#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_optim::shapes::ComputeInPlaceReuse;
using onnx_optim::shapes::InPlaceReuse;
using onnx_optim::shapes::ShapesContext;

namespace Test {

namespace {

NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type(op_type);
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  for (const auto &out : outputs) {
    node.add_output(out);
  }
  return node;
}

void SetFloatTensorType(ValueInfoProto &vi, const std::vector<int64_t> &shape) {
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *sp = tt->add_shape();
  for (int64_t d : shape) {
    sp->add_dim()->set_dim_value(d);
  }
}

ValueInfoProto *AddInput(GraphProto &graph, const std::string &name,
                         const std::vector<int64_t> &shape) {
  ValueInfoProto *vi = graph.add_input();
  vi->set_name(name);
  SetFloatTensorType(*vi, shape);
  return vi;
}

ValueInfoProto *AddOutput(GraphProto &graph, const std::string &name,
                          const std::vector<int64_t> &shape) {
  ValueInfoProto *vi = graph.add_output();
  vi->set_name(name);
  SetFloatTensorType(*vi, shape);
  return vi;
}

} // namespace

// A linear chain of element-wise Abs nodes: every node but the first (whose
// input is the declared graph input) may reuse its input buffer in place.
TEST(OnnxOptimInPlaceReuse, AbsChainReusesIntermediates) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 3u);
  // Node 0 reads the declared graph input X, which must not be overwritten.
  EXPECT_TRUE(reuse[0].empty());
  // Nodes 1 and 2 may reuse their (intermediate) input buffer.
  ASSERT_EQ(reuse[1].size(), 1u);
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0}));
  ASSERT_EQ(reuse[2].size(), 1u);
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 0}));
}

// An intermediate that is read by more than one later node cannot be reused
// before its last use.
TEST(OnnxOptimInPlaceReuse, ValueReadTwiceIsReusedOnlyAtLastUse) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Add", {"A", "B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 3u);
  EXPECT_TRUE(reuse[0].empty());
  // A is still alive (read by node 2), so node 1 must not reuse it.
  EXPECT_TRUE(reuse[1].empty());
  // node 2 is the last use of both A and B; the first compatible input wins.
  ASSERT_EQ(reuse[2].size(), 1u);
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 0}));
}

// A node whose output shape differs from its input shape offers no reuse.
TEST(OnnxOptimInPlaceReuse, ShapeMismatchYieldsNoReuse) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  // Abs keeps the shape; Transpose flips it to [4, 3], so the Transpose
  // output cannot alias its [3, 4] input even though both are intermediates.
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  NodeProto transpose = MakeNode("Transpose", {"A"}, {"B"});
  *graph.add_node() = transpose;
  *graph.add_node() = MakeNode("Transpose", {"B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 3u);
  EXPECT_TRUE(reuse[0].empty());
  // [3,4] -> [4,3]: shapes differ, no in-place reuse.
  EXPECT_TRUE(reuse[1].empty());
  // [4,3] -> [3,4]: shapes differ, no in-place reuse.
  EXPECT_TRUE(reuse[2].empty());
}

// The buffer of a declared graph output must survive, so it is never offered
// as a reusable input to a later node.
TEST(OnnxOptimInPlaceReuse, GraphOutputInputIsNotReused) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {2, 2});
  AddOutput(graph, "A", {2, 2});
  AddOutput(graph, "Y", {2, 2});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 2u);
  EXPECT_TRUE(reuse[0].empty());
  // A is a declared graph output; node 1 must not overwrite it in place.
  EXPECT_TRUE(reuse[1].empty());
}

} // namespace Test
