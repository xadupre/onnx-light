// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the structural in-place reuse guess
// (:cpp:func:`onnx_optim::annotations::ComputeInPlaceReuse`). The analysis is
// driven entirely by the shapes inferred into a ``ShapesContext`` and by
// value lifetimes, so each test runs shape inference on a small graph and
// then checks which (output, input) reuse opportunities are reported.

#include "onnx_optim/annotations/inplace_reuse.h"

#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_optim::OptimShape;
using onnx_optim::OptimTensor;
using onnx_optim::TensorType;
using onnx_optim::annotations::ComputeContext;
using onnx_optim::annotations::ComputeInPlaceReuse;
using onnx_optim::annotations::InPlaceReuse;
using onnx_optim::annotations::InPlaceReuseKind;
using onnx_optim::annotations::WriteInPlaceReuseToMetadata;
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

void SetTensorType(ValueInfoProto &vi, TensorProto::DataType elem_type,
                   const std::vector<int64_t> &shape) {
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto *sp = tt->add_shape();
  for (int64_t d : shape) {
    sp->add_dim()->set_dim_value(d);
  }
}

void SetFloatTensorType(ValueInfoProto &vi, const std::vector<int64_t> &shape) {
  SetTensorType(vi, TensorProto::DataType::FLOAT, shape);
}

NodeProto MakeCastNode(const std::string &input, const std::string &output,
                       TensorProto::DataType to) {
  NodeProto node = MakeNode("Cast", {input}, {output});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("to");
  attr->set_i(static_cast<int64_t>(to));
  return node;
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

ValueInfoProto *AddTypedOutput(GraphProto &graph, const std::string &name,
                               TensorProto::DataType elem_type, const std::vector<int64_t> &shape) {
  ValueInfoProto *vi = graph.add_output();
  vi->set_name(name);
  SetTensorType(*vi, elem_type, shape);
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
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
  ASSERT_EQ(reuse[2].size(), 1u);
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
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
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
}

// A Transpose may still reuse its input buffer when the byte size matches even
// though the shape layout differs.
TEST(OnnxOptimInPlaceReuse, TransposeWithSameByteSizeIsReportedAsGreater) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  // Abs keeps the shape; each Transpose changes the layout but keeps the same
  // element count and dtype, so both opportunities are reported as kGreater.
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  NodeProto transpose = MakeNode("Transpose", {"A"}, {"B"});
  *graph.add_node() = transpose;
  *graph.add_node() = MakeNode("Transpose", {"B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 3u);
  EXPECT_TRUE(reuse[0].empty());
  ASSERT_EQ(reuse[1].size(), 1u);
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kGreater}));
  ASSERT_EQ(reuse[2].size(), 1u);
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kGreater}));
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

// When the only compatible input buffer is strictly larger than the output,
// the opportunity is still reported but flagged as ``kGreater``.
TEST(OnnxOptimInPlaceReuse, LargerInputBufferIsReportedAsGreater) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {4});
  AddTypedOutput(graph, "Y", TensorProto::DataType::INT32, {4});
  // A is INT64 (32 bytes); Y is INT32 (16 bytes). A's buffer is strictly
  // larger than Y, so Y may still reuse it in place.
  *graph.add_node() = MakeCastNode("X", "A", TensorProto::DataType::INT64);
  *graph.add_node() = MakeCastNode("A", "Y", TensorProto::DataType::INT32);

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 2u);
  // Node 0 reads the declared graph input X.
  EXPECT_TRUE(reuse[0].empty());
  // Node 1 reuses the larger intermediate A buffer for its smaller output.
  ASSERT_EQ(reuse[1].size(), 1u);
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kGreater}));
}

// When both a same-sized and a strictly larger input are available for the
// same output, the same-sized (kEqual) input is preferred even when the
// larger one appears first in the input list.
TEST(OnnxOptimInPlaceReuse, EqualSizedInputIsPreferredOverLarger) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {4});
  AddTypedOutput(graph, "Y", TensorProto::DataType::INT32, {4});
  // BIG (INT64, 32 bytes) is produced first and listed first; EQ (INT32,
  // 16 bytes) matches Y exactly. Both are last used by the consumer node.
  *graph.add_node() = MakeNode("P", {"X"}, {"BIG"});
  *graph.add_node() = MakeNode("Q", {"X"}, {"EQ"});
  *graph.add_node() = MakeNode("Consume", {"BIG", "EQ"}, {"Y"});

  // Populate the descriptors directly: the structural reuse guess depends only
  // on the inferred shapes/types, not on the op semantics.
  ShapesContext ctx;
  ctx.Set("X", OptimTensor(nullptr, TensorType::kInt32, OptimShape{4}));
  ctx.Set("BIG", OptimTensor(nullptr, TensorType::kInt64, OptimShape{4}));
  ctx.Set("EQ", OptimTensor(nullptr, TensorType::kInt32, OptimShape{4}));
  ctx.Set("Y", OptimTensor(nullptr, TensorType::kInt32, OptimShape{4}));

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 3u);
  // The consumer (node 2) reuses EQ (input index 1), the same-sized buffer,
  // rather than the larger BIG buffer at the lower input index 0.
  ASSERT_EQ(reuse[2].size(), 1u);
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 1, InPlaceReuseKind::kEqual}));
}

// WriteInPlaceReuseToMetadata records the opportunities of each node into its
// ``metadata_props``, leaving nodes without any opportunity untouched.
TEST(OnnxOptimInPlaceReuse, WriteInPlaceReuseToMetadata) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  WriteInPlaceReuseToMetadata(graph, ctx);
  ASSERT_EQ(graph.node().size(), 3);
  // Node 0 reads the declared graph input X, so it has no reuse and no
  // metadata is written for it.
  EXPECT_EQ(graph.node()[0].metadata_props().size(), 0);
  ASSERT_EQ(graph.node()[1].metadata_props().size(), 2);
  EXPECT_EQ(graph.node()[1].metadata_props()[0].key().as_string(),
            std::string(onnx_optim::annotations::kInPlaceReuseMetadataKey));
  EXPECT_EQ(graph.node()[1].metadata_props()[0].value().as_string(), std::string("0:0:equal"));
  EXPECT_EQ(graph.node()[1].metadata_props()[1].key().as_string(),
            std::string(onnx_optim::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(graph.node()[1].metadata_props()[1].value().as_string(), std::string("A"));
  ASSERT_EQ(graph.node()[2].metadata_props().size(), 2);
  EXPECT_EQ(graph.node()[2].metadata_props()[0].value().as_string(), std::string("0:0:equal"));
  EXPECT_EQ(graph.node()[2].metadata_props()[1].key().as_string(),
            std::string(onnx_optim::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(graph.node()[2].metadata_props()[1].value().as_string(), std::string("B"));
}

// A strictly larger reused buffer is recorded with the ``greater`` kind, and an
// existing entry under the same key is replaced in place.
TEST(OnnxOptimInPlaceReuse, WriteInPlaceReuseToMetadataGreaterAndUpdate) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {4});
  AddTypedOutput(graph, "Y", TensorProto::DataType::INT32, {4});
  *graph.add_node() = MakeCastNode("X", "A", TensorProto::DataType::INT64);
  *graph.add_node() = MakeCastNode("A", "Y", TensorProto::DataType::INT32);
  // Pre-existing entry under the same key must be replaced, not duplicated.
  (*graph.mutable_node())[1].add_metadata(onnx_optim::annotations::kInPlaceReuseMetadataKey,
                                          "stale");

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  WriteInPlaceReuseToMetadata(graph, ctx);
  EXPECT_EQ(graph.node()[0].metadata_props().size(), 0);
  ASSERT_EQ(graph.node()[1].metadata_props().size(), 2);
  EXPECT_EQ(graph.node()[1].metadata_props()[0].value().as_string(), std::string("0:0:greater"));
  EXPECT_EQ(graph.node()[1].metadata_props()[1].key().as_string(),
            std::string(onnx_optim::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(graph.node()[1].metadata_props()[1].value().as_string(), std::string("A"));
}

// By default a declared graph input is never overwritten in place, but when
// ``allow_input_overwrite`` is set the first node may reuse the input buffer
// once it reaches its last use.
TEST(OnnxOptimInPlaceReuse, AllowInputOverwriteReusesGraphInput) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  // Default: the graph input X must not be overwritten.
  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 2u);
  EXPECT_TRUE(reuse[0].empty());

  // Opt-in: node 0 is the last (and only) use of X, so it may reuse it.
  std::vector<std::vector<InPlaceReuse>> reuse_ovw =
      ComputeInPlaceReuse(graph, ctx, /*allow_input_overwrite=*/true);
  ASSERT_EQ(reuse_ovw.size(), 2u);
  ASSERT_EQ(reuse_ovw[0].size(), 1u);
  EXPECT_EQ(reuse_ovw[0][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
  ASSERT_EQ(reuse_ovw[1].size(), 1u);
  EXPECT_EQ(reuse_ovw[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
}

// Even with ``allow_input_overwrite`` set, an input that is read by more than
// one node is only reused at its last use, and an input that is also a graph
// output is still protected.
TEST(OnnxOptimInPlaceReuse, AllowInputOverwriteRespectsLifetimes) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  // X feeds two nodes, so node 0 must not overwrite it; node 1 is the last use.
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Add", {"X", "A"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse =
      ComputeInPlaceReuse(graph, ctx, /*allow_input_overwrite=*/true);
  ASSERT_EQ(reuse.size(), 2u);
  // X is still alive (read by node 1), so node 0 must not reuse it.
  EXPECT_TRUE(reuse[0].empty());
  // Node 1 is the last use of X; the first compatible input wins.
  ASSERT_EQ(reuse[1].size(), 1u);
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
}

// A graph input that is also a declared graph output must never be reused,
// even when ``allow_input_overwrite`` is set, since it must outlive the run.
TEST(OnnxOptimInPlaceReuse, AllowInputOverwriteKeepsInputThatIsAlsoOutput) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {2, 2});
  AddOutput(graph, "X", {2, 2});
  AddOutput(graph, "Y", {2, 2});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse =
      ComputeInPlaceReuse(graph, ctx, /*allow_input_overwrite=*/true);
  ASSERT_EQ(reuse.size(), 1u);
  // X is also a declared output, so it is protected regardless of the option.
  EXPECT_TRUE(reuse[0].empty());
}

// The ComputeContext class stores the per-node reuse result and exposes it
// through Size()/Reuse()/NodeReuse(), matching the free-function output.
TEST(OnnxOptimInPlaceReuse, ComputeContextStoresResult) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  ComputeContext inplace;
  EXPECT_TRUE(inplace.Empty());
  EXPECT_EQ(inplace.Size(), 0u);
  inplace.ComputeInPlaceReuseGraph(graph, ctx);

  EXPECT_FALSE(inplace.Empty());
  ASSERT_EQ(inplace.Size(), 3u);
  // The stored result matches the free-function wrapper.
  EXPECT_EQ(inplace.Reuse(), ComputeInPlaceReuse(graph, ctx));
  EXPECT_TRUE(inplace.NodeReuse(0).empty());
  ASSERT_EQ(inplace.NodeReuse(1).size(), 1u);
  EXPECT_EQ(inplace.NodeReuse(1)[0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
  ASSERT_EQ(inplace.NodeReuse(2).size(), 1u);
  EXPECT_EQ(inplace.NodeReuse(2)[0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));

  // NodeReuse rejects an out-of-bounds index.
  EXPECT_THROW(inplace.NodeReuse(3), std::out_of_range);

  // Clear empties the stored result, recomputation refills it.
  inplace.Clear();
  EXPECT_TRUE(inplace.Empty());
}

// ComputeContext honours allow_input_overwrite just like the free function.
TEST(OnnxOptimInPlaceReuse, ComputeContextAllowInputOverwrite) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, /*allow_input_overwrite=*/true);
  ASSERT_EQ(inplace.Size(), 2u);
  ASSERT_EQ(inplace.NodeReuse(0).size(), 1u);
  EXPECT_EQ(inplace.NodeReuse(0)[0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
}

// ComputeContext::WriteToMetadata records the same triplets as the free
// function and rejects a graph whose node count no longer matches the result.
TEST(OnnxOptimInPlaceReuse, ComputeContextWriteToMetadata) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx);
  inplace.WriteToMetadata(graph);
  ASSERT_EQ(graph.node().size(), 3);
  EXPECT_EQ(graph.node()[0].metadata_props().size(), 0);
  ASSERT_EQ(graph.node()[1].metadata_props().size(), 2);
  EXPECT_EQ(graph.node()[1].metadata_props()[0].value().as_string(), std::string("0:0:equal"));
  EXPECT_EQ(graph.node()[1].metadata_props()[1].key().as_string(),
            std::string(onnx_optim::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(graph.node()[1].metadata_props()[1].value().as_string(), std::string("A"));
  ASSERT_EQ(graph.node()[2].metadata_props().size(), 2);
  EXPECT_EQ(graph.node()[2].metadata_props()[0].value().as_string(), std::string("0:0:equal"));
  EXPECT_EQ(graph.node()[2].metadata_props()[1].key().as_string(),
            std::string(onnx_optim::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(graph.node()[2].metadata_props()[1].value().as_string(), std::string("B"));

  // Writing into a graph with a different node count is rejected.
  GraphProto smaller;
  smaller.set_name("g");
  *smaller.add_node() = MakeNode("Abs", {"X"}, {"A"});
  EXPECT_THROW(inplace.WriteToMetadata(smaller), std::invalid_argument);
}

} // namespace Test
