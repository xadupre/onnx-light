// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the structural in-place reuse guess
// (:cpp:func:`core::annotations::ComputeInPlaceReuse`). The analysis is
// driven entirely by the shapes inferred into a ``ShapesContext`` and by
// value lifetimes, so each test runs shape inference on a small graph and
// then checks which (output, input) reuse opportunities are reported.

#include "onnx_core/compute/inplace_reuse.h"

#include "onnx_core/compute/compute_context.h"
#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::annotations::ComputeContext;
using core::annotations::ComputeInPlaceReuse;
using core::annotations::InPlaceReuse;
using core::annotations::InPlaceReuseKind;
using core::annotations::WriteInPlaceReuseToMetadata;
using core::shapes::ShapesContext;
using core::symbolic::SymDim;
using core::symbolic::SymShape;
using core::symbolic::SymTensor;
using core::symbolic::TensorType;

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
TEST(OnnxOptimInPlaceReuse, TransposeWithSameByteSizeIsReportedAsEqual) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  // Abs keeps the shape; each Transpose changes the layout but keeps the same
  // element count and dtype, so both opportunities are reported as kEqual.
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
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
  ASSERT_EQ(reuse[2].size(), 1u);
  EXPECT_EQ(reuse[2][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
}

// A Transpose whose input has a symbolic batch dimension is reported as
// kEqual: the byte-size expressions for [batch,4,4] and [4,batch,4] are
// both "64*batch" after simplification, so the buffer sizes are equal.
TEST(OnnxOptimInPlaceReuse, TransposeWithSymbolicDimIsReportedAsEqual) {
  GraphProto graph;
  graph.set_name("g");

  // Declare minimal input/output so the graph proto is valid; the actual
  // shapes are supplied to the ShapesContext manually below.
  graph.add_input()->set_name("X");
  graph.add_output()->set_name("Y");

  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Transpose", {"A"}, {"Y"});

  // Provide symbolic shapes directly — X and A share [batch, 4, 4]; the
  // Transpose output Y has the same elements rearranged as [4, batch, 4].
  ShapesContext ctx;
  ctx.Set("X",
          SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim("batch"), SymDim(4), SymDim(4)}));
  ctx.Set("A",
          SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim("batch"), SymDim(4), SymDim(4)}));
  ctx.Set("Y",
          SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(4), SymDim("batch"), SymDim(4)}));

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 2u);
  // Node 0: Abs(X) → A — X is a declared graph input, must not be overwritten.
  EXPECT_TRUE(reuse[0].empty());
  // Node 1: Transpose(A) → Y — A's buffer equals Y's byte size, so kEqual.
  ASSERT_EQ(reuse[1].size(), 1u);
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
}

// A square-matrix Transpose ([4,4] -> [4,4]) has identical input and output
// shapes, so SameStorage returns true and the reuse is classified as kEqual.
TEST(OnnxOptimInPlaceReuse, TransposeSquareShapeIsReportedAsEqual) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {4, 4});
  AddOutput(graph, "Y", {4, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Transpose", {"A"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  std::vector<std::vector<InPlaceReuse>> reuse = ComputeInPlaceReuse(graph, ctx);
  ASSERT_EQ(reuse.size(), 2u);
  EXPECT_TRUE(reuse[0].empty());
  ASSERT_EQ(reuse[1].size(), 1u);
  EXPECT_EQ(reuse[1][0], (InPlaceReuse{0, 0, InPlaceReuseKind::kEqual}));
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
  ctx.Set("X", SymTensor(nullptr, TensorType::kInt32, SymShape{4}));
  ctx.Set("BIG", SymTensor(nullptr, TensorType::kInt64, SymShape{4}));
  ctx.Set("EQ", SymTensor(nullptr, TensorType::kInt32, SymShape{4}));
  ctx.Set("Y", SymTensor(nullptr, TensorType::kInt32, SymShape{4}));

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
  // Node 0 reads the declared graph input X for the last time.
  ASSERT_EQ(graph.node()[0].metadata_props().size(), 1);
  EXPECT_EQ(std::string(graph.node()[0].metadata_props()[0].key()),
            std::string(core::annotations::kNotUsedAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[0].metadata_props()[0].value()), std::string("X"));
  ASSERT_EQ(graph.node()[1].metadata_props().size(), 2);
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[0].key()),
            std::string(core::annotations::kInPlaceReuseMetadataKey));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[0].value()), std::string("0:0:equal"));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[1].key()),
            std::string(core::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[1].value()), std::string("A"));
  ASSERT_EQ(graph.node()[2].metadata_props().size(), 2);
  EXPECT_EQ(std::string(graph.node()[2].metadata_props()[0].value()), std::string("0:0:equal"));
  EXPECT_EQ(std::string(graph.node()[2].metadata_props()[1].key()),
            std::string(core::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[2].metadata_props()[1].value()), std::string("B"));
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
  (*graph.mutable_node())[1].add_metadata(core::annotations::kInPlaceReuseMetadataKey, "stale");

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  WriteInPlaceReuseToMetadata(graph, ctx);
  ASSERT_EQ(graph.node()[0].metadata_props().size(), 1);
  EXPECT_EQ(std::string(graph.node()[0].metadata_props()[0].key()),
            std::string(core::annotations::kNotUsedAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[0].metadata_props()[0].value()), std::string("X"));
  ASSERT_EQ(graph.node()[1].metadata_props().size(), 2);
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[0].value()), std::string("0:0:greater"));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[1].key()),
            std::string(core::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[1].value()), std::string("A"));
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
  ASSERT_EQ(graph.node()[0].metadata_props().size(), 1);
  EXPECT_EQ(std::string(graph.node()[0].metadata_props()[0].key()),
            std::string(core::annotations::kNotUsedAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[0].metadata_props()[0].value()), std::string("X"));
  ASSERT_EQ(graph.node()[1].metadata_props().size(), 2);
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[0].value()), std::string("0:0:equal"));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[1].key()),
            std::string(core::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[1].metadata_props()[1].value()), std::string("A"));
  ASSERT_EQ(graph.node()[2].metadata_props().size(), 2);
  EXPECT_EQ(std::string(graph.node()[2].metadata_props()[0].value()), std::string("0:0:equal"));
  EXPECT_EQ(std::string(graph.node()[2].metadata_props()[1].key()),
            std::string(core::annotations::kReleaseAfterMetadataKey));
  EXPECT_EQ(std::string(graph.node()[2].metadata_props()[1].value()), std::string("B"));

  // Writing into a graph with a different node count is rejected.
  GraphProto smaller;
  smaller.set_name("g");
  *smaller.add_node() = MakeNode("Abs", {"X"}, {"A"});
  EXPECT_THROW(inplace.WriteToMetadata(smaller), std::invalid_argument);
}

// When value_tags are passed, released values with the "shape" tag are stored
// in ReleaseAfterShapeTagged and written to kReleaseAfterShapeTagMetadataKey.
TEST(OnnxOptimInPlaceReuse, ComputeContextShapeTagReleaseInfo) {
  // Graph: Shape(X) -> S, Reshape(X, S) -> Y
  // S is the output of Shape, so it should be tagged "shape".
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {2, 3});
  AddOutput(graph, "Y", {2, 3});
  auto *shape_node = graph.add_node();
  shape_node->set_op_type("Shape");
  shape_node->add_input("X");
  shape_node->add_output("S");
  auto *reshape_node = graph.add_node();
  reshape_node->set_op_type("Reshape");
  reshape_node->add_input("X");
  reshape_node->add_input("S");
  reshape_node->add_output("Y");

  ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.ComputeShapeGraph(graph);

  // Build value_tags: mark "S" as shape-tagged (as InferValueAndNodeTags would).
  const std::unordered_map<std::string, std::string> value_tags = {{"S", "shape"}};

  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, value_tags);

  ASSERT_EQ(inplace.Size(), 2u);
  // S is released after the Reshape node (node 1).
  EXPECT_TRUE(inplace.NodeReleaseAfterShapeTagged(0).empty());
  EXPECT_EQ(inplace.NodeReleaseAfterShapeTagged(1), (std::vector<std::string>{"S"}));

  // ReleaseAfterShapeTagged matches the per-node accessors.
  ASSERT_EQ(inplace.ReleaseAfterShapeTagged().size(), 2u);
  EXPECT_TRUE(inplace.ReleaseAfterShapeTagged()[0].empty());
  EXPECT_EQ(inplace.ReleaseAfterShapeTagged()[1], (std::vector<std::string>{"S"}));
}

// WriteToMetadata writes kReleaseAfterShapeTagMetadataKey when shape-tagged
// values exist in the release list.
TEST(OnnxOptimInPlaceReuse, ComputeContextWriteToMetadataShapeTag) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {2, 3});
  AddOutput(graph, "Y", {2, 3});
  auto *shape_node = graph.add_node();
  shape_node->set_op_type("Shape");
  shape_node->add_input("X");
  shape_node->add_output("S");
  auto *reshape_node = graph.add_node();
  reshape_node->set_op_type("Reshape");
  reshape_node->add_input("X");
  reshape_node->add_input("S");
  reshape_node->add_output("Y");

  ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.ComputeShapeGraph(graph);

  const std::unordered_map<std::string, std::string> value_tags = {{"S", "shape"}};
  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, value_tags);
  inplace.WriteToMetadata(graph);

  // Node 0 (Shape) has no release and no in-place reuse, and X is still used
  // by the following node, so no metadata is written.
  EXPECT_EQ(graph.node()[0].metadata_props().size(), 0);

  // Node 1 (Reshape): releases S which is shape-tagged, and this is the last
  // use of declared graph input X.
  bool found_release_after = false;
  bool found_shape_tag = false;
  bool found_not_used_after = false;
  for (int i = 0; i < graph.node()[1].metadata_props().size(); ++i) {
    const auto &prop = graph.node()[1].metadata_props()[i];
    if (prop.key() == std::string(core::annotations::kReleaseAfterMetadataKey)) {
      EXPECT_EQ(prop.value(), std::string("S"));
      found_release_after = true;
    }
    if (prop.key() == std::string(core::annotations::kReleaseAfterShapeTagMetadataKey)) {
      EXPECT_EQ(prop.value(), std::string("S"));
      found_shape_tag = true;
    }
    if (prop.key() == std::string(core::annotations::kNotUsedAfterMetadataKey)) {
      EXPECT_EQ(prop.value(), std::string("X"));
      found_not_used_after = true;
    }
  }
  EXPECT_TRUE(found_release_after);
  EXPECT_TRUE(found_shape_tag);
  EXPECT_TRUE(found_not_used_after);
}

// Without value_tags, ReleaseAfterShapeTagged is all-empty and
// kReleaseAfterShapeTagMetadataKey is not written.
TEST(OnnxOptimInPlaceReuse, ComputeContextNoValueTagsYieldsEmptyShapeTagged) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"Y"});

  ShapesContext ctx;
  ctx.ComputeShapeGraph(graph);

  ComputeContext inplace;
  inplace.ComputeInPlaceReuseGraph(graph, ctx);
  inplace.WriteToMetadata(graph);

  ASSERT_EQ(inplace.Size(), 2u);
  // Without value_tags the shape-tagged vector is itself empty.
  EXPECT_TRUE(inplace.ReleaseAfterShapeTagged().empty());

  // kReleaseAfterShapeTagMetadataKey must not appear in metadata.
  for (int n = 0; n < graph.node().size(); ++n) {
    for (int i = 0; i < graph.node()[n].metadata_props().size(); ++i) {
      EXPECT_NE(std::string(graph.node()[n].metadata_props()[i].key()),
                std::string(core::annotations::kReleaseAfterShapeTagMetadataKey));
    }
  }
}

TEST(OnnxOptimInPlaceReuse, ComputeEventActionNames) {
  using core::annotations::ComputeEventAction;
  using core::annotations::ComputeEventActionName;
  EXPECT_STREQ(ComputeEventActionName(ComputeEventAction::kInPlace), "inplace");
  EXPECT_STREQ(ComputeEventActionName(ComputeEventAction::kRelease), "release");
  EXPECT_STREQ(ComputeEventActionName(ComputeEventAction::kReleaseShapeTag), "release_shape_tag");
}

TEST(OnnxOptimInPlaceReuse, ComputeContextDecisionEvents) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {2, 3});
  AddOutput(graph, "Y", {2, 3});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Shape", {"X"}, {"S"});
  *graph.add_node() = MakeNode("Reshape", {"A", "S"}, {"Y"});

  ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.ComputeShapeGraph(graph);

  ComputeContext inplace;
  EXPECT_FALSE(inplace.events_enabled());
  EXPECT_TRUE(inplace.Events().empty());
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, {{"S", "shape"}});
  EXPECT_TRUE(inplace.Events().empty());

  inplace.set_events_enabled(true);
  inplace.ClearEvents();
  inplace.ComputeInPlaceReuseGraph(graph, ctx, false, {{"S", "shape"}});

  using core::annotations::ComputeEventAction;
  int inplace_count = 0;
  int release_count = 0;
  int shape_release_count = 0;
  bool found_expected_inplace = false;
  for (const auto &ev : inplace.Events()) {
    if (ev.action == ComputeEventAction::kInPlace) {
      ++inplace_count;
      if (ev.node_index == 2 && ev.output_index == 0 && ev.input_index == 0 &&
          ev.kind == InPlaceReuseKind::kEqual) {
        found_expected_inplace = true;
      }
    } else if (ev.action == ComputeEventAction::kRelease) {
      ++release_count;
    } else if (ev.action == ComputeEventAction::kReleaseShapeTag) {
      ++shape_release_count;
      EXPECT_EQ(ev.name, "S");
      EXPECT_EQ(ev.node_index, 2);
    }
  }
  EXPECT_EQ(inplace_count, 1);
  EXPECT_EQ(release_count, 2);
  EXPECT_EQ(shape_release_count, 1);
  EXPECT_TRUE(found_expected_inplace);

  inplace.ClearEvents();
  EXPECT_TRUE(inplace.Events().empty());
}

// WriteInPlaceReuseToMetadata free function also accepts value_tags and writes
// kReleaseAfterShapeTagMetadataKey accordingly.
TEST(OnnxOptimInPlaceReuse, WriteInPlaceReuseToMetadataWithShapeTags) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {2, 3});
  AddOutput(graph, "Y", {2, 3});
  auto *shape_node = graph.add_node();
  shape_node->set_op_type("Shape");
  shape_node->add_input("X");
  shape_node->add_output("S");
  auto *reshape_node = graph.add_node();
  reshape_node->set_op_type("Reshape");
  reshape_node->add_input("X");
  reshape_node->add_input("S");
  reshape_node->add_output("Y");

  ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.ComputeShapeGraph(graph);

  const std::unordered_map<std::string, std::string> value_tags = {{"S", "shape"}};
  WriteInPlaceReuseToMetadata(graph, ctx, value_tags);

  bool found_shape_tag = false;
  for (int i = 0; i < graph.node()[1].metadata_props().size(); ++i) {
    if (std::string(graph.node()[1].metadata_props()[i].key()) ==
        std::string(core::annotations::kReleaseAfterShapeTagMetadataKey)) {
      EXPECT_EQ(std::string(graph.node()[1].metadata_props()[i].value()), std::string("S"));
      found_shape_tag = true;
    }
  }
  EXPECT_TRUE(found_shape_tag);
}

// ComputeContext::Compute runs shape inference, value tagging, in-place reuse
// and peak memory in one call, holding every result alive in the context.
TEST(OnnxOptimInPlaceReuse, ComputeContextComputeOrchestration) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ComputeContext ctx;
  ctx.Compute(graph);

  // Shapes were inferred and are held alive by the context.
  EXPECT_TRUE(ctx.Shapes().Has("A"));
  EXPECT_TRUE(ctx.Shapes().Has("Y"));
  // In-place reuse / release info was computed for every node.
  EXPECT_EQ(ctx.Size(), 3u);
  // Peak memory has one entry per node (Abs has no scratch, so all zero).
  ASSERT_EQ(ctx.PeakMemory().size(), 3u);
  EXPECT_EQ(ctx.NodePeakMemory(0), 0);
}

// ComputeContext::WriteToGraph pushes inferred shapes into value_info and the
// reuse / release info into node metadata.
TEST(OnnxOptimInPlaceReuse, ComputeContextWriteToGraph) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ComputeContext ctx;
  ctx.Compute(graph);
  ctx.WriteToGraph(graph);

  // Inferred shape for the intermediate "A" is now recorded in value_info.
  bool found_value_info = false;
  for (const ValueInfoProto &vi : graph.value_info()) {
    if (std::string(vi.name()) == std::string("A")) {
      found_value_info = true;
    }
  }
  EXPECT_TRUE(found_value_info);

  // The release-after metadata is present on the node freeing "A".
  bool found_release = false;
  for (int i = 0; i < graph.node()[1].metadata_props().size(); ++i) {
    if (std::string(graph.node()[1].metadata_props()[i].key()) ==
        std::string(core::annotations::kReleaseAfterMetadataKey)) {
      found_release = true;
    }
  }
  EXPECT_TRUE(found_release);
}

// ComputeContext::BuildExecutionPlan derives the execution plan from the
// results held by the context.
TEST(OnnxOptimInPlaceReuse, ComputeContextBuildExecutionPlan) {
  GraphProto graph;
  graph.set_name("g");
  AddInput(graph, "X", {3, 4});
  AddOutput(graph, "Y", {3, 4});
  *graph.add_node() = MakeNode("Abs", {"X"}, {"A"});
  *graph.add_node() = MakeNode("Abs", {"A"}, {"B"});
  *graph.add_node() = MakeNode("Abs", {"B"}, {"Y"});

  ComputeContext ctx;
  ctx.Compute(graph);
  core::runtime::ExecutionPlan plan = ctx.BuildExecutionPlan(graph);

  EXPECT_EQ(plan.num_nodes(), 3u);
  EXPECT_FALSE(plan.actions().empty());
  // Declared input / output are kept and never released.
  EXPECT_EQ(plan.keep().count("X"), 1u);
  EXPECT_EQ(plan.keep().count("Y"), 1u);
}

} // namespace Test
