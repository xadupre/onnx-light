// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const std::string &domain = "") {
  NodeProto node;
  node.set_op_type(op_type);
  if (!domain.empty()) {
    node.set_domain(domain);
  }
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  for (const auto &out : outputs) {
    node.add_output(out);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapeInference, DispatchesAbs) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, DispatchesAddWithBroadcast) {
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeInference, DispatchesAnd) {
  NodeProto node = MakeNode("And", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapeInference, DispatchesAcos) {
  NodeProto node = MakeNode("Acos", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, DispatchesAcosh) {
  NodeProto node = MakeNode("Acosh", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapeInference, DispatchesConcat) {
  NodeProto node = MakeNode("Concat", {"A", "B"}, {"C"});
  AddAttribute<int64_t>(node, "axis", 0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(6), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapeInference, AcceptsExplicitAiOnnxDomain) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"}, "ai.onnx");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kFloat,
                                onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  ctx.Set("X", std::move(input));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  EXPECT_TRUE(ctx.Has("Y"));
}

TEST(OnnxOptimShapeInference, RejectsUnknownDomain) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"}, "com.acme");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, RejectsUnsupportedOpType) {
  NodeProto node = MakeNode("NoSuchOp", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, RejectsNodeWithMissingInputs) {
  NodeProto node = MakeNode("Add", {"A"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, ComputeShapesProcessesNodesInOrder) {
  // Build a graph proto with three nodes:
  //   T = Add(A, B)
  //   U = Abs(T)
  //   V = And(M, N)
  GraphProto graph;
  *graph.add_node() = MakeNode("Add", {"A", "B"}, {"T"});
  *graph.add_node() = MakeNode("Abs", {"T"}, {"U"});
  *graph.add_node() = MakeNode("And", {"M", "N"}, {"V"});

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_ab{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_ab));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_ab));
  onnx_optim::OptimShape shape_bool{onnx_optim::OptimDim(4)};
  ctx.Set("M", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_bool));
  ctx.Set("N", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_bool));

  onnx_optim::shapes::ComputeShapes(ctx, graph.node());

  ASSERT_TRUE(ctx.Has("T"));
  ASSERT_TRUE(ctx.Has("U"));
  ASSERT_TRUE(ctx.Has("V"));
  EXPECT_EQ(ctx.Get("T").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("T").Shape(), shape_ab);
  EXPECT_EQ(ctx.Get("U").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("U").Shape(), shape_ab);
  EXPECT_EQ(ctx.Get("V").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("V").Shape(), shape_bool);
}

TEST(OnnxOptimShapeInference, ComputeShapesEmptyListIsNoOp) {
  GraphProto graph;
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_NO_THROW(onnx_optim::shapes::ComputeShapes(ctx, graph.node()));
  EXPECT_TRUE(ctx.Empty());
}

TEST(OnnxOptimShapeInference, ComputeShapesPropagatesErrorFromNode) {
  GraphProto graph;
  *graph.add_node() = MakeNode("Abs", {"X"}, {"Y"});
  *graph.add_node() = MakeNode("NoSuchOp", {"Y"}, {"Z"});

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));

  EXPECT_THROW(onnx_optim::shapes::ComputeShapes(ctx, graph.node()), std::invalid_argument);
  // The first node should still have been processed before the error.
  EXPECT_TRUE(ctx.Has("Y"));
}

// ── CheckInputsAvailable / CheckOutputsNotAvailable ────────────────────

TEST(OnnxOptimShapeInferenceChecks, InputsAvailableAcceptsWhenAllPresent) {
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_NO_THROW(onnx_optim::shapes::CheckInputsAvailable(ctx, node));
}

TEST(OnnxOptimShapeInferenceChecks, InputsAvailableSkipsEmptyOptionalInputs) {
  // The middle input is an empty string: ONNX uses that to signal an
  // optional input that is not provided. The check must ignore it.
  NodeProto node = MakeNode("SomeOp", {"A", "", "C"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("C", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_NO_THROW(onnx_optim::shapes::CheckInputsAvailable(ctx, node));
}

TEST(OnnxOptimShapeInferenceChecks, InputsAvailableThrowsWhenMissing) {
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::CheckInputsAvailable(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInferenceChecks, OutputsNotAvailableAcceptsWhenAllAbsent) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_NO_THROW(onnx_optim::shapes::CheckOutputsNotAvailable(ctx, node));
}

TEST(OnnxOptimShapeInferenceChecks, OutputsNotAvailableSkipsEmptyOptionalOutputs) {
  // ONNX uses an empty output name to mark an optional output that is
  // not produced by the node; the check must ignore it.
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y", ""});
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_NO_THROW(onnx_optim::shapes::CheckOutputsNotAvailable(ctx, node));
}

TEST(OnnxOptimShapeInferenceChecks, OutputsNotAvailableThrowsWhenPresent) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::CheckOutputsNotAvailable(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInferenceChecks, ComputeShapeNodeRejectsMissingInput) {
  // Going through the dispatcher: input "B" is missing from ctx, so
  // CheckInputsAvailable should throw std::invalid_argument before
  // dispatch.
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInferenceChecks, ComputeShapeNodeRejectsAlreadyComputedOutput) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesReduceSumNoAxesInputReducesAll) {
  // opset >= 13: axes input is omitted -> default behaviour reduces all axes.
  NodeProto node = MakeNode("ReduceSum", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "keepdims", 0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  // Scalar (rank-0) output because every dim is reduced and keepdims=0.
  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeInference, DispatchesReduceSumWithAxesInputValueAsShape) {
  // opset >= 13 with an axes input carrying ValueAsShape (a constant).
  NodeProto node = MakeNode("ReduceSum", {"X", "axes"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  onnx_optim::OptimTensor axes_tensor(nullptr, onnx_optim::TensorType::kInt64,
                                      onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  axes_tensor.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  ctx.Set("axes", std::move(axes_tensor));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  // keepdims defaults to 1: rank preserved, axis 1 collapsed to 1.
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapeInference, DispatchesRoiAlign) {
  NodeProto node = MakeNode("RoiAlign", {"X", "rois", "batch_indices"}, {"Y"});
  AddAttribute<int64_t>(node, "output_height", 7);
  AddAttribute<int64_t>(node, "output_width", 7);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(256),
                                          onnx_optim::OptimDim(38), onnx_optim::OptimDim(50)}));
  ctx.Set("rois", onnx_optim::OptimTensor(
                      nullptr, onnx_optim::TensorType::kFloat,
                      onnx_optim::OptimShape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(4)}));
  ctx.Set("batch_indices",
          onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                  onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(256),
                                    onnx_optim::OptimDim(7), onnx_optim::OptimDim(7)}));
}

} // namespace Test
