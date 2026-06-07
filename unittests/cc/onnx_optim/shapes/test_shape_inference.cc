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

TEST(OnnxOptimShapeInference, SupportsUtilsStringLookupForNodeOutput) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  const auto &output_name = node.output(0);
  ASSERT_TRUE(ctx.Has(output_name));
  EXPECT_EQ(ctx.Get(output_name),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
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

TEST(OnnxOptimShapeInference, DispatchesMulWithBroadcast) {
  NodeProto node = MakeNode("Mul", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                 onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(5)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
}

TEST(OnnxOptimShapeInference, DispatchesDivWithBroadcast) {
  NodeProto node = MakeNode("Div", {"A", "B"}, {"C"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                 onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(5)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
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

TEST(OnnxOptimShapeInference, DispatchesCos) {
  NodeProto node = MakeNode("Cos", {"X"}, {"Y"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, DispatchesCosh) {
  NodeProto node = MakeNode("Cosh", {"X"}, {"Y"});
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

TEST(OnnxOptimShapeInference, DispatchesReduceMaxNoAxesInputReducesAll) {
  NodeProto node = MakeNode("ReduceMax", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "keepdims", 0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeInference, DispatchesReduceMinWithAxesInputValueAsShape) {
  NodeProto node = MakeNode("ReduceMin", {"X", "axes"}, {"Y"});
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

TEST(OnnxOptimShapeInference, DispatchesAffineGrid2DWithConstantSize) {
  NodeProto node = MakeNode("AffineGrid", {"theta", "size"}, {"grid"});

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3)}));
  onnx_optim::OptimTensor size_t(nullptr, onnx_optim::TensorType::kInt64,
                                 onnx_optim::OptimShape{onnx_optim::OptimDim(4)});
  size_t.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                                onnx_optim::OptimDim(5), onnx_optim::OptimDim(6)});
  ctx.Set("size", std::move(size_t));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("grid"));
  EXPECT_EQ(ctx.Get("grid").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("grid").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5),
                                    onnx_optim::OptimDim(6), onnx_optim::OptimDim(2)}));
}

// ── ComputeShapeGraph / ComputeShapeModel ─────────────────────────────

namespace {

// Sets the tensor-type and shape of a ValueInfoProto in one call. A
// shape entry with ``dim_value < 0`` is encoded as a symbolic dim
// parameter taken from ``symbolic_names[index]``; non-negative
// entries become concrete ``dim_value`` dimensions.
void SetValueInfoTensorType(ValueInfoProto &vi, TensorProto::DataType dtype,
                            const std::vector<int64_t> &shape,
                            const std::vector<std::string> &symbolic_names = {}) {
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    TensorShapeProto::Dimension *d = sp->add_dim();
    if (shape[i] < 0) {
      d->set_dim_param(i < symbolic_names.size() ? symbolic_names[i] : std::string("?"));
    } else {
      d->set_dim_value(shape[i]);
    }
  }
}

// Builds a Constant node whose ``value_ints`` attribute carries the
// given dims (used as the ``shape`` input of a Reshape).
NodeProto MakeConstantValueIntsNode(const std::string &out, const std::vector<int64_t> &dims) {
  NodeProto node = MakeNode("Constant", {}, {out});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("value_ints");
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : dims) {
    attr->add_ints(v);
  }
  return node;
}

// Builds the Reshape model used by the tests:
//   X --(input)--> Reshape --(output)--> Y
//                    ^
//                    |
//          Constant(value_ints=target)
// ``input_shape`` describes X (negative entries denote symbolic
// dims), ``target`` is the literal Constant payload.
ModelProto MakeReshapeWithConstantModel(const std::vector<int64_t> &input_shape,
                                        const std::vector<int64_t> &target,
                                        const std::vector<std::string> &symbolic_names = {}) {
  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  ValueInfoProto *in = graph->add_input();
  in->set_name("X");
  SetValueInfoTensorType(*in, TensorProto::DataType::FLOAT, input_shape, symbolic_names);
  ValueInfoProto *out = graph->add_output();
  out->set_name("Y");
  *graph->add_node() = MakeConstantValueIntsNode("S", target);
  *graph->add_node() = MakeNode("Reshape", {"X", "S"}, {"Y"});
  return model;
}

} // namespace

TEST(OnnxOptimShapeInference, ComputeShapeModelReshapeStaticShape) {
  // Reshape(X, Constant([-1, 2])) with a fully-static input shape.
  // The -1 must be back-filled from the input element count (3 * 4 / 2 = 6).
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{3, 4}, /*target=*/{-1, 2});
  onnx_optim::shapes::ShapesContext ctx;

  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  ASSERT_TRUE(ctx.Has("X"));
  EXPECT_EQ(ctx.Get("X").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));
  ASSERT_TRUE(ctx.Has("S"));
  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(6), onnx_optim::OptimDim(2)}));
  // The model's opset import should have been recorded in ``ctx``.
  EXPECT_EQ(ctx.OpsetVersion(""), 18);
}

TEST(OnnxOptimShapeInference, ComputeShapeModelReshapeDynamicShape) {
  // Reshape(X, Constant([-1, 2])) where X has a symbolic first dim:
  // the -1 cannot be evaluated to a concrete integer and must remain
  // symbolic. The second dim is forwarded from the Constant.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  onnx_optim::shapes::ShapesContext ctx;

  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  ASSERT_TRUE(ctx.Has("X"));
  ASSERT_EQ(ctx.Get("X").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("X").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("X").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("X").Shape()[1], onnx_optim::OptimDim(4));
  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], onnx_optim::OptimDim(2));
}

TEST(OnnxOptimShapeInference, ComputeShapeModelPrefillPrefersOutputAnchor) {
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 2}, /*symbolic_names=*/{"ANCHOR"});

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model, /*prefill_with_value_info_output=*/true);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "ANCHOR");
  EXPECT_EQ(ctx.Get("Y").Shape()[1], onnx_optim::OptimDim(2));
}

TEST(OnnxOptimShapeInference, ComputeShapeModelSeedsInitializerAsShape) {
  // The Reshape ``shape`` input is provided as a graph initializer
  // (i.e. weight-style data) rather than as a Constant node. The
  // value-as-shape annotation must be reconstructed from the
  // initializer's int64 payload so that the resulting Y shape is
  // concrete.
  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  ValueInfoProto *in = graph->add_input();
  in->set_name("X");
  SetValueInfoTensorType(*in, TensorProto::DataType::FLOAT, /*shape=*/{3, 4});
  ValueInfoProto *out = graph->add_output();
  out->set_name("Y");
  TensorProto *init = graph->add_initializer();
  init->set_name("S");
  init->set_data_type(TensorProto::DataType::INT64);
  init->add_dims(std::vector<uint64_t>{2});
  init->add_int64_data(std::vector<int64_t>{-1, 2});
  *graph->add_node() = MakeNode("Reshape", {"X", "S"}, {"Y"});

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  ASSERT_TRUE(ctx.Has("S"));
  EXPECT_TRUE(ctx.Get("S").HasValueAsShape());
  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(6), onnx_optim::OptimDim(2)}));
}

TEST(OnnxOptimShapeInference, ComputeShapeModelRejectsModelWithoutGraph) {
  ModelProto model;
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeModel(ctx, model), std::invalid_argument);
}

// ── ApplyInferredShapesTo{Graph,Model} ────────────────────────────────

TEST(OnnxOptimShapeInference, ApplyInferredShapesToModelFillsOutputAndValueInfo) {
  // Reshape(X, Constant([-1, 2])) with a fully-static input shape.
  // After ApplyInferredShapesToModel, the Y output of the graph must
  // carry the inferred {6, 2} shape and the intermediate S tensor
  // must appear in graph.value_info().
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{3, 4}, /*target=*/{-1, 2});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);
  onnx_optim::shapes::ApplyInferredShapesToModel(ctx, model);

  ASSERT_TRUE(model.has_graph());
  const GraphProto &graph = model.graph();
  // The Y output has type float and shape {6, 2}.
  ASSERT_EQ(graph.output_size(), 1u);
  const ValueInfoProto &out = graph.output(0);
  EXPECT_EQ(out.name().as_string(), "Y");
  ASSERT_TRUE(out.has_type());
  ASSERT_TRUE(out.type().has_tensor_type());
  EXPECT_EQ(static_cast<int>(out.type().tensor_type().elem_type()),
            static_cast<int>(TensorProto::DataType::FLOAT));
  ASSERT_TRUE(out.type().tensor_type().has_shape());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[0].dim_value(), 6);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 2);
  // The intermediate S tensor (Constant output) ends up in value_info.
  bool found_s = false;
  for (std::size_t i = 0; i < graph.value_info_size(); ++i) {
    const ValueInfoProto &vi = graph.value_info()[i];
    if (vi.name().as_string() == "S") {
      found_s = true;
      ASSERT_TRUE(vi.has_type());
      ASSERT_TRUE(vi.type().has_tensor_type());
      EXPECT_EQ(static_cast<int>(vi.type().tensor_type().elem_type()),
                static_cast<int>(TensorProto::DataType::INT64));
    }
  }
  EXPECT_TRUE(found_s);
  // The X input is not duplicated in value_info.
  for (std::size_t i = 0; i < graph.value_info_size(); ++i) {
    EXPECT_NE(graph.value_info()[i].name().as_string(), std::string("X"));
  }
}

TEST(OnnxOptimShapeInference, ApplyInferredShapesToModelPreservesSymbolicDims) {
  // Reshape(X, Constant([-1, 2])) with a symbolic first input dim:
  // the inferred Y output must keep the first dim as a dim_param.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);
  onnx_optim::shapes::ApplyInferredShapesToModel(ctx, model);

  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.has_type() && out.type().has_tensor_type());
  ASSERT_TRUE(out.type().tensor_type().has_shape());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  // First dim should be symbolic (dim_param), second should be 2.
  EXPECT_FALSE(out.type().tensor_type().shape().dim()[0].has_dim_value());
  EXPECT_TRUE(out.type().tensor_type().shape().dim()[0].has_dim_param());
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 2);
}

TEST(OnnxOptimShapeInference, ApplyInferredShapesToModelRejectsModelWithoutGraph) {
  ModelProto model;
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::ApplyInferredShapesToModel(ctx, model), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, InferShapesModelEndToEnd) {
  // Convenience wrapper: runs ComputeShapeModel + ApplyInferredShapesToModel.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{3, 4}, /*target=*/{-1, 2});
  onnx_optim::shapes::InferShapesModel(model);

  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.has_type() && out.type().has_tensor_type());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[0].dim_value(), 6);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 2);
}

TEST(OnnxOptimShapeInference, InferShapesModelWithPrefillPrefersOutputAnchor) {
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 2}, /*symbolic_names=*/{"ANCHOR"});

  onnx_optim::shapes::InferShapesModel(model, /*prefill_with_value_info_output=*/true);

  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.has_type() && out.type().has_tensor_type());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  EXPECT_TRUE(out.type().tensor_type().shape().dim()[0].has_dim_param());
  EXPECT_EQ(out.type().tensor_type().shape().dim()[0].dim_param().as_string(), "ANCHOR");
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 2);
}

// ── Model-local functions ────────────────────────────────────────────

namespace {

// Builds ``model.functions()`` entry ``local:func_add(a, b) -> c`` whose
// body is a single ``Add`` node.
FunctionProto *AddLocalFuncAdd(ModelProto &model) {
  FunctionProto *func = model.add_functions();
  func->set_name("func_add");
  func->set_domain("local");
  func->add_input("a");
  func->add_input("b");
  func->add_output("c");
  OperatorSetIdProto *opset = func->add_opset_import();
  opset->set_domain("");
  opset->set_version(static_cast<int64_t>(18));
  NodeProto *body = func->add_node();
  body->set_op_type("Add");
  body->add_input("a");
  body->add_input("b");
  body->add_output("c");
  return func;
}

// Builds a minimal model:
//   inputs X, Y (float, shape {3, 4}); output Z = local:func_add(X, Y).
ModelProto MakeLocalFunctionAddModel() {
  ModelProto model;
  model.set_ir_version(static_cast<int64_t>(8));
  OperatorSetIdProto *ai = model.add_opset_import();
  ai->set_domain("");
  ai->set_version(static_cast<int64_t>(18));
  OperatorSetIdProto *loc = model.add_opset_import();
  loc->set_domain("local");
  loc->set_version(static_cast<int64_t>(1));
  AddLocalFuncAdd(model);

  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  ValueInfoProto *x = graph->add_input();
  x->set_name("X");
  SetValueInfoTensorType(*x, TensorProto::DataType::FLOAT, /*shape=*/{3, 4});
  ValueInfoProto *y = graph->add_input();
  y->set_name("Y");
  SetValueInfoTensorType(*y, TensorProto::DataType::FLOAT, /*shape=*/{3, 4});
  ValueInfoProto *z = graph->add_output();
  z->set_name("Z");
  // Leave Z's type empty so shape inference must recover it.
  z->add_type();

  NodeProto *call = graph->add_node();
  call->set_op_type("func_add");
  call->set_domain("local");
  call->add_input("X");
  call->add_input("Y");
  call->add_output("Z");
  return model;
}

} // namespace

TEST(OnnxOptimShapeInference, ComputeShapeModelExpandsLocalFunctionCall) {
  // A node calling the model-local function ``local:func_add`` must be
  // expanded: the function body's ``Add`` is run in a sub-context that
  // rebinds the function-local names ``a``/``b``/``c`` to the caller's
  // ``X``/``Y``/``Z``, and the inferred output is mapped back.
  ModelProto model = MakeLocalFunctionAddModel();
  onnx_optim::shapes::ShapesContext ctx;

  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));
  // The local-function map should have been populated from model.functions().
  EXPECT_TRUE(ctx.HasLocalFunction("local:func_add"));
}

TEST(OnnxOptimShapeInference, InferShapesModelWritesBackLocalFunctionOutput) {
  // End-to-end: ``InferShapesModel`` writes the inferred Z shape back to
  // the proto.
  ModelProto model = MakeLocalFunctionAddModel();

  onnx_optim::shapes::InferShapesModel(model);

  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.has_type() && out.type().has_tensor_type());
  ASSERT_TRUE(out.type().tensor_type().has_shape());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[0].dim_value(), 3);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 4);
}

TEST(OnnxOptimShapeInference, ComputeShapeNodeRejectsUnknownNonLocalFunctionDomain) {
  // Sanity check: a node in an unknown domain that does **not** match any
  // registered model-local function still triggers the domain check.
  NodeProto node = MakeNode("does_not_exist", {"X"}, {"Y"}, "com.acme");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

} // namespace Test
