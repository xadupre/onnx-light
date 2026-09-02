// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_inference.h"

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, SupportsUtilsStringLookupForNodeOutput) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  const auto &output_name = node.output(0);
  ASSERT_TRUE(ctx.Has(output_name));
  EXPECT_EQ(ctx.Get(output_name),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, InfersPartialSTFTShapeWithDynamicSignalLength) {
  NodeProto node = MakeNode("STFT", {"signal", "frame_step", "window"}, {"output"});
  core::shapes::ShapesContext ctx;
  int64_t frame_step = 2;
  ctx.Set("signal", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              {core::symbolic::SymDim("batch"),
                                               core::symbolic::SymDim("signal_length"),
                                               core::symbolic::SymDim(1)}));
  ctx.Set("frame_step",
          core::symbolic::SymTensor(&frame_step, core::symbolic::TensorType::kInt64, {}));
  ctx.Set("window", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              {core::symbolic::SymDim(5)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("output"));
  const core::symbolic::SymShape &shape = ctx.Get("output").Shape();
  ASSERT_EQ(shape.Rank(), 4);
  EXPECT_EQ(shape[0], core::symbolic::SymDim("batch"));
  EXPECT_FALSE(shape[1].IsInt());
  EXPECT_EQ(shape[2], core::symbolic::SymDim(3));
  EXPECT_EQ(shape[3], core::symbolic::SymDim(2));
}

TEST(OnnxOptimShapeInference, STFTDefaultsFrameLengthToDynamicSignalLength) {
  NodeProto node = MakeNode("STFT", {"signal", "frame_step"}, {"output"});
  core::shapes::ShapesContext ctx;
  int64_t frame_step = 2;
  ctx.Set("signal", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              {core::symbolic::SymDim("batch"),
                                               core::symbolic::SymDim("signal_length"),
                                               core::symbolic::SymDim(1)}));
  ctx.Set("frame_step",
          core::symbolic::SymTensor(&frame_step, core::symbolic::TensorType::kInt64, {}));

  ctx.ComputeShapeNode(node);

  const core::symbolic::SymShape &shape = ctx.Get("output").Shape();
  ASSERT_EQ(shape.Rank(), 4);
  EXPECT_EQ(shape[0], core::symbolic::SymDim("batch"));
  EXPECT_EQ(shape[1], core::symbolic::SymDim(1));
  EXPECT_FALSE(shape[2].IsInt());
  EXPECT_EQ(shape[3], core::symbolic::SymDim(2));
}

TEST(OnnxOptimShapeInference, STFTRejectsOneSidedComplexSignal) {
  NodeProto node = MakeNode("STFT", {"signal", "frame_step"}, {"output"});
  core::shapes::ShapesContext ctx;
  ctx.Set("signal",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              {core::symbolic::SymDim(1), core::symbolic::SymDim(10), core::symbolic::SymDim(2)}));
  ctx.Set("frame_step", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {}));

  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesAddWithBroadcast) {
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeInference, DispatchesMulWithBroadcast) {
  NodeProto node = MakeNode("Mul", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                   core::symbolic::SymDim(5)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(5)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
}

TEST(OnnxOptimShapeInference, DispatchesDivWithBroadcast) {
  NodeProto node = MakeNode("Div", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(3), core::symbolic::SymDim(4),
                                   core::symbolic::SymDim(5)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(5)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
}

TEST(OnnxOptimShapeInference, DispatchesAnd) {
  NodeProto node = MakeNode("And", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapeInference, DispatchesAcos) {
  NodeProto node = MakeNode("Acos", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, DispatchesAcosh) {
  NodeProto node = MakeNode("Acosh", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapeInference, DispatchesCos) {
  NodeProto node = MakeNode("Cos", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapeInference, DispatchesCosh) {
  NodeProto node = MakeNode("Cosh", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapeInference, DispatchesConcat) {
  NodeProto node = MakeNode("Concat", {"A", "B"}, {"C"});
  AddAttribute<int64_t>(node, "axis", 0);
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(3)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(6), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapeInference, AcceptsExplicitAiOnnxDomain) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"}, "ai.onnx");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymTensor input(nullptr, core::symbolic::TensorType::kFloat,
                                  core::symbolic::SymShape{core::symbolic::SymDim(2)});
  ctx.Set("X", std::move(input));

  ctx.ComputeShapeNode(node);

  EXPECT_TRUE(ctx.Has("Y"));
}

TEST(OnnxOptimShapeInference, RejectsUnknownDomain) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"}, "com.acme");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, RejectsUnsupportedOpType) {
  NodeProto node = MakeNode("NoSuchOp", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, RejectsNodeWithMissingInputs) {
  NodeProto node = MakeNode("Add", {"A"}, {"C"});
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
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

  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_ab{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_ab));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_ab));
  core::symbolic::SymShape shape_bool{core::symbolic::SymDim(4)};
  ctx.Set("M", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_bool));
  ctx.Set("N", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_bool));

  ctx.ComputeShapes(graph.node());

  ASSERT_TRUE(ctx.Has("T"));
  ASSERT_TRUE(ctx.Has("U"));
  ASSERT_TRUE(ctx.Has("V"));
  EXPECT_EQ(ctx.Get("T").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("T").Shape(), shape_ab);
  EXPECT_EQ(ctx.Get("U").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("U").Shape(), shape_ab);
  EXPECT_EQ(ctx.Get("V").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("V").Shape(), shape_bool);
}

TEST(OnnxOptimShapeInference, ComputeShapesEmptyListIsNoOp) {
  GraphProto graph;
  core::shapes::ShapesContext ctx;
  EXPECT_NO_THROW(ctx.ComputeShapes(graph.node()));
  EXPECT_TRUE(ctx.Empty());
}

TEST(OnnxOptimShapeInference, ComputeShapesPropagatesErrorFromNode) {
  GraphProto graph;
  *graph.add_node() = MakeNode("Abs", {"X"}, {"Y"});
  *graph.add_node() = MakeNode("NoSuchOp", {"Y"}, {"Z"});

  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));

  EXPECT_THROW(ctx.ComputeShapes(graph.node()), std::invalid_argument);
  // The first node should still have been processed before the error.
  EXPECT_TRUE(ctx.Has("Y"));
}

// ── CheckInputsAvailable / CheckOutputsNotAvailable ────────────────────

TEST(OnnxOptimShapeInferenceChecks, InputsAvailableAcceptsWhenAllPresent) {
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_NO_THROW(ctx.CheckInputsAvailable(node));
}

TEST(OnnxOptimShapeInferenceChecks, InputsAvailableSkipsEmptyOptionalInputs) {
  // The middle input is an empty string: ONNX uses that to signal an
  // optional input that is not provided. The check must ignore it.
  NodeProto node = MakeNode("SomeOp", {"A", "", "C"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("C", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_NO_THROW(ctx.CheckInputsAvailable(node));
}

TEST(OnnxOptimShapeInferenceChecks, InputsAvailableThrowsWhenMissing) {
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.CheckInputsAvailable(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInferenceChecks, OutputsNotAvailableAcceptsWhenAllAbsent) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_NO_THROW(ctx.CheckOutputsNotAvailable(node));
}

TEST(OnnxOptimShapeInferenceChecks, OutputsNotAvailableSkipsEmptyOptionalOutputs) {
  // ONNX uses an empty output name to mark an optional output that is
  // not produced by the node; the check must ignore it.
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y", ""});
  core::shapes::ShapesContext ctx;
  EXPECT_NO_THROW(ctx.CheckOutputsNotAvailable(node));
}

TEST(OnnxOptimShapeInferenceChecks, OutputsNotAvailableThrowsWhenPresent) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.CheckOutputsNotAvailable(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInferenceChecks, ComputeShapeNodeRejectsMissingInput) {
  // Going through the dispatcher: input "B" is missing from ctx, so
  // CheckInputsAvailable should throw std::invalid_argument before
  // dispatch.
  NodeProto node = MakeNode("Add", {"A", "B"}, {"C"});
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInferenceChecks, ComputeShapeNodeRejectsAlreadyComputedOutput) {
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesReduceSumNoAxesInputReducesAll) {
  // opset >= 13: axes input is omitted -> default behaviour reduces all axes.
  NodeProto node = MakeNode("ReduceSum", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "keepdims", 0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  // Scalar (rank-0) output because every dim is reduced and keepdims=0.
  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeInference, DispatchesReduceSumWithAxesInputValueAsShape) {
  // opset >= 13 with an axes input carrying ValueAsShape (a constant).
  NodeProto node = MakeNode("ReduceSum", {"X", "axes"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  core::symbolic::SymTensor axes_tensor(nullptr, core::symbolic::TensorType::kInt64,
                                        core::symbolic::SymShape{core::symbolic::SymDim(1)});
  axes_tensor.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(1)});
  ctx.Set("axes", std::move(axes_tensor));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  // keepdims defaults to 1: rank preserved, axis 1 collapsed to 1.
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapeInference, DispatchesReduceMaxNoAxesInputReducesAll) {
  NodeProto node = MakeNode("ReduceMax", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "keepdims", 0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapeInference, DispatchesReduceMinWithAxesInputValueAsShape) {
  NodeProto node = MakeNode("ReduceMin", {"X", "axes"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));
  core::symbolic::SymTensor axes_tensor(nullptr, core::symbolic::TensorType::kInt64,
                                        core::symbolic::SymShape{core::symbolic::SymDim(1)});
  axes_tensor.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(1)});
  ctx.Set("axes", std::move(axes_tensor));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapeInference, DispatchesRoiAlign) {
  NodeProto node = MakeNode("RoiAlign", {"X", "rois", "batch_indices"}, {"Y"});
  AddAttribute<int64_t>(node, "output_height", 7);
  AddAttribute<int64_t>(node, "output_width", 7);

  core::shapes::ShapesContext ctx;
  ctx.Set("X",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(256),
                                       core::symbolic::SymDim(38), core::symbolic::SymDim(50)}));
  ctx.Set("rois", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                            core::symbolic::SymShape{core::symbolic::SymDim(5),
                                                                     core::symbolic::SymDim(4)}));
  ctx.Set("batch_indices",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{core::symbolic::SymDim(5)}));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(5), core::symbolic::SymDim(256),
                                      core::symbolic::SymDim(7), core::symbolic::SymDim(7)}));
}

TEST(OnnxOptimShapeInference, DispatchesAffineGrid2DWithConstantSize) {
  NodeProto node = MakeNode("AffineGrid", {"theta", "size"}, {"grid"});

  core::shapes::ShapesContext ctx;
  ctx.Set("theta", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(3)}));
  core::symbolic::SymTensor size_t(nullptr, core::symbolic::TensorType::kInt64,
                                   core::symbolic::SymShape{core::symbolic::SymDim(4)});
  size_t.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                               core::symbolic::SymDim(5), core::symbolic::SymDim(6)});
  ctx.Set("size", std::move(size_t));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("grid"));
  EXPECT_EQ(ctx.Get("grid").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("grid").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(5),
                                      core::symbolic::SymDim(6), core::symbolic::SymDim(2)}));
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
      if (i < symbolic_names.size() && !symbolic_names[i].empty()) {
        d->set_dim_param(symbolic_names[i]);
      }
      // Otherwise leave the dim unset (no name information).
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
  core::shapes::ShapesContext ctx;

  ctx.ComputeShapeModel(model);

  ASSERT_TRUE(ctx.Has("X"));
  EXPECT_EQ(ctx.Get("X").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));
  ASSERT_TRUE(ctx.Has("S"));
  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(6), core::symbolic::SymDim(2)}));
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
  core::shapes::ShapesContext ctx;

  ctx.ComputeShapeModel(model);

  ASSERT_TRUE(ctx.Has("X"));
  ASSERT_EQ(ctx.Get("X").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("X").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("X").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("X").Shape()[1], core::symbolic::SymDim(4));
  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(2));
}

TEST(OnnxOptimShapeInference, ComputeShapeModelPrefillPrefersOutputAnchor) {
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 2}, /*symbolic_names=*/{"ANCHOR"});

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model, /*prefill_with_value_info_output=*/true);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "ANCHOR");
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(2));
  // The inferred symbol and the anchor symbol "ANCHOR" are recorded
  // as an equality constraint so downstream passes can unify them.
  EXPECT_EQ(ctx.ConstraintsSize(), 1u);
  const auto &constraints = ctx.Constraints();
  // One side of the recorded pair must be the anchor symbol.
  ASSERT_EQ(constraints.size(), 1u);
  const auto &c = *constraints.begin();
  EXPECT_TRUE(c.first == "ANCHOR" || c.second == "ANCHOR");
}

TEST(OnnxOptimShapeInference, ComputeShapeModelPrefillPreservesGraphInputSymbol) {
  // Regression test: when an output anchor uses a different symbolic name
  // than the input ("ANCHOR" vs "N"), the prefill+propagate pass must not
  // rename the graph input dim from "N" to "ANCHOR". Both names are
  // user-provided and authoritative for their own value.
  //
  // ``Y = Relu(X)`` keeps the input shape, so the inferred Y has dim
  // ``N`` and the anchor declares ``ANCHOR`` for the same position. The
  // merge records the equality ``N == ANCHOR``; the propagation step
  // must privilege the anchor on Y but leave X alone.
  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  ValueInfoProto *in = graph->add_input();
  in->set_name("X");
  SetValueInfoTensorType(*in, TensorProto::DataType::FLOAT, /*shape=*/{-1, 4},
                         /*symbolic_names=*/{"N"});
  ValueInfoProto *out = graph->add_output();
  out->set_name("Y");
  SetValueInfoTensorType(*out, TensorProto::DataType::FLOAT, /*shape=*/{-1, 4},
                         /*symbolic_names=*/{"ANCHOR"});
  *graph->add_node() = MakeNode("Relu", {"X"}, {"Y"});

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model, /*prefill_with_value_info_output=*/true);

  // Y adopts the anchor symbol.
  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "ANCHOR");
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(4));
  // The graph-input symbol "N" survives propagation untouched, even
  // though the constraint ``N == ANCHOR`` was recorded.
  ASSERT_TRUE(ctx.Has("X"));
  ASSERT_EQ(ctx.Get("X").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("X").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("X").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("X").Shape()[1], core::symbolic::SymDim(4));
}

TEST(OnnxOptimShapeInference, ComputeShapeModelPrefillRaisesOnDimConflict) {
  // Reshape with target [-1, 2] applied to X[N, 4] produces Y[?, 2].
  // The anchor declares Y as ["ANCHOR", 4] — the trailing concrete dim
  // (4 vs 2) is incompatible and must raise.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 4}, /*symbolic_names=*/{"ANCHOR"});

  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ComputeShapeModel(model,
                                     /*prefill_with_value_info_output=*/true),
               std::invalid_argument);
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
  init->add_dims(std::vector<int64_t>{2});
  init->add_int64_data(std::vector<int64_t>{-1, 2});
  *graph->add_node() = MakeNode("Reshape", {"X", "S"}, {"Y"});

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model);

  ASSERT_TRUE(ctx.Has("S"));
  EXPECT_TRUE(ctx.Get("S").HasValueAsShape());
  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(6), core::symbolic::SymDim(2)}));
}

TEST(OnnxOptimShapeInference, ComputeShapeModelRejectsModelWithoutGraph) {
  ModelProto model;
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ComputeShapeModel(model), std::invalid_argument);
}

// ── ApplyInferredShapesTo{Graph,Model} ────────────────────────────────

TEST(OnnxOptimShapeInference, ApplyInferredShapesToModelFillsOutputAndValueInfo) {
  // Reshape(X, Constant([-1, 2])) with a fully-static input shape.
  // After ApplyInferredShapesToModel, the Y output of the graph must
  // carry the inferred {6, 2} shape and the intermediate S tensor
  // must appear in graph.value_info().
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{3, 4}, /*target=*/{-1, 2});
  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model);
  ctx.ApplyInferredShapesToModel(model);

  ASSERT_TRUE(model.has_graph());
  const GraphProto &graph = model.graph();
  // The Y output has type float and shape {6, 2}.
  ASSERT_EQ(graph.output_size(), 1u);
  const ValueInfoProto &out = graph.output(0);
  EXPECT_EQ(out.name(), "Y");
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
  for (int i = 0; i < static_cast<int>(graph.value_info_size()); ++i) {
    const ValueInfoProto &vi = graph.value_info()[i];
    if (vi.name() == "S") {
      found_s = true;
      ASSERT_TRUE(vi.has_type());
      ASSERT_TRUE(vi.type().has_tensor_type());
      EXPECT_EQ(static_cast<int>(vi.type().tensor_type().elem_type()),
                static_cast<int>(TensorProto::DataType::INT64));
    }
  }
  EXPECT_TRUE(found_s);
  // The X input is not duplicated in value_info.
  for (int i = 0; i < static_cast<int>(graph.value_info_size()); ++i) {
    EXPECT_NE(std::string(graph.value_info()[i].name()), std::string("X"));
  }
}

TEST(OnnxOptimShapeInference, ApplyInferredShapesToModelPreservesSymbolicDims) {
  // Reshape(X, Constant([-1, 2])) with a symbolic first input dim:
  // the inferred Y output must keep the first dim as a dim_param.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model);
  ctx.ApplyInferredShapesToModel(model);

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
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(ctx.ApplyInferredShapesToModel(model), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, InferShapesModelEndToEnd) {
  // Convenience wrapper: runs ComputeShapeModel + ApplyInferredShapesToModel.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{3, 4}, /*target=*/{-1, 2});
  core::shapes::InferShapesModel(model);

  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.has_type() && out.type().has_tensor_type());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[0].dim_value(), 6);
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 2);
}

TEST(OnnxOptimShapeInference, ComputeShapeModelPrefillAnchorExpressionTokens) {
  // The output anchor declares a compound symbolic dimension ("A+B").
  // The renaming pass must register not only the full expression but
  // also its individual tokens ("A", "B") as preferred names, so that
  // they survive constraint propagation instead of being substituted by
  // unrelated inferred symbols.
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 2}, /*symbolic_names=*/{"A+B"});

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model, /*prefill_with_value_info_output=*/true);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "A+B");
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(2));
  // The input dim "N" is unrelated to the anchor tokens and must not be
  // renamed by the propagation pass.
  ASSERT_TRUE(ctx.Has("X"));
  ASSERT_EQ(ctx.Get("X").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("X").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("X").Shape()[0].AsExpr(), "N");
}

TEST(OnnxOptimShapeInference, ComputeShapeModelAcceptsMalformedAnchorExpression) {
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 2}, /*symbolic_names=*/{"Y::0"});

  core::shapes::ShapesContext ctx;
  EXPECT_NO_THROW(ctx.ComputeShapeModel(model, /*prefill_with_value_info_output=*/true));

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsExpr(), "Y::0");
}

TEST(OnnxOptimShapeInference, InferShapesModelWithPrefillPrefersOutputAnchor) {
  ModelProto model = MakeReshapeWithConstantModel(/*input_shape=*/{-1, 4},
                                                  /*target=*/{-1, 2},
                                                  /*symbolic_names=*/{"N"});
  SetValueInfoTensorType(*model.mutable_graph()->mutable_output(0), TensorProto::DataType::FLOAT,
                         /*shape=*/{-1, 2}, /*symbolic_names=*/{"ANCHOR"});

  core::shapes::InferShapesModel(model, /*prefill_with_value_info_output=*/true);

  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.has_type() && out.type().has_tensor_type());
  ASSERT_EQ(out.type().tensor_type().shape().dim_size(), 2u);
  EXPECT_TRUE(out.type().tensor_type().shape().dim()[0].has_dim_param());
  EXPECT_EQ(std::string(out.type().tensor_type().shape().dim()[0].dim_param()), "ANCHOR");
  EXPECT_EQ(out.type().tensor_type().shape().dim()[1].dim_value(), 2);
}

TEST(OnnxOptimShapeInference, ComputeShapeModelReplacesConcatExprWithInputAnchor) {
  // ``present = Concat(past_key, new_key, axis=2)`` concatenates the
  // KV-cache sequence dims ``past_seq`` and ``seq``, so node-level inference
  // produces ``present`` (and its downstream consumer ``consumed``) with
  // axis-2 dim ``past_seq+seq``. The model also declares an input ``mask`` of
  // shape ``[batch, total_seq]`` (so ``total_seq`` is a first-class
  // graph-input symbol) and the ``present`` output anchor declares
  // ``[batch, 2, total_seq, 4]``. Shape inference must recognise that
  // ``past_seq+seq == total_seq`` and rewrite the inferred compound
  // expression to the input anchor ``total_seq`` everywhere it appears,
  // including the intermediate ``consumed`` (which is not a graph output and
  // therefore does not receive the anchor directly).
  ModelProto model;
  model.set_ir_version(9);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  ValueInfoProto *past = graph->add_input();
  past->set_name("past_key");
  SetValueInfoTensorType(*past, TensorProto::DataType::FLOAT, /*shape=*/{-1, 2, -1, 4},
                         /*symbolic_names=*/{"batch", "", "past_seq", ""});
  ValueInfoProto *new_key = graph->add_input();
  new_key->set_name("new_key");
  SetValueInfoTensorType(*new_key, TensorProto::DataType::FLOAT, /*shape=*/{-1, 2, -1, 4},
                         /*symbolic_names=*/{"batch", "", "seq", ""});
  ValueInfoProto *mask = graph->add_input();
  mask->set_name("mask");
  SetValueInfoTensorType(*mask, TensorProto::DataType::FLOAT, /*shape=*/{-1, -1},
                         /*symbolic_names=*/{"batch", "total_seq"});

  ValueInfoProto *out = graph->add_output();
  out->set_name("present");
  SetValueInfoTensorType(*out, TensorProto::DataType::FLOAT, /*shape=*/{-1, 2, -1, 4},
                         /*symbolic_names=*/{"batch", "", "total_seq", ""});

  NodeProto concat = MakeNode("Concat", {"past_key", "new_key"}, {"present"});
  AddAttribute<int64_t>(concat, "axis", 2);
  *graph->add_node() = std::move(concat);
  *graph->add_node() = MakeNode("Identity", {"present"}, {"consumed"});

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model);

  // The graph output adopts the declared anchor directly.
  ASSERT_TRUE(ctx.Has("present"));
  ASSERT_EQ(ctx.Get("present").Shape().Rank(), 4u);
  EXPECT_TRUE(ctx.Get("present").Shape()[2].IsExpr());
  EXPECT_EQ(ctx.Get("present").Shape()[2].AsExpr(), "total_seq");
  // The intermediate consumer is rewritten from ``past_seq+seq`` to the
  // equivalent graph-input anchor ``total_seq``.
  ASSERT_TRUE(ctx.Has("consumed"));
  ASSERT_EQ(ctx.Get("consumed").Shape().Rank(), 4u);
  EXPECT_TRUE(ctx.Get("consumed").Shape()[2].IsExpr());
  EXPECT_EQ(ctx.Get("consumed").Shape()[2].AsExpr(), "total_seq");
}

TEST(OnnxOptimShapeInference, ComputeShapeModelKeepsConcatExprWithoutInputAnchor) {
  // Counterpart to the test above: when the equivalent anchor ``e`` only
  // appears on a graph **output** (never on an input), the internally
  // computed expression ``past_seq+seq`` carries more information than the
  // opaque output label and must be preserved on the intermediate
  // ``consumed`` rather than collapsed to ``e``.
  ModelProto model;
  model.set_ir_version(9);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");

  ValueInfoProto *past = graph->add_input();
  past->set_name("past_key");
  SetValueInfoTensorType(*past, TensorProto::DataType::FLOAT, /*shape=*/{-1, 2, -1, 4},
                         /*symbolic_names=*/{"batch", "", "past_seq", ""});
  ValueInfoProto *new_key = graph->add_input();
  new_key->set_name("new_key");
  SetValueInfoTensorType(*new_key, TensorProto::DataType::FLOAT, /*shape=*/{-1, 2, -1, 4},
                         /*symbolic_names=*/{"batch", "", "seq", ""});

  ValueInfoProto *out = graph->add_output();
  out->set_name("present");
  SetValueInfoTensorType(*out, TensorProto::DataType::FLOAT, /*shape=*/{-1, 2, -1, 4},
                         /*symbolic_names=*/{"batch", "", "e", ""});

  NodeProto concat = MakeNode("Concat", {"past_key", "new_key"}, {"present"});
  AddAttribute<int64_t>(concat, "axis", 2);
  *graph->add_node() = std::move(concat);
  *graph->add_node() = MakeNode("Identity", {"present"}, {"consumed"});

  core::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model);

  ASSERT_TRUE(ctx.Has("consumed"));
  ASSERT_EQ(ctx.Get("consumed").Shape().Rank(), 4u);
  EXPECT_TRUE(ctx.Get("consumed").Shape()[2].IsExpr());
  EXPECT_EQ(ctx.Get("consumed").Shape()[2].AsExpr(), "past_seq+seq");
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
  core::shapes::ShapesContext ctx;

  ctx.ComputeShapeModel(model);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));
  // The local-function map should have been populated from model.functions().
  EXPECT_TRUE(ctx.HasLocalFunction("local:func_add"));
}

TEST(OnnxOptimShapeInference, InferShapesModelWritesBackLocalFunctionOutput) {
  // End-to-end: ``InferShapesModel`` writes the inferred Z shape back to
  // the proto.
  ModelProto model = MakeLocalFunctionAddModel();

  core::shapes::InferShapesModel(model);

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
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
}

TEST(OnnxOptimShapeInference, ComputeShapeNodeUsesRegisteredCustomDomainCallback) {
  NodeProto node = MakeNode("CustomIdentity", {"X"}, {"Y"}, "com.acme");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         {core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  bool called = false;
  ctx.SetCustomShapeInferenceFunction(
      "com.acme", "CustomIdentity",
      [&called](core::shapes::ShapesContext &inner_ctx, const NodeProto &inner_node) {
        called = true;
        const std::string input_name = inner_node.input(0);
        const std::string output_name = inner_node.output(0);
        inner_ctx.Set(output_name, core::symbolic::SymTensor(inner_ctx.Get(input_name)));
      });

  EXPECT_NO_THROW(ctx.ComputeShapeNode(node));
  EXPECT_TRUE(called);
  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

namespace {

// Builds a model whose local function body uses ``ref_attr_name`` to
// reference a formal attribute (``to``) supplied by the call site.
//   local:func_cast(a) -> b   with body: b = Cast(a) [to = ref(to)]
ModelProto MakeLocalFunctionRefAttrCastModel(int64_t cast_to) {
  ModelProto model;
  model.set_ir_version(static_cast<int64_t>(8));
  OperatorSetIdProto *ai = model.add_opset_import();
  ai->set_domain("");
  ai->set_version(static_cast<int64_t>(18));
  OperatorSetIdProto *loc = model.add_opset_import();
  loc->set_domain("local");
  loc->set_version(static_cast<int64_t>(1));

  FunctionProto *func = model.add_functions();
  func->set_name("func_cast");
  func->set_domain("local");
  func->add_input("a");
  func->add_output("b");
  func->add_attribute("to");
  OperatorSetIdProto *fopset = func->add_opset_import();
  fopset->set_domain("");
  fopset->set_version(static_cast<int64_t>(18));
  NodeProto *body = func->add_node();
  body->set_op_type("Cast");
  body->add_input("a");
  body->add_output("b");
  AttributeProto *ref = body->add_attribute();
  ref->set_name("to");
  ref->set_ref_attr_name("to");
  ref->set_type(AttributeProto::AttributeType::INT);

  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  ValueInfoProto *x = graph->add_input();
  x->set_name("X");
  SetValueInfoTensorType(*x, TensorProto::DataType::FLOAT, /*shape=*/{2, 3});
  ValueInfoProto *z = graph->add_output();
  z->set_name("Z");
  z->add_type();

  NodeProto *call = graph->add_node();
  call->set_op_type("func_cast");
  call->set_domain("local");
  call->add_input("X");
  call->add_output("Z");
  AttributeProto *attr = call->add_attribute();
  attr->set_name("to");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(cast_to);
  return model;
}

} // namespace

TEST(OnnxOptimShapeInference, ExpandsLocalFunctionWithLinkedAttribute) {
  // A function body using ``ref_attr_name`` must have its attribute
  // references bound to the call-site attribute values before shape
  // inference runs over the body. The Cast op's ``to`` attribute
  // (referenced from the function body) determines the output element
  // type, so the inferred Z must take that dtype.
  ModelProto model =
      MakeLocalFunctionRefAttrCastModel(static_cast<int64_t>(TensorProto::DataType::INT64));
  core::shapes::ShapesContext ctx;

  ctx.ComputeShapeModel(model);

  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Z").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesContextLessEqualConstraint, AddAndQuery) {
  core::shapes::ShapesContext ctx;
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 0u);
  // Self-bound is dropped.
  EXPECT_FALSE(ctx.AddLessEqualConstraint("a", "a"));
  // Empty operands are dropped.
  EXPECT_FALSE(ctx.AddLessEqualConstraint("", "rhs"));
  EXPECT_FALSE(ctx.AddLessEqualConstraint("a", ""));
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 0u);

  EXPECT_TRUE(ctx.AddLessEqualConstraint("a", "b"));
  EXPECT_TRUE(ctx.HasLessEqualConstraint("a", "b"));
  // Constraint is ordered: ``b <= a`` is *not* recorded.
  EXPECT_FALSE(ctx.HasLessEqualConstraint("b", "a"));
  // Re-inserting the same pair is a no-op.
  EXPECT_FALSE(ctx.AddLessEqualConstraint("a", "b"));
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 1u);
  // ``a <= a`` always reports true.
  EXPECT_TRUE(ctx.HasLessEqualConstraint("a", "a"));

  EXPECT_TRUE(ctx.AddLessEqualConstraint("a", "N*M"));
  EXPECT_EQ(ctx.LessEqualConstraintsSize(), 2u);
}

TEST(OnnxOptimShapesContextEventLog, DisabledByDefault) {
  core::shapes::ShapesContext ctx;
  EXPECT_FALSE(ctx.events_enabled());
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  // No events are recorded while logging is disabled.
  EXPECT_TRUE(ctx.Events().empty());
}

TEST(OnnxOptimShapesContextEventLog, SetRecordsAddAndReplace) {
  using core::shapes::ShapeEventAction;
  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);
  EXPECT_TRUE(ctx.Events().empty());

  // First Set on an absent name -> add event with the descriptor snapshot.
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim("N")}));
  ASSERT_EQ(ctx.Events().size(), 1u);
  const auto &add_ev = ctx.Events()[0];
  EXPECT_EQ(add_ev.action, ShapeEventAction::kAdd);
  EXPECT_EQ(add_ev.name, "X");
  EXPECT_EQ(add_ev.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(add_ev.shape, (std::vector<std::string>{"2", "N"}));
  EXPECT_TRUE(add_ev.op_type.empty());

  // Second Set on the same name -> replace event with the new dtype/shape.
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(5)}));
  ASSERT_EQ(ctx.Events().size(), 2u);
  EXPECT_EQ(ctx.Events()[1].action, ShapeEventAction::kReplace);
  EXPECT_EQ(ctx.Events()[1].name, "X");
  EXPECT_EQ(ctx.Events()[1].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_EQ(ctx.Events()[1].shape, (std::vector<std::string>{"5"}));

  ctx.ClearEvents();
  EXPECT_TRUE(ctx.Events().empty());
}

TEST(OnnxOptimShapesContextEventLog, ComputeShapeNodeRecordsComputeNodeEvent) {
  using core::shapes::ShapeEventAction;
  NodeProto node = MakeNode("Abs", {"X"}, {"Y"});
  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.ClearEvents();

  ctx.ComputeShapeNode(node);

  // The Abs kernel writes the output descriptor (one add event) and the
  // dispatch itself appends one compute_node event summarising the node.
  ASSERT_EQ(ctx.Events().size(), 2u);
  EXPECT_EQ(ctx.Events()[0].action, ShapeEventAction::kAdd);
  EXPECT_EQ(ctx.Events()[0].name, "Y");
  EXPECT_EQ(ctx.Events()[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(ctx.Events()[0].shape, (std::vector<std::string>{"2", "3"}));

  const auto &node_ev = ctx.Events()[1];
  EXPECT_EQ(node_ev.action, ShapeEventAction::kComputeNode);
  EXPECT_EQ(node_ev.op_domain, "ai.onnx");
  EXPECT_EQ(node_ev.op_type, "Abs");
  EXPECT_EQ(node_ev.inputs, (std::vector<std::string>{"X"}));
  EXPECT_EQ(node_ev.data_type, static_cast<int32_t>(TensorProto::DataType::UNDEFINED));
  EXPECT_TRUE(node_ev.shape.empty());
}

TEST(OnnxOptimShapesContextEventLog, ConstraintsRecordEvents) {
  using core::shapes::ShapeEventAction;
  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);

  // A new equality constraint records a kConstraint event with the
  // canonicalised operands in ``inputs``.
  EXPECT_TRUE(ctx.AddConstraint("N", "M"));
  ASSERT_EQ(ctx.Events().size(), 1u);
  const auto &eq_ev = ctx.Events()[0];
  EXPECT_EQ(eq_ev.action, ShapeEventAction::kConstraint);
  EXPECT_EQ(eq_ev.inputs, (std::vector<std::string>{"M", "N"}));
  EXPECT_EQ(eq_ev.data_type, static_cast<int32_t>(TensorProto::DataType::UNDEFINED));

  // Duplicate / self constraints do not append events.
  EXPECT_FALSE(ctx.AddConstraint("M", "N"));
  EXPECT_FALSE(ctx.AddConstraint("N", "N"));
  EXPECT_EQ(ctx.Events().size(), 1u);

  // A new upper-bound constraint records a kConstraintMax event.
  EXPECT_TRUE(ctx.AddLessEqualConstraint("nnz", "2*N"));
  ASSERT_EQ(ctx.Events().size(), 2u);
  const auto &le_ev = ctx.Events()[1];
  EXPECT_EQ(le_ev.action, ShapeEventAction::kConstraintMax);
  EXPECT_EQ(le_ev.inputs, (std::vector<std::string>{"nnz", "2*N"}));

  EXPECT_FALSE(ctx.AddLessEqualConstraint("nnz", "2*N"));
  EXPECT_EQ(ctx.Events().size(), 2u);
}

TEST(OnnxOptimShapesContextEventLog, ConstraintsDoNotRecordWhenDisabled) {
  core::shapes::ShapesContext ctx;
  EXPECT_TRUE(ctx.AddConstraint("N", "M"));
  EXPECT_TRUE(ctx.AddLessEqualConstraint("nnz", "2*N"));
  EXPECT_TRUE(ctx.Events().empty());
}

TEST(OnnxOptimShapesContextEventLog, ActionNames) {
  using core::shapes::ShapeEventAction;
  using core::shapes::ShapeEventActionName;
  EXPECT_STREQ(ShapeEventActionName(ShapeEventAction::kAdd), "add");
  EXPECT_STREQ(ShapeEventActionName(ShapeEventAction::kReplace), "replace");
  EXPECT_STREQ(ShapeEventActionName(ShapeEventAction::kComputeNode), "compute_node");
  EXPECT_STREQ(ShapeEventActionName(ShapeEventAction::kConstraint), "constraint");
  EXPECT_STREQ(ShapeEventActionName(ShapeEventAction::kConstraintMax), "constraint_max");
}

TEST(OnnxOptimShapesContextEventLog, NodeIndexTagsInputsInitializersAndNodes) {
  using core::shapes::ShapeEvent;
  using core::shapes::ShapeEventAction;
  // Graph: input X (float [3,4]), initializer S (int64 [2]) and a single
  // Reshape node producing Y. Inputs are tagged with node_index -1,
  // initializers with -2, and descriptors / compute_node events produced by
  // node 0 with 0.
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
  init->add_dims(std::vector<int64_t>{2});
  init->add_int64_data(std::vector<int64_t>{-1, 2});
  *graph->add_node() = MakeNode("Reshape", {"X", "S"}, {"Y"});

  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);
  ctx.ComputeShapeModel(model);

  auto first_event = [&](ShapeEventAction action, const std::string &name) -> const ShapeEvent * {
    for (const auto &ev : ctx.Events()) {
      if (ev.action == action && ev.name == name) {
        return &ev;
      }
    }
    return nullptr;
  };

  const ShapeEvent *x_ev = first_event(ShapeEventAction::kAdd, "X");
  ASSERT_NE(x_ev, nullptr);
  EXPECT_EQ(x_ev->node_index, -1);

  const ShapeEvent *s_ev = first_event(ShapeEventAction::kAdd, "S");
  ASSERT_NE(s_ev, nullptr);
  EXPECT_EQ(s_ev->node_index, -2);

  const ShapeEvent *y_ev = first_event(ShapeEventAction::kAdd, "Y");
  ASSERT_NE(y_ev, nullptr);
  EXPECT_EQ(y_ev->node_index, 0);

  const ShapeEvent *node_ev = nullptr;
  for (const auto &ev : ctx.Events()) {
    if (ev.action == ShapeEventAction::kComputeNode) {
      node_ev = &ev;
      break;
    }
  }
  ASSERT_NE(node_ev, nullptr);
  EXPECT_EQ(node_ev->op_type, "Reshape");
  EXPECT_EQ(node_ev->node_index, 0);
}

TEST(OnnxOptimShapesContextEventLog, TopLevelEventsHaveEmptyGraphName) {
  // A model with no subgraphs: all events must have an empty subgraph_attr_name.
  NodeProto node = MakeNode("Relu", {"X"}, {"Y"});
  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("g");
  ValueInfoProto *in = graph->add_input();
  in->set_name("X");
  SetValueInfoTensorType(*in, TensorProto::DataType::FLOAT, /*shape=*/{2, 3});
  ValueInfoProto *out = graph->add_output();
  out->set_name("Y");
  *graph->add_node() = node;

  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);
  ctx.ComputeShapeModel(model);

  for (const auto &ev : ctx.Events()) {
    EXPECT_TRUE(ev.subgraph_attr_name.empty())
        << "Expected empty subgraph_attr_name for top-level event, got: " << ev.subgraph_attr_name;
  }
}

TEST(OnnxOptimShapesContextEventLog, IfSubgraphEventsCarryBranchGraphName) {
  // Build a simple If model: branches each produce Abs(X).
  // Events from then_branch must carry subgraph_attr_name="then_branch" and
  // from else_branch subgraph_attr_name="else_branch"; outer events must have
  // empty subgraph_attr_name.
  using core::shapes::ShapeEvent;
  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("main");
  // Inputs
  ValueInfoProto *cond_vi = graph->add_input();
  cond_vi->set_name("cond");
  SetValueInfoTensorType(*cond_vi, TensorProto::DataType::BOOL, {});
  ValueInfoProto *x_vi = graph->add_input();
  x_vi->set_name("X");
  SetValueInfoTensorType(*x_vi, TensorProto::DataType::FLOAT, {2, 3});
  ValueInfoProto *out = graph->add_output();
  out->set_name("Z");

  NodeProto *if_node = graph->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("Z");

  // then_branch: Z_then = Abs(X)
  {
    AttributeProto *attr = if_node->add_attribute();
    attr->set_name("then_branch");
    attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto *tb = attr->mutable_g();
    *tb->add_node() = MakeNode("Abs", {"X"}, {"Z_then"});
    tb->add_output()->set_name("Z_then");
  }
  // else_branch: Z_else = Neg(X)
  {
    AttributeProto *attr = if_node->add_attribute();
    attr->set_name("else_branch");
    attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto *eb = attr->mutable_g();
    *eb->add_node() = MakeNode("Neg", {"X"}, {"Z_else"});
    eb->add_output()->set_name("Z_else");
  }

  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);
  ctx.ComputeShapeModel(model);

  bool found_then = false;
  bool found_else = false;
  for (const auto &ev : ctx.Events()) {
    if (ev.subgraph_attr_name == "then_branch") {
      found_then = true;
    }
    if (ev.subgraph_attr_name == "else_branch") {
      found_else = true;
    }
  }
  EXPECT_TRUE(found_then) << "No event with subgraph_attr_name='then_branch' found";
  EXPECT_TRUE(found_else) << "No event with subgraph_attr_name='else_branch' found";
}

TEST(OnnxOptimShapesContextEventLog, LoopSubgraphEventsCarryBodyGraphName) {
  // Build a Loop model: sum = Loop(M, cond, init) with a body that adds 1.
  using core::shapes::ShapeEvent;

  ModelProto model;
  model.set_ir_version(8);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("main");

  // Graph inputs
  ValueInfoProto *m_vi = graph->add_input();
  m_vi->set_name("M");
  SetValueInfoTensorType(*m_vi, TensorProto::DataType::INT64, {});
  ValueInfoProto *cond_vi = graph->add_input();
  cond_vi->set_name("cond");
  SetValueInfoTensorType(*cond_vi, TensorProto::DataType::BOOL, {});
  ValueInfoProto *init_vi = graph->add_input();
  init_vi->set_name("s_init");
  SetValueInfoTensorType(*init_vi, TensorProto::DataType::FLOAT, {});
  graph->add_output()->set_name("s_final");
  graph->add_output()->set_name("scan_out");

  NodeProto *loop_node = graph->add_node();
  loop_node->set_op_type("Loop");
  loop_node->add_input("M");
  loop_node->add_input("cond");
  loop_node->add_input("s_init");
  loop_node->add_output("s_final");
  loop_node->add_output("scan_out");

  // body subgraph: s_out = Add(s_in, one_init)
  AttributeProto *body_attr = loop_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->mutable_g();
  body->set_name("loop_body");
  body->add_input()->set_name("iter");
  body->add_input()->set_name("cond_in");
  body->add_input()->set_name("s_in");
  // initializer ``one``
  TensorProto *one = body->add_initializer();
  one->set_name("one");
  one->set_data_type(TensorProto::DataType::FLOAT);
  one->add_float_data(1.0f);
  *body->add_node() = MakeNode("Add", {"s_in", "one"}, {"s_out"});
  body->add_output()->set_name("cond_in");
  body->add_output()->set_name("s_out");
  body->add_output()->set_name("s_out");

  core::shapes::ShapesContext ctx;
  ctx.set_events_enabled(true);
  ctx.ComputeShapeModel(model);

  bool found_body = false;
  for (const auto &ev : ctx.Events()) {
    if (ev.subgraph_attr_name == "body") {
      found_body = true;
      break;
    }
  }
  EXPECT_TRUE(found_body) << "No event with subgraph_attr_name='body' found";
}

} // namespace Test
