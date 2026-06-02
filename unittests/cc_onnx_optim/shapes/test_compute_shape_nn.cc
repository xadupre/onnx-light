// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeAveragePoolNode(const std::vector<int64_t> &kernel_shape,
                              const std::vector<int64_t> &strides = {},
                              const std::vector<int64_t> &pads = {}, int64_t ceil_mode = 0,
                              const char *auto_pad = nullptr,
                              const std::vector<int64_t> &dilations = {}) {
  NodeProto node;
  node.set_op_type("AveragePool");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", kernel_shape);
  if (!strides.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "strides", strides);
  }
  if (!pads.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "pads", pads);
  }
  if (ceil_mode != 0) {
    AddAttribute<int64_t>(node, "ceil_mode", ceil_mode);
  }
  if (auto_pad != nullptr) {
    AddAttribute<std::string>(node, "auto_pad", std::string(auto_pad));
  }
  if (!dilations.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "dilations", dilations);
  }
  return node;
}

void SetInput(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &shape,
              onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, dtype, shape));
}

NodeProto MakeDropoutNode(bool with_ratio = false, bool with_training_mode = false,
                          bool with_mask_output = false) {
  NodeProto node;
  node.set_op_type("Dropout");
  node.add_input("X");
  if (with_ratio) {
    node.add_input("ratio");
  }
  if (with_training_mode) {
    node.add_input("training_mode");
  }
  node.add_output("Y");
  if (with_mask_output) {
    node.add_output("mask");
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesNnAveragePool, Default2D) {
  // mirrors test_cc_averagepool_2d_default: kernel (2,2), stride 1, no pad.
  NodeProto node = MakeAveragePoolNode({2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                                       onnx_optim::OptimDim(32), onnx_optim::OptimDim(32)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 31);
  EXPECT_EQ(out[3].AsInt(), 31);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnAveragePool, StridesAndPads) {
  // mirrors test_cc_averagepool_2d_pads: kernel (5,5), pad 2 on every side.
  NodeProto node = MakeAveragePoolNode({5, 5}, /*strides=*/{}, /*pads=*/{2, 2, 2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                                       onnx_optim::OptimDim(28), onnx_optim::OptimDim(28)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 28);
  EXPECT_EQ(out[3].AsInt(), 28);
}

TEST(OnnxOptimShapesNnAveragePool, CeilModeDropsLastWindow) {
  // mirrors test_cc_averagepool_2d_ceil_last_window_starts_on_pad:
  // input 2x2, kernel 3x3, stride 3, pad (1,1,1,1), ceil_mode=1 -> 1x1 not 2x2.
  NodeProto node =
      MakeAveragePoolNode({3, 3}, /*strides=*/{3, 3}, /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(out[3].AsInt(), 1);
}

TEST(OnnxOptimShapesNnAveragePool, Default1D) {
  NodeProto node = MakeAveragePoolNode({2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                                       onnx_optim::OptimDim(32)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[2].AsInt(), 31);
}

TEST(OnnxOptimShapesNnAveragePool, Default3D) {
  NodeProto node = MakeAveragePoolNode({2, 2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(3), onnx_optim::OptimDim(3),
                                       onnx_optim::OptimDim(3)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(out[4].AsInt(), 2);
}

TEST(OnnxOptimShapesNnAveragePool, PropagatesSymbolicSpatialDim) {
  NodeProto node = MakeAveragePoolNode({2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3),
                                       onnx_optim::OptimDim("H"), onnx_optim::OptimDim(32)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  // N and H propagate symbolically; the static dim is computed.
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_EQ(out[3].AsInt(), 31);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("MaxPool");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsMissingKernelShape) {
  NodeProto node;
  node.set_op_type("AveragePool");
  node.add_input("X");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAveragePool, AutoPadSameUpper) {
  // SAME_UPPER on input 1x1x5x5 with kernel 3x3 stride 2 -> output is
  // ceil(5/2) = 3 along each spatial axis.
  NodeProto node = MakeAveragePoolNode({3, 3}, /*strides=*/{2, 2}, {}, 0, "SAME_UPPER");
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(5), onnx_optim::OptimDim(5)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsUnknownAutoPad) {
  NodeProto node = MakeAveragePoolNode({2, 2}, {}, {}, 0, "BAD_VALUE");
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAveragePool, Dilations2D) {
  // kernel 2x2, dilations (2,2), ceil_mode=1 on 1x1x4x4 input -> 2x2 output
  // (mirrors test_averagepool_2d_dilations).
  NodeProto node = MakeAveragePoolNode({2, 2}, /*strides=*/{1, 1}, /*pads=*/{}, /*ceil_mode=*/1,
                                       /*auto_pad=*/nullptr, /*dilations=*/{2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});

  onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsRankMismatch) {
  NodeProto node = MakeAveragePoolNode({2, 2, 2}); // 3D kernel
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnDropout, PropagatesPrimaryOutputShapeAndType) {
  NodeProto node = MakeDropoutNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});

  onnx_optim::shapes::nn::ComputeShapeDropout(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimTensor &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  EXPECT_EQ(y.Shape()[0].AsInt(), 2);
  EXPECT_EQ(y.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapesNnDropout, PropagatesMaskAsBool) {
  NodeProto node = MakeDropoutNode(/*with_ratio=*/true, /*with_training_mode=*/true,
                                   /*with_mask_output=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  ctx.Set("ratio", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{}));
  ctx.Set("training_mode", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                                   onnx_optim::OptimShape{}));

  onnx_optim::shapes::nn::ComputeShapeDropout(ctx, node, "X", "ratio", "training_mode");

  ASSERT_TRUE(ctx.Has("mask"));
  const onnx_optim::OptimTensor &mask = ctx.Get("mask");
  EXPECT_EQ(mask.Dtype(), onnx_optim::TensorType::kBool);
  ASSERT_EQ(mask.Shape().Rank(), 2u);
  EXPECT_EQ(mask.Shape()[0].AsInt(), 2);
  EXPECT_EQ(mask.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapesNnDropout, RejectsNonScalarOptionalInputs) {
  NodeProto node = MakeDropoutNode(/*with_ratio=*/true, /*with_training_mode=*/true,
                                   /*with_mask_output=*/false);
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  ctx.Set("ratio", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  ctx.Set("training_mode", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                                   onnx_optim::OptimShape{}));

  EXPECT_THROW(
      onnx_optim::shapes::nn::ComputeShapeDropout(ctx, node, "X", "ratio", "training_mode"),
      std::invalid_argument);
}

namespace {

NodeProto MakeRoiAlignNode(int64_t output_height = 1, int64_t output_width = 1) {
  NodeProto node;
  node.set_op_type("RoiAlign");
  node.add_input("X");
  node.add_input("rois");
  node.add_input("batch_indices");
  node.add_output("Y");
  if (output_height != 1) {
    AddAttribute<int64_t>(node, "output_height", output_height);
  }
  if (output_width != 1) {
    AddAttribute<int64_t>(node, "output_width", output_width);
  }
  return node;
}

void SetRoiAlignInputs(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &x,
                       const onnx_optim::OptimShape &rois,
                       const onnx_optim::OptimShape &batch_indices,
                       onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, dtype, x));
  ctx.Set("rois", onnx_optim::OptimTensor(nullptr, dtype, rois));
  ctx.Set("batch_indices",
          onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, batch_indices));
}

} // namespace

TEST(OnnxOptimShapesNnRoiAlign, DefaultOutputSize) {
  // Default output_height=1, output_width=1; 2 RoIs over a 1x3x10x10 map.
  NodeProto node = MakeRoiAlignNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                                           onnx_optim::OptimDim(10), onnx_optim::OptimDim(10)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(2)});

  onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(out[3].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnRoiAlign, CustomOutputSize) {
  // Standard Mask R-CNN style: 7x7 pooled output per RoI.
  NodeProto node = MakeRoiAlignNode(/*output_height=*/7, /*output_width=*/7);
  onnx_optim::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(256),
                                           onnx_optim::OptimDim(38), onnx_optim::OptimDim(50)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(5)});

  onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 5);
  EXPECT_EQ(out[1].AsInt(), 256);
  EXPECT_EQ(out[2].AsInt(), 7);
  EXPECT_EQ(out[3].AsInt(), 7);
}

TEST(OnnxOptimShapesNnRoiAlign, FallsBackToBatchIndicesForNumRois) {
  // num_rois is symbolic on the rois input but static on batch_indices.
  NodeProto node = MakeRoiAlignNode(/*output_height=*/2, /*output_width=*/3);
  onnx_optim::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4),
                                           onnx_optim::OptimDim(16), onnx_optim::OptimDim(16)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim("R"), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(7)});

  onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  EXPECT_TRUE(out[0].IsInt());
  EXPECT_EQ(out[0].AsInt(), 7);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnRoiAlign, PropagatesSymbolicNumRoisAndChannels) {
  NodeProto node = MakeRoiAlignNode(/*output_height=*/4, /*output_width=*/4);
  onnx_optim::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim("C"),
                                           onnx_optim::OptimDim(20), onnx_optim::OptimDim(20)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim("R"), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim("R")});

  onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "R");
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_EQ(out[1].AsExpr(), "C");
  EXPECT_EQ(out[2].AsInt(), 4);
  EXPECT_EQ(out[3].AsInt(), 4);
}

TEST(OnnxOptimShapesNnRoiAlign, RejectsWrongOpType) {
  NodeProto node = MakeRoiAlignNode();
  node.set_op_type("MaxRoiPool");
  onnx_optim::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                           onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  EXPECT_THROW(
      onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnRoiAlign, RejectsWrongRanks) {
  NodeProto node = MakeRoiAlignNode();
  onnx_optim::shapes::ShapesContext ctx;
  // X with rank 3 instead of 4.
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                           onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  EXPECT_THROW(
      onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnRoiAlign, RejectsNonPositiveOutputSize) {
  NodeProto node = MakeRoiAlignNode();
  AddAttribute<int64_t>(node, "output_height", 0);
  onnx_optim::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                           onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4)},
                    onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  EXPECT_THROW(
      onnx_optim::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices"),
      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// BatchNormalization
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeBatchNormalizationNode(int n_outputs = 1) {
  NodeProto node;
  node.set_op_type("BatchNormalization");
  node.add_input("X");
  node.add_input("scale");
  node.add_input("B");
  node.add_input("input_mean");
  node.add_input("input_var");
  node.add_output("Y");
  for (int i = 1; i < n_outputs; ++i) {
    node.add_output(std::string("Y") + std::to_string(i));
  }
  return node;
}

void SetBatchNormalizationInputs(onnx_optim::shapes::ShapesContext &ctx,
                                 const onnx_optim::OptimShape &x_shape,
                                 const onnx_optim::OptimShape &c_shape,
                                 onnx_optim::TensorType x_dtype = onnx_optim::TensorType::kFloat,
                                 onnx_optim::TensorType c_dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, x_dtype, x_shape));
  ctx.Set("scale", onnx_optim::OptimTensor(nullptr, c_dtype, c_shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, c_dtype, c_shape));
  ctx.Set("input_mean", onnx_optim::OptimTensor(nullptr, c_dtype, c_shape));
  ctx.Set("input_var", onnx_optim::OptimTensor(nullptr, c_dtype, c_shape));
}

} // namespace

TEST(OnnxOptimShapesNnBatchNormalization, InferenceModePropagatesXShape) {
  NodeProto node = MakeBatchNormalizationNode(/*n_outputs=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  SetBatchNormalizationInputs(
      ctx,
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                             onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)},
      onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  onnx_optim::shapes::nn::ComputeShapeBatchNormalization(ctx, node, "X", "input_mean");
  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &y_shape = ctx.Get("Y").Shape();
  ASSERT_EQ(y_shape.Rank(), 4u);
  EXPECT_EQ(y_shape[0], onnx_optim::OptimDim(2));
  EXPECT_EQ(y_shape[1], onnx_optim::OptimDim(3));
  EXPECT_EQ(y_shape[2], onnx_optim::OptimDim(4));
  EXPECT_EQ(y_shape[3], onnx_optim::OptimDim(5));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnBatchNormalization, TrainingModeAssignsChannelShapeToSecondaryOutputs) {
  NodeProto node = MakeBatchNormalizationNode(/*n_outputs=*/3);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 15);
  SetBatchNormalizationInputs(ctx,
                              onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                     onnx_optim::OptimDim(3),
                                                     onnx_optim::OptimDim(4)},
                              onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  onnx_optim::shapes::nn::ComputeShapeBatchNormalization(ctx, node, "X", "input_mean");
  ASSERT_TRUE(ctx.Has("Y1"));
  ASSERT_TRUE(ctx.Has("Y2"));
  const onnx_optim::OptimShape &mean_shape = ctx.Get("Y1").Shape();
  ASSERT_EQ(mean_shape.Rank(), 1u);
  EXPECT_EQ(mean_shape[0], onnx_optim::OptimDim(3));
  const onnx_optim::OptimShape &var_shape = ctx.Get("Y2").Shape();
  ASSERT_EQ(var_shape.Rank(), 1u);
  EXPECT_EQ(var_shape[0], onnx_optim::OptimDim(3));
}

TEST(OnnxOptimShapesNnBatchNormalization, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Conv");
  node.add_input("X");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeBatchNormalization(ctx, node, "X", "input_mean"),
               std::invalid_argument);
}

namespace {

NodeProto MakeRNNNode(const std::string &op_type, int n_outputs, int64_t hidden_size = -1,
                      const char *direction = nullptr, int64_t layout = 0) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("X");
  node.add_input("W");
  node.add_input("R");
  if (n_outputs >= 1) {
    node.add_output("Y");
  }
  if (n_outputs >= 2) {
    node.add_output("Y_h");
  }
  if (n_outputs >= 3) {
    node.add_output("Y_c");
  }
  if (hidden_size > 0) {
    AddAttribute<int64_t>(node, "hidden_size", hidden_size);
  }
  if (direction != nullptr) {
    AddAttribute<std::string>(node, "direction", std::string(direction));
  }
  if (layout != 0) {
    AddAttribute<int64_t>(node, "layout", layout);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesNnRNN, ForwardLayout0RNN) {
  NodeProto node = MakeRNNNode("RNN", /*n_outputs=*/2, /*hidden_size=*/5);
  onnx_optim::shapes::ShapesContext ctx;
  // X = [seq=4, batch=2, input=3]
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4),
                                                              onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3)}));
  onnx_optim::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 4u);
  EXPECT_EQ(y[0].AsInt(), 4);
  EXPECT_EQ(y[1].AsInt(), 1);
  EXPECT_EQ(y[2].AsInt(), 2);
  EXPECT_EQ(y[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);

  ASSERT_TRUE(ctx.Has("Y_h"));
  const onnx_optim::OptimShape &h = ctx.Get("Y_h").Shape();
  ASSERT_EQ(h.Rank(), 3u);
  EXPECT_EQ(h[0].AsInt(), 1);
  EXPECT_EQ(h[1].AsInt(), 2);
  EXPECT_EQ(h[2].AsInt(), 5);
}

TEST(OnnxOptimShapesNnRNN, BidirectionalRNN) {
  NodeProto node =
      MakeRNNNode("RNN", /*n_outputs=*/2, /*hidden_size=*/5, /*direction=*/"bidirectional");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4),
                                                              onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3)}));
  onnx_optim::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  const onnx_optim::OptimShape &y = ctx.Get("Y").Shape();
  EXPECT_EQ(y[1].AsInt(), 2);
  const onnx_optim::OptimShape &h = ctx.Get("Y_h").Shape();
  EXPECT_EQ(h[0].AsInt(), 2);
}

TEST(OnnxOptimShapesNnRNN, LSTMLayout1WithYc) {
  NodeProto node = MakeRNNNode("LSTM", /*n_outputs=*/3, /*hidden_size=*/4, /*direction=*/nullptr,
                               /*layout=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  // X layout=1: [batch=2, seq=6, input=3]
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(6),
                                                              onnx_optim::OptimDim(3)}));
  onnx_optim::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  const onnx_optim::OptimShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 4u);
  EXPECT_EQ(y[0].AsInt(), 2);
  EXPECT_EQ(y[1].AsInt(), 6);
  EXPECT_EQ(y[2].AsInt(), 1);
  EXPECT_EQ(y[3].AsInt(), 4);

  const onnx_optim::OptimShape &y_c = ctx.Get("Y_c").Shape();
  ASSERT_EQ(y_c.Rank(), 3u);
  EXPECT_EQ(y_c[0].AsInt(), 2);
  EXPECT_EQ(y_c[1].AsInt(), 1);
  EXPECT_EQ(y_c[2].AsInt(), 4);
}

TEST(OnnxOptimShapesNnRNN, HiddenSizeFallbackFromR) {
  // No hidden_size attribute; falls back to R.shape[2].
  NodeProto node = MakeRNNNode("GRU", /*n_outputs=*/2);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4),
                                                              onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3)}));
  ctx.Set("R", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                              onnx_optim::OptimDim(21),
                                                              onnx_optim::OptimDim(7)}));
  onnx_optim::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  const onnx_optim::OptimShape &h = ctx.Get("Y_h").Shape();
  EXPECT_EQ(h[2].AsInt(), 7);
}

TEST(OnnxOptimShapesNnRNN, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Conv");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeRNN(ctx, node, "X", nullptr),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnRNN, RejectsWrongInputRank) {
  NodeProto node = MakeRNNNode("RNN", /*n_outputs=*/2, /*hidden_size=*/5);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeRNN(ctx, node, "X", nullptr),
               std::invalid_argument);
}

namespace {

NodeProto MakeAttentionNode(int n_outputs = 1) {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_output("Y");
  if (n_outputs >= 2) {
    node.add_output("present_key");
  }
  if (n_outputs >= 3) {
    node.add_output("present_value");
  }
  if (n_outputs >= 4) {
    node.add_output("qk_matmul_output");
  }
  return node;
}

onnx_optim::OptimShape ShapeAttn4(int64_t a, int64_t b, int64_t c, int64_t d) {
  return onnx_optim::OptimShape{onnx_optim::OptimDim(a), onnx_optim::OptimDim(b),
                                onnx_optim::OptimDim(c), onnx_optim::OptimDim(d)};
}

void SetAttnInputs(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &q,
                   const onnx_optim::OptimShape &k, const onnx_optim::OptimShape &v) {
  ctx.Set("Q", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, q));
  ctx.Set("K", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, k));
  ctx.Set("V", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, v));
}

} // namespace

TEST(OnnxOptimShapesNnAttention, BasicShape) {
  NodeProto node = MakeAttentionNode(/*n_outputs=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, /*q=*/ShapeAttn4(2, 4, 8, 16),
                /*k=*/ShapeAttn4(2, 4, 12, 16),
                /*v=*/ShapeAttn4(2, 4, 12, 32));

  onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 8);
  EXPECT_EQ(out[3].AsInt(), 32);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnAttention, GroupedQueryAttention) {
  NodeProto node = MakeAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 8, 5, 6), ShapeAttn4(1, 2, 7, 6), ShapeAttn4(1, 2, 7, 6));

  onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 8);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 6);
}

TEST(OnnxOptimShapesNnAttention, AllFourOutputs) {
  NodeProto node = MakeAttentionNode(/*n_outputs=*/4);
  onnx_optim::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(2, 4, 8, 16), ShapeAttn4(2, 4, 12, 16), ShapeAttn4(2, 4, 12, 32));

  onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  // Y = (B, Hq, Lq, Dv).
  const auto &y = ctx.Get("Y").Shape();
  EXPECT_EQ(y[1].AsInt(), 4);
  EXPECT_EQ(y[3].AsInt(), 32);
  // present_key = (B, Hkv, Lkv, D).
  const auto &pk = ctx.Get("present_key").Shape();
  EXPECT_EQ(pk[1].AsInt(), 4);
  EXPECT_EQ(pk[2].AsInt(), 12);
  EXPECT_EQ(pk[3].AsInt(), 16);
  // present_value = (B, Hkv, Lkv, Dv).
  const auto &pv = ctx.Get("present_value").Shape();
  EXPECT_EQ(pv[3].AsInt(), 32);
  // qk_matmul_output = (B, Hq, Lq, Lkv).
  const auto &qk = ctx.Get("qk_matmul_output").Shape();
  EXPECT_EQ(qk[1].AsInt(), 4);
  EXPECT_EQ(qk[2].AsInt(), 8);
  EXPECT_EQ(qk[3].AsInt(), 12);
}

TEST(OnnxOptimShapesNnAttention, RejectsMismatchedKvSeqLen) {
  NodeProto node = MakeAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 5, 8), ShapeAttn4(1, 2, 7, 8));
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsInvalidGqa) {
  NodeProto node = MakeAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  // q_num_heads=5 not divisible by kv_num_heads=2.
  SetAttnInputs(ctx, ShapeAttn4(1, 5, 4, 8), ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8));
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsWrongRank) {
  NodeProto node = MakeAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("Q", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(4),
                                                              onnx_optim::OptimDim(8)}));
  ctx.Set("K", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       ShapeAttn4(2, 4, 8, 16)));
  ctx.Set("V", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       ShapeAttn4(2, 4, 8, 16)));
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsWrongOpType) {
  NodeProto node = MakeAttentionNode();
  node.set_op_type("NotAttention");
  onnx_optim::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8));
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

namespace {

NodeProto MakeDeformConvNode(const std::vector<int64_t> &kernel_shape,
                             const std::vector<int64_t> &strides = {},
                             const std::vector<int64_t> &pads = {},
                             const std::vector<int64_t> &dilations = {}) {
  NodeProto node;
  node.set_op_type("DeformConv");
  node.add_input("X");
  node.add_input("W");
  node.add_input("offset");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", kernel_shape);
  if (!strides.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "strides", strides);
  }
  if (!pads.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "pads", pads);
  }
  if (!dilations.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "dilations", dilations);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesNnDeformConv, BasicShape3x3NoPadding) {
  // 1x1x4x4 input, 1x1x3x3 kernel, no padding, stride 1, dilation 1
  // → output is 1x1x2x2.
  NodeProto node = MakeDeformConvNode({3, 3});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnDeformConv, WithPaddingStrideAndDilation) {
  // 1x3x5x5 input, 2x3x3x3 kernel, pads=1, stride=2, dilation=1
  // → out spatial = floor((5 + 2 - 3) / 2) + 1 = 3. Output 1x2x3x3.
  NodeProto node = MakeDeformConvNode({3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 1});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                                          onnx_optim::OptimDim(5), onnx_optim::OptimDim(5)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 2);
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnDeformConv, SymbolicBatchPropagates) {
  NodeProto node = MakeDeformConvNode({3, 3});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_FALSE(out[0].IsInt());
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesNnDeformConv, RejectsWrongOpType) {
  NodeProto node = MakeDeformConvNode({3, 3});
  node.set_op_type("NotDeformConv");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W"),
               std::invalid_argument);
}

namespace {

NodeProto MakeConvNode(const std::vector<int64_t> &kernel_shape,
                       const std::vector<int64_t> &strides = {},
                       const std::vector<int64_t> &pads = {},
                       const std::vector<int64_t> &dilations = {},
                       const std::string &auto_pad = "") {
  NodeProto node;
  node.set_op_type("Conv");
  node.add_input("X");
  node.add_input("W");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", kernel_shape);
  if (!strides.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "strides", strides);
  }
  if (!pads.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "pads", pads);
  }
  if (!dilations.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "dilations", dilations);
  }
  if (!auto_pad.empty()) {
    AddAttribute<std::string>(node, "auto_pad", auto_pad);
  }
  return node;
}

NodeProto MakeConvIntegerNode(const std::vector<int64_t> &kernel_shape) {
  NodeProto node;
  node.set_op_type("ConvInteger");
  node.add_input("X");
  node.add_input("W");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", kernel_shape);
  return node;
}

NodeProto MakeConvTransposeNode(const std::vector<int64_t> &kernel_shape,
                                const std::vector<int64_t> &strides = {},
                                const std::vector<int64_t> &pads = {},
                                const std::vector<int64_t> &output_padding = {},
                                const std::vector<int64_t> &output_shape = {}, int64_t group = 1) {
  NodeProto node;
  node.set_op_type("ConvTranspose");
  node.add_input("X");
  node.add_input("W");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", kernel_shape);
  if (!strides.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "strides", strides);
  }
  if (!pads.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "pads", pads);
  }
  if (!output_padding.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "output_padding", output_padding);
  }
  if (!output_shape.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "output_shape", output_shape);
  }
  if (group != 1) {
    AddAttribute<int64_t>(node, "group", group);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesNnConv, BasicShape3x3NoPadding) {
  NodeProto node = MakeConvNode({3, 3});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeConv(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnConv, SameUpperReturnsCeiledShape) {
  NodeProto node = MakeConvNode({3, 3}, {2, 2}, {}, {}, "SAME_UPPER");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(5), onnx_optim::OptimDim(5)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeConv(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  // ceil(5/2) = 3.
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnConvInteger, BasicShapeReturnsInt32) {
  NodeProto node = MakeConvIntegerNode({2, 2});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kUint8,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kUint8,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::nn::ComputeShapeConvInteger(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt32);
}

TEST(OnnxOptimShapesNnConvTranspose, BasicShape3x3NoPadding) {
  // 1x1x3x3 input, 1x2x3x3 weight, defaults → out spatial = 1*(3-1) + 1*3 = 5.
  NodeProto node = MakeConvTransposeNode({3, 3});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(2),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeConvTranspose(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 2); // M = W.shape[1] * group = 2.
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnConvTranspose, OutputShapeHonored) {
  NodeProto node = MakeConvTransposeNode({3, 3}, {2, 2}, {}, {}, {6, 6});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));
  ctx.Set("W", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::nn::ComputeShapeConvTranspose(ctx, node, "X", "W");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 6);
}

namespace {

NodeProto MakeCol2ImNode(const std::vector<int64_t> &pads = {},
                         const std::vector<int64_t> &strides = {},
                         const std::vector<int64_t> &dilations = {}) {
  NodeProto node;
  node.set_op_type("Col2Im");
  node.add_input("input");
  node.add_input("image_shape");
  node.add_input("block_shape");
  node.add_output("output");
  if (!pads.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "pads", pads);
  }
  if (!strides.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "strides", strides);
  }
  if (!dilations.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "dilations", dilations);
  }
  return node;
}

NodeProto MakeFlattenNode(const char *x_name, const char *y_name,
                          const int64_t *axis_value = nullptr) {
  NodeProto node;
  node.set_op_type("Flatten");
  node.add_input(x_name);
  node.add_output(y_name);
  if (axis_value != nullptr) {
    AddAttribute<int64_t>(node, "axis", *axis_value);
  }
  return node;
}

onnx_optim::OptimTensor MakeIntInitializer(const std::vector<int64_t> &values) {
  onnx_optim::OptimShape values_as_shape;
  for (int64_t v : values) {
    values_as_shape.PushBack(onnx_optim::OptimDim(v));
  }
  onnx_optim::OptimTensor t(
      nullptr, onnx_optim::TensorType::kInt64,
      onnx_optim::OptimShape{onnx_optim::OptimDim(static_cast<int64_t>(values.size()))});
  t.SetValueAsShape(values_as_shape);
  return t;
}

} // namespace

TEST(OnnxOptimShapesNnCol2Im, BasicShape2D) {
  // input: (1, 5, 5); image_shape=[5,5]; block_shape=[1,5] → output (1, 1, 5, 5).
  NodeProto node = MakeCol2ImNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(5),
                                                                  onnx_optim::OptimDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));

  onnx_optim::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  ASSERT_TRUE(ctx.Has("output"));
  const onnx_optim::OptimShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("output").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnCol2Im, ChannelsDivisibleByBlockProduct) {
  // input: (2, 12, 8); block_shape=[3,4] → product 12 → C = 1.
  NodeProto node = MakeCol2ImNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(12),
                                                                  onnx_optim::OptimDim(8)}));
  ctx.Set("image_shape", MakeIntInitializer({3, 4}));
  ctx.Set("block_shape", MakeIntInitializer({3, 4}));

  onnx_optim::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  const onnx_optim::OptimShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 4);
}

TEST(OnnxOptimShapesNnCol2Im, SymbolicBatchPropagates) {
  NodeProto node = MakeCol2ImNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim("N"),
                                                                  onnx_optim::OptimDim(5),
                                                                  onnx_optim::OptimDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));

  onnx_optim::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  const onnx_optim::OptimShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_FALSE(out[0].IsInt());
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 5);
}

TEST(OnnxOptimShapesNnCol2Im, UnknownInitializersProduceSymbolicSpatial) {
  // When image_shape has no value annotation but its 1-D shape exposes the
  // rank statically, the spatial rank is still recovered and spatial dims
  // are symbolic.
  NodeProto node = MakeCol2ImNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(5),
                                                                  onnx_optim::OptimDim(5)}));
  ctx.Set("image_shape", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                                 onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  ctx.Set("block_shape", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                                 onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  const onnx_optim::OptimShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_FALSE(out[1].IsInt()); // block_product unknown → C symbolic.
  EXPECT_FALSE(out[2].IsInt());
  EXPECT_FALSE(out[3].IsInt());
}

TEST(OnnxOptimShapesNnCol2Im, RejectsWrongOpType) {
  NodeProto node = MakeCol2ImNode();
  node.set_op_type("NotCol2Im");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(5),
                                                                  onnx_optim::OptimDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));
  EXPECT_THROW(
      onnx_optim::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnCol2Im, RejectsWrongInputRank) {
  NodeProto node = MakeCol2ImNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("input", onnx_optim::OptimTensor(
                       nullptr, onnx_optim::TensorType::kFloat,
                       onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));
  EXPECT_THROW(
      onnx_optim::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnFlatten, DefaultAxisFlattensAllButFirstDim) {
  NodeProto node = MakeFlattenNode("X", "Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 60);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnFlatten, AxisZeroMakesLeadingDimOne) {
  const int64_t axis = 0;
  NodeProto node = MakeFlattenNode("X", "Y", &axis);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 24);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
}

TEST(OnnxOptimShapesNnFlatten, NegativeAxisCountsFromBack) {
  const int64_t axis = -1;
  NodeProto node = MakeFlattenNode("X", "Y", &axis);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 24);
  EXPECT_EQ(out[1].AsInt(), 5);
}

TEST(OnnxOptimShapesNnFlatten, SymbolicDimsProduceSymbolicProduct) {
  NodeProto node = MakeFlattenNode("X", "Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(std::string("N")),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_FALSE(out[0].IsInt());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_TRUE(out[1].IsInt());
  EXPECT_EQ(out[1].AsInt(), 12);
}

TEST(OnnxOptimShapesNnFlatten, OutOfRangeAxisThrows) {
  const int64_t axis = 5;
  NodeProto node = MakeFlattenNode("X", "Y", &axis);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeFlatten(ctx, node, "X"), std::invalid_argument);
}

} // namespace Test
