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
                              const char *auto_pad = nullptr) {
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
  return node;
}

void SetInput(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &shape,
              onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, dtype, shape));
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

TEST(OnnxOptimShapesNnAveragePool, RejectsAutoPadSameUpper) {
  NodeProto node = MakeAveragePoolNode({2, 2}, {}, {}, 0, "SAME_UPPER");
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsRankMismatch) {
  NodeProto node = MakeAveragePoolNode({2, 2, 2}); // 3D kernel
  onnx_optim::shapes::ShapesContext ctx;
  SetInput(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                       onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)});
  EXPECT_THROW(onnx_optim::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
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

} // namespace Test
