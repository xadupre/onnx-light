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

} // namespace Test
