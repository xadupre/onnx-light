// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <tuple>

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

void SetInput(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &shape,
              core::symbolic::TensorType dtype = core::symbolic::TensorType::kFloat) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, dtype, shape));
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
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(32), core::symbolic::SymDim(32)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 31);
  EXPECT_EQ(out[3].AsInt(), 31);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnAveragePool, StridesAndPads) {
  // mirrors test_cc_averagepool_2d_pads: kernel (5,5), pad 2 on every side.
  NodeProto node = MakeAveragePoolNode({5, 5}, /*strides=*/{}, /*pads=*/{2, 2, 2, 2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(28), core::symbolic::SymDim(28)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 28);
  EXPECT_EQ(out[3].AsInt(), 28);
}

TEST(OnnxOptimShapesNnAveragePool, CeilModeDropsLastWindow) {
  // mirrors test_cc_averagepool_2d_ceil_last_window_starts_on_pad:
  // input 2x2, kernel 3x3, stride 3, pad (1,1,1,1), ceil_mode=1 -> 1x1 not 2x2.
  NodeProto node =
      MakeAveragePoolNode({3, 3}, /*strides=*/{3, 3}, /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/1);
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(2), core::symbolic::SymDim(2)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(out[3].AsInt(), 1);
}

TEST(OnnxOptimShapesNnAveragePool, Default1D) {
  NodeProto node = MakeAveragePoolNode({2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(32)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[2].AsInt(), 31);
}

TEST(OnnxOptimShapesNnAveragePool, Default3D) {
  NodeProto node = MakeAveragePoolNode({2, 2, 2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(3), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(3)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(out[4].AsInt(), 2);
}

TEST(OnnxOptimShapesNnAveragePool, PropagatesSymbolicSpatialDim) {
  NodeProto node = MakeAveragePoolNode({2, 2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim("H"), core::symbolic::SymDim(32)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
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
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsMissingKernelShape) {
  NodeProto node;
  node.set_op_type("AveragePool");
  node.add_input("X");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAveragePool, AutoPadSameUpper) {
  // SAME_UPPER on input 1x1x5x5 with kernel 3x3 stride 2 -> output is
  // ceil(5/2) = 3 along each spatial axis.
  NodeProto node = MakeAveragePoolNode({3, 3}, /*strides=*/{2, 2}, {}, 0, "SAME_UPPER");
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(5), core::symbolic::SymDim(5)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnAveragePool, Dilations2D) {
  // kernel 2x2, dilations (2,2), ceil_mode=1 on 1x1x4x4 input -> 2x2 output
  // (mirrors test_averagepool_2d_dilations).
  NodeProto node = MakeAveragePoolNode({2, 2}, /*strides=*/{1, 1}, /*pads=*/{}, /*ceil_mode=*/1,
                                       /*auto_pad=*/nullptr, /*dilations=*/{2, 2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});

  onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesNnAveragePool, RejectsRankMismatch) {
  NodeProto node = MakeAveragePoolNode({2, 2, 2}); // 3D kernel
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAveragePool(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeLpPoolNode(const std::vector<int64_t> &kernel_shape,
                         const std::vector<int64_t> &strides = {},
                         const std::vector<int64_t> &pads = {}, int64_t p = 2,
                         const char *auto_pad = nullptr,
                         const std::vector<int64_t> &dilations = {}) {
  NodeProto node;
  node.set_op_type("LpPool");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", kernel_shape);
  if (!strides.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "strides", strides);
  }
  if (!pads.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "pads", pads);
  }
  AddAttribute<int64_t>(node, "p", p);
  if (auto_pad != nullptr) {
    AddAttribute<std::string>(node, "auto_pad", std::string(auto_pad));
  }
  if (!dilations.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "dilations", dilations);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesNnLpPool, Default2D) {
  // mirrors test_cc_lppool_2d_default: kernel (2,2), no strides, no pads.
  NodeProto node = MakeLpPoolNode({2, 2}, /*strides=*/{}, /*pads=*/{}, /*p=*/4);
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(32), core::symbolic::SymDim(32)});

  onnx_shapes::shapes::nn::ComputeShapeLpPool(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 31);
  EXPECT_EQ(out[3].AsInt(), 31);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnLpPool, PadsAndStrides) {
  // mirrors test_cc_lppool_2d_pads: kernel (3,3), pad 2 on every side.
  NodeProto node = MakeLpPoolNode({3, 3}, /*strides=*/{}, /*pads=*/{2, 2, 2, 2}, /*p=*/3);
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(28), core::symbolic::SymDim(28)});

  onnx_shapes::shapes::nn::ComputeShapeLpPool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 30);
  EXPECT_EQ(out[3].AsInt(), 30);
}

TEST(OnnxOptimShapesNnLpPool, AutoPadSameUpper) {
  NodeProto node = MakeLpPoolNode({2, 2}, /*strides=*/{}, /*pads=*/{}, /*p=*/2, "SAME_UPPER");
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(32), core::symbolic::SymDim(32)});

  onnx_shapes::shapes::nn::ComputeShapeLpPool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 32);
  EXPECT_EQ(out[3].AsInt(), 32);
}

TEST(OnnxOptimShapesNnLpPool, Dilations2D) {
  // mirrors test_cc_lppool_2d_dilations: kernel (2,2), dilations (2,2) on
  // 1x1x4x4 input -> 2x2 output.
  NodeProto node = MakeLpPoolNode({2, 2}, /*strides=*/{1, 1}, /*pads=*/{}, /*p=*/2,
                                  /*auto_pad=*/nullptr, /*dilations=*/{2, 2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});

  onnx_shapes::shapes::nn::ComputeShapeLpPool(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesNnLpPool, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("AveragePool");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeLpPool(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesNnLpPool, RejectsMissingKernelShape) {
  NodeProto node;
  node.set_op_type("LpPool");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<int64_t>(node, "p", 2);
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(4), core::symbolic::SymDim(4)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeLpPool(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesNnMaxPool, Default2DSingleOutput) {
  // mirrors test_cc_maxpool_2d_default: kernel (2, 2), no strides, no pads
  // -> 31x31 output and no Indices output declared.
  NodeProto node;
  node.set_op_type("MaxPool");
  node.add_input("X");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});

  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                         core::symbolic::SymDim(32), core::symbolic::SymDim(32)});

  onnx_shapes::shapes::nn::ComputeShapeMaxPool(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 31);
  EXPECT_EQ(out[3].AsInt(), 31);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnMaxPool, EmitsIndicesWhenSecondOutputDeclared) {
  // mirrors test_cc_maxpool_with_argmax_2d_precomputed_pads: kernel (5, 5),
  // pads (2, 2, 2, 2), Indices output present with int64 dtype.
  NodeProto node;
  node.set_op_type("MaxPool");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Indices");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
  AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                         core::symbolic::SymDim(5), core::symbolic::SymDim(5)});

  onnx_shapes::shapes::nn::ComputeShapeMaxPool(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Indices"));
  const core::symbolic::SymShape &y_shape = ctx.Get("Y").Shape();
  const core::symbolic::SymShape &i_shape = ctx.Get("Indices").Shape();
  EXPECT_EQ(y_shape[2].AsInt(), 5);
  EXPECT_EQ(y_shape[3].AsInt(), 5);
  EXPECT_EQ(i_shape[2].AsInt(), 5);
  EXPECT_EQ(i_shape[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("Indices").Dtype(), core::symbolic::TensorType::kInt64);
}

TEST(OnnxOptimShapesNnMaxUnpool, ComputesShapeFromAttributes) {
  // mirrors test_cc_maxunpool_export_without_output_shape: 2x2 kernel, stride
  // 2 -> output spatial 4x4 from a 2x2 input.
  NodeProto node;
  node.set_op_type("MaxUnpool");
  node.add_input("X");
  node.add_input("I");
  node.add_output("output");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
  AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});

  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));
  ctx.Set("I", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt64,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::nn::ComputeShapeMaxUnpool(ctx, node, "X", "I", nullptr);

  ASSERT_TRUE(ctx.Has("output"));
  const core::symbolic::SymShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
  EXPECT_EQ(out[3].AsInt(), 4);
  EXPECT_EQ(ctx.Get("output").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnDropout, PropagatesPrimaryOutputShapeAndType) {
  NodeProto node = MakeDropoutNode();
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  onnx_shapes::shapes::nn::ComputeShapeDropout(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  EXPECT_EQ(y.Shape()[0].AsInt(), 2);
  EXPECT_EQ(y.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapesNnDropout, PropagatesMaskAsBool) {
  NodeProto node = MakeDropoutNode(/*with_ratio=*/true, /*with_training_mode=*/true,
                                   /*with_mask_output=*/true);
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  ctx.Set("ratio", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{}));
  ctx.Set("training_mode", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                                     core::symbolic::SymShape{}));

  onnx_shapes::shapes::nn::ComputeShapeDropout(ctx, node, "X", "ratio", "training_mode");

  ASSERT_TRUE(ctx.Has("mask"));
  const core::symbolic::SymTensor &mask = ctx.Get("mask");
  EXPECT_EQ(mask.Dtype(), core::symbolic::TensorType::kBool);
  ASSERT_EQ(mask.Shape().Rank(), 2u);
  EXPECT_EQ(mask.Shape()[0].AsInt(), 2);
  EXPECT_EQ(mask.Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapesNnDropout, RejectsNonScalarOptionalInputs) {
  NodeProto node = MakeDropoutNode(/*with_ratio=*/true, /*with_training_mode=*/true,
                                   /*with_mask_output=*/false);
  core::shapes::ShapesContext ctx;
  SetInput(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  ctx.Set("ratio", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1)}));
  ctx.Set("training_mode", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                                     core::symbolic::SymShape{}));

  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeDropout(ctx, node, "X", "ratio", "training_mode"),
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

void SetRoiAlignInputs(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &x,
                       const core::symbolic::SymShape &rois,
                       const core::symbolic::SymShape &batch_indices,
                       core::symbolic::TensorType dtype = core::symbolic::TensorType::kFloat) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, dtype, x));
  ctx.Set("rois", core::symbolic::SymTensor(nullptr, dtype, rois));
  ctx.Set("batch_indices",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, batch_indices));
}

} // namespace

TEST(OnnxOptimShapesNnRoiAlign, DefaultOutputSize) {
  // Default output_height=1, output_width=1; 2 RoIs over a 1x3x10x10 map.
  NodeProto node = MakeRoiAlignNode();
  core::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                             core::symbolic::SymDim(10),
                                             core::symbolic::SymDim(10)},
                    core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(2)});

  onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(out[3].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnRoiAlign, CustomOutputSize) {
  // Standard Mask R-CNN style: 7x7 pooled output per RoI.
  NodeProto node = MakeRoiAlignNode(/*output_height=*/7, /*output_width=*/7);
  core::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(256),
                                             core::symbolic::SymDim(38),
                                             core::symbolic::SymDim(50)},
                    core::symbolic::SymShape{core::symbolic::SymDim(5), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(5)});

  onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 5);
  EXPECT_EQ(out[1].AsInt(), 256);
  EXPECT_EQ(out[2].AsInt(), 7);
  EXPECT_EQ(out[3].AsInt(), 7);
}

TEST(OnnxOptimShapesNnRoiAlign, FallsBackToBatchIndicesForNumRois) {
  // num_rois is symbolic on the rois input but static on batch_indices.
  NodeProto node = MakeRoiAlignNode(/*output_height=*/2, /*output_width=*/3);
  core::shapes::ShapesContext ctx;
  SetRoiAlignInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4),
                               core::symbolic::SymDim(16), core::symbolic::SymDim(16)},
      core::symbolic::SymShape{core::symbolic::SymDim("R"), core::symbolic::SymDim(4)},
      core::symbolic::SymShape{core::symbolic::SymDim(7)});

  onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_TRUE(out[0].IsInt());
  EXPECT_EQ(out[0].AsInt(), 7);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnRoiAlign, PropagatesSymbolicNumRoisAndChannels) {
  NodeProto node = MakeRoiAlignNode(/*output_height=*/4, /*output_width=*/4);
  core::shapes::ShapesContext ctx;
  SetRoiAlignInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim("C"),
                               core::symbolic::SymDim(20), core::symbolic::SymDim(20)},
      core::symbolic::SymShape{core::symbolic::SymDim("R"), core::symbolic::SymDim(4)},
      core::symbolic::SymShape{core::symbolic::SymDim("R")});

  onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
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
  core::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                             core::symbolic::SymDim(4), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(1)});
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices"),
      std::invalid_argument);
}

namespace {

NodeProto MakeMaxRoiPoolNode(int64_t pooled_h = 2, int64_t pooled_w = 2,
                             bool include_pooled_shape = true) {
  NodeProto node;
  node.set_op_type("MaxRoiPool");
  node.add_input("X");
  node.add_input("rois");
  node.add_output("Y");
  if (include_pooled_shape) {
    AddAttribute<std::vector<int64_t>>(node, "pooled_shape", {pooled_h, pooled_w});
  }
  return node;
}

void SetMaxRoiPoolInputs(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &x,
                         const core::symbolic::SymShape &rois,
                         core::symbolic::TensorType dtype = core::symbolic::TensorType::kFloat) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, dtype, x));
  ctx.Set("rois", core::symbolic::SymTensor(nullptr, dtype, rois));
}

} // namespace

TEST(OnnxOptimShapesNnMaxRoiPool, StaticShape) {
  NodeProto node = MakeMaxRoiPoolNode(/*pooled_h=*/3, /*pooled_w=*/4);
  core::shapes::ShapesContext ctx;
  SetMaxRoiPoolInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(8),
                               core::symbolic::SymDim(16), core::symbolic::SymDim(16)},
      core::symbolic::SymShape{core::symbolic::SymDim(5), core::symbolic::SymDim(5)});

  onnx_shapes::shapes::nn::ComputeShapeMaxRoiPool(ctx, node, "X", "rois");

  const core::symbolic::SymTensor &out = ctx.Get("Y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kFloat);
  const core::symbolic::SymShape &out_shape = out.Shape();
  ASSERT_EQ(out_shape.Rank(), 4u);
  EXPECT_EQ(out_shape[0].AsInt(), 5);
  EXPECT_EQ(out_shape[1].AsInt(), 8);
  EXPECT_EQ(out_shape[2].AsInt(), 3);
  EXPECT_EQ(out_shape[3].AsInt(), 4);
}

TEST(OnnxOptimShapesNnMaxRoiPool, PropagatesSymbolicNumRoisAndChannels) {
  NodeProto node = MakeMaxRoiPoolNode(/*pooled_h=*/7, /*pooled_w=*/7);
  core::shapes::ShapesContext ctx;
  SetMaxRoiPoolInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim("C"),
                               core::symbolic::SymDim(32), core::symbolic::SymDim(32)},
      core::symbolic::SymShape{core::symbolic::SymDim("R"), core::symbolic::SymDim(5)});

  onnx_shapes::shapes::nn::ComputeShapeMaxRoiPool(ctx, node, "X", "rois");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "R");
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_EQ(out[1].AsExpr(), "C");
  EXPECT_EQ(out[2].AsInt(), 7);
  EXPECT_EQ(out[3].AsInt(), 7);
}

TEST(OnnxOptimShapesNnMaxRoiPool, RejectsMissingPooledShape) {
  NodeProto node = MakeMaxRoiPoolNode(/*pooled_h=*/2, /*pooled_w=*/2,
                                      /*include_pooled_shape=*/false);
  core::shapes::ShapesContext ctx;
  SetMaxRoiPoolInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                               core::symbolic::SymDim(4), core::symbolic::SymDim(4)},
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(5)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeMaxRoiPool(ctx, node, "X", "rois"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnMaxRoiPool, RejectsWrongOpType) {
  NodeProto node = MakeMaxRoiPoolNode();
  node.set_op_type("RoiAlign");
  core::shapes::ShapesContext ctx;
  SetMaxRoiPoolInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                               core::symbolic::SymDim(4), core::symbolic::SymDim(4)},
      core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(5)});
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeMaxRoiPool(ctx, node, "X", "rois"),
               std::invalid_argument);
}

namespace {

NodeProto MakeNonMaxSuppressionNode() {
  NodeProto node;
  node.set_op_type("NonMaxSuppression");
  node.add_input("boxes");
  node.add_input("scores");
  node.add_output("selected_indices");
  return node;
}

} // namespace

TEST(OnnxOptimShapesNnNonMaxSuppression, OutputIsRank2Int64WithSymbolicFirstDim) {
  NodeProto node = MakeNonMaxSuppressionNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("boxes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(6),
                                                                      core::symbolic::SymDim(4)}));
  ctx.Set("scores", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                       core::symbolic::SymDim(1),
                                                                       core::symbolic::SymDim(6)}));

  onnx_shapes::shapes::nn::ComputeShapeNonMaxSuppression(ctx, node, "boxes", "scores");

  ASSERT_TRUE(ctx.Has("selected_indices"));
  const core::symbolic::SymShape &out = ctx.Get("selected_indices").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_TRUE(out[1].IsInt());
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(ctx.Get("selected_indices").Dtype(), core::symbolic::TensorType::kInt64);
}

TEST(OnnxOptimShapesNnNonMaxSuppression, RejectsBadRanks) {
  NodeProto node = MakeNonMaxSuppressionNode();
  core::shapes::ShapesContext ctx;
  // boxes wrong rank (2 instead of 3).
  ctx.Set("boxes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(6),
                                                                      core::symbolic::SymDim(4)}));
  ctx.Set("scores", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                       core::symbolic::SymDim(1),
                                                                       core::symbolic::SymDim(6)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeNonMaxSuppression(ctx, node, "boxes", "scores"),
               std::invalid_argument);

  // boxes last dim != 4.
  ctx.Set("boxes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(6),
                                                                      core::symbolic::SymDim(5)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeNonMaxSuppression(ctx, node, "boxes", "scores"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnNonMaxSuppression, RejectsWrongOpType) {
  NodeProto node = MakeNonMaxSuppressionNode();
  node.set_op_type("RoiAlign");
  core::shapes::ShapesContext ctx;
  ctx.Set("boxes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(4)}));
  ctx.Set("scores", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                              core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                       core::symbolic::SymDim(1),
                                                                       core::symbolic::SymDim(1)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeNonMaxSuppression(ctx, node, "boxes", "scores"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnRoiAlign, RejectsWrongRanks) {
  NodeProto node = MakeRoiAlignNode();
  core::shapes::ShapesContext ctx;
  // X with rank 3 instead of 4.
  SetRoiAlignInputs(ctx,
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                             core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(1)});
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnRoiAlign, RejectsNonPositiveOutputSize) {
  NodeProto node = MakeRoiAlignNode();
  AddAttribute<int64_t>(node, "output_height", 0);
  core::shapes::ShapesContext ctx;
  SetRoiAlignInputs(ctx,
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                             core::symbolic::SymDim(4), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(4)},
                    core::symbolic::SymShape{core::symbolic::SymDim(1)});
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeRoiAlign(ctx, node, "X", "rois", "batch_indices"),
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

void SetBatchNormalizationInputs(
    core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &x_shape,
    const core::symbolic::SymShape &c_shape,
    core::symbolic::TensorType x_dtype = core::symbolic::TensorType::kFloat,
    core::symbolic::TensorType c_dtype = core::symbolic::TensorType::kFloat) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, x_dtype, x_shape));
  ctx.Set("scale", core::symbolic::SymTensor(nullptr, c_dtype, c_shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, c_dtype, c_shape));
  ctx.Set("input_mean", core::symbolic::SymTensor(nullptr, c_dtype, c_shape));
  ctx.Set("input_var", core::symbolic::SymTensor(nullptr, c_dtype, c_shape));
}

} // namespace

TEST(OnnxOptimShapesNnBatchNormalization, InferenceModePropagatesXShape) {
  NodeProto node = MakeBatchNormalizationNode(/*n_outputs=*/1);
  core::shapes::ShapesContext ctx;
  SetBatchNormalizationInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                               core::symbolic::SymDim(4), core::symbolic::SymDim(5)},
      core::symbolic::SymShape{core::symbolic::SymDim(3)});
  onnx_shapes::shapes::nn::ComputeShapeBatchNormalization(ctx, node, "X", "input_mean");
  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &y_shape = ctx.Get("Y").Shape();
  ASSERT_EQ(y_shape.Rank(), 4u);
  EXPECT_EQ(y_shape[0], core::symbolic::SymDim(2));
  EXPECT_EQ(y_shape[1], core::symbolic::SymDim(3));
  EXPECT_EQ(y_shape[2], core::symbolic::SymDim(4));
  EXPECT_EQ(y_shape[3], core::symbolic::SymDim(5));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnBatchNormalization, TrainingModeAssignsChannelShapeToSecondaryOutputs) {
  NodeProto node = MakeBatchNormalizationNode(/*n_outputs=*/3);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 15);
  SetBatchNormalizationInputs(ctx,
                              core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                       core::symbolic::SymDim(3),
                                                       core::symbolic::SymDim(4)},
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  onnx_shapes::shapes::nn::ComputeShapeBatchNormalization(ctx, node, "X", "input_mean");
  ASSERT_TRUE(ctx.Has("Y1"));
  ASSERT_TRUE(ctx.Has("Y2"));
  const core::symbolic::SymShape &mean_shape = ctx.Get("Y1").Shape();
  ASSERT_EQ(mean_shape.Rank(), 1u);
  EXPECT_EQ(mean_shape[0], core::symbolic::SymDim(3));
  const core::symbolic::SymShape &var_shape = ctx.Get("Y2").Shape();
  ASSERT_EQ(var_shape.Rank(), 1u);
  EXPECT_EQ(var_shape[0], core::symbolic::SymDim(3));
}

TEST(OnnxOptimShapesNnBatchNormalization, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Conv");
  node.add_input("X");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeBatchNormalization(ctx, node, "X", "input_mean"),
      std::invalid_argument);
}

namespace {

NodeProto MakeGroupNormalizationNode(int64_t num_groups) {
  NodeProto node;
  node.set_op_type("GroupNormalization");
  node.add_input("X");
  node.add_input("scale");
  node.add_input("bias");
  node.add_output("Y");
  AddAttribute<int64_t>(node, "num_groups", num_groups);
  return node;
}

void SetGroupNormalizationInputs(core::shapes::ShapesContext &ctx,
                                 const core::symbolic::SymShape &x_shape,
                                 const core::symbolic::SymShape &scale_shape,
                                 const core::symbolic::SymShape &bias_shape) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, x_shape));
  ctx.Set("scale",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, scale_shape));
  ctx.Set("bias",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, bias_shape));
}

} // namespace

TEST(OnnxOptimShapesNnGroupNormalization, ValidatesInputsAndPropagatesX) {
  NodeProto node = MakeGroupNormalizationNode(2);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 21);
  SetGroupNormalizationInputs(
      ctx,
      core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4),
                               core::symbolic::SymDim(int64_t{0}), core::symbolic::SymDim("W")},
      core::symbolic::SymShape{core::symbolic::SymDim(4)},
      core::symbolic::SymShape{core::symbolic::SymDim(4)});

  onnx_shapes::shapes::nn::ComputeShapeGroupNormalization(ctx, node, "X");

  const core::symbolic::SymShape &output = ctx.Get("Y").Shape();
  ASSERT_EQ(output.Rank(), 4u);
  EXPECT_EQ(output[0].AsExpr(), "N");
  EXPECT_EQ(output[1].AsInt(), 4);
  EXPECT_EQ(output[2].AsInt(), 0);
  EXPECT_EQ(output[3].AsExpr(), "W");
}

TEST(OnnxOptimShapesNnGroupNormalization, AcceptsUnknownChannel) {
  NodeProto node = MakeGroupNormalizationNode(2);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 21);
  SetGroupNormalizationInputs(ctx,
                              core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                       core::symbolic::SymDim("?"),
                                                       core::symbolic::SymDim("W")},
                              core::symbolic::SymShape{core::symbolic::SymDim("?")},
                              core::symbolic::SymShape{core::symbolic::SymDim("?")});

  EXPECT_NO_THROW(onnx_shapes::shapes::nn::ComputeShapeGroupNormalization(ctx, node, "X"));
}

TEST(OnnxOptimShapesNnGroupNormalization, RejectsInvalidInputs) {
  const std::vector<std::tuple<core::symbolic::SymShape, core::symbolic::SymShape,
                               core::symbolic::SymShape, int64_t>>
      invalid_inputs = {
          {{core::symbolic::SymDim(2)},
           {core::symbolic::SymDim(2)},
           {core::symbolic::SymDim(2)},
           1},
          {{core::symbolic::SymDim(2), core::symbolic::SymDim(4)},
           {core::symbolic::SymDim(4), core::symbolic::SymDim(1)},
           {core::symbolic::SymDim(4)},
           2},
          {{core::symbolic::SymDim(2), core::symbolic::SymDim(4)},
           {core::symbolic::SymDim(3)},
           {core::symbolic::SymDim(4)},
           2},
          {{core::symbolic::SymDim(2), core::symbolic::SymDim(4)},
           {core::symbolic::SymDim(4)},
           {core::symbolic::SymDim(3)},
           2},
          {{core::symbolic::SymDim(2), core::symbolic::SymDim(5)},
           {core::symbolic::SymDim(5)},
           {core::symbolic::SymDim(5)},
           2},
          {{core::symbolic::SymDim(2), core::symbolic::SymDim(4)},
           {core::symbolic::SymDim(4)},
           {core::symbolic::SymDim(4)},
           0},
      };
  for (const auto &[x_shape, scale_shape, bias_shape, num_groups] : invalid_inputs) {
    NodeProto node = MakeGroupNormalizationNode(num_groups);
    core::shapes::ShapesContext ctx;
    ctx.SetOpsetVersion("ai.onnx", 21);
    SetGroupNormalizationInputs(ctx, x_shape, scale_shape, bias_shape);
    EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeGroupNormalization(ctx, node, "X"),
                 std::invalid_argument);
  }
}

TEST(OnnxOptimShapesNnGroupNormalization, PreservesDeprecatedOpset18ScaleShape) {
  NodeProto node = MakeGroupNormalizationNode(2);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 18);
  SetGroupNormalizationInputs(ctx,
                              core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                       core::symbolic::SymDim(4),
                                                       core::symbolic::SymDim(3)},
                              core::symbolic::SymShape{core::symbolic::SymDim(2)},
                              core::symbolic::SymShape{core::symbolic::SymDim(2)});

  EXPECT_NO_THROW(onnx_shapes::shapes::nn::ComputeShapeGroupNormalization(ctx, node, "X"));
}

TEST(OnnxOptimShapesNnMeanVarianceNormalization, PropagatesInputShapeAndType) {
  NodeProto node;
  node.set_op_type("MeanVarianceNormalization");
  node.add_input("X");
  node.add_output("Y");

  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::nn::ComputeShapeMeanVarianceNormalization(ctx, node, "X");
  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 3u);
  EXPECT_EQ(y[0], core::symbolic::SymDim(2));
  EXPECT_EQ(y[1], core::symbolic::SymDim(3));
  EXPECT_EQ(y[2], core::symbolic::SymDim(4));
}

TEST(OnnxOptimShapesNnMeanVarianceNormalization, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("BatchNormalization");
  node.add_input("X");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeMeanVarianceNormalization(ctx, node, "X"),
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
  core::shapes::ShapesContext ctx;
  // X = [seq=4, batch=2, input=3]
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4),
                                                                  core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3)}));
  onnx_shapes::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 4u);
  EXPECT_EQ(y[0].AsInt(), 4);
  EXPECT_EQ(y[1].AsInt(), 1);
  EXPECT_EQ(y[2].AsInt(), 2);
  EXPECT_EQ(y[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);

  ASSERT_TRUE(ctx.Has("Y_h"));
  const core::symbolic::SymShape &h = ctx.Get("Y_h").Shape();
  ASSERT_EQ(h.Rank(), 3u);
  EXPECT_EQ(h[0].AsInt(), 1);
  EXPECT_EQ(h[1].AsInt(), 2);
  EXPECT_EQ(h[2].AsInt(), 5);
}

TEST(OnnxOptimShapesNnRNN, BidirectionalRNN) {
  NodeProto node =
      MakeRNNNode("RNN", /*n_outputs=*/2, /*hidden_size=*/5, /*direction=*/"bidirectional");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4),
                                                                  core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3)}));
  onnx_shapes::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  EXPECT_EQ(y[1].AsInt(), 2);
  const core::symbolic::SymShape &h = ctx.Get("Y_h").Shape();
  EXPECT_EQ(h[0].AsInt(), 2);
}

TEST(OnnxOptimShapesNnRNN, LSTMLayout1WithYc) {
  NodeProto node = MakeRNNNode("LSTM", /*n_outputs=*/3, /*hidden_size=*/4, /*direction=*/nullptr,
                               /*layout=*/1);
  core::shapes::ShapesContext ctx;
  // X layout=1: [batch=2, seq=6, input=3]
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(6),
                                                                  core::symbolic::SymDim(3)}));
  onnx_shapes::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 4u);
  EXPECT_EQ(y[0].AsInt(), 2);
  EXPECT_EQ(y[1].AsInt(), 6);
  EXPECT_EQ(y[2].AsInt(), 1);
  EXPECT_EQ(y[3].AsInt(), 4);

  const core::symbolic::SymShape &y_c = ctx.Get("Y_c").Shape();
  ASSERT_EQ(y_c.Rank(), 3u);
  EXPECT_EQ(y_c[0].AsInt(), 2);
  EXPECT_EQ(y_c[1].AsInt(), 1);
  EXPECT_EQ(y_c[2].AsInt(), 4);
}

TEST(OnnxOptimShapesNnRNN, HiddenSizeFallbackFromR) {
  // No hidden_size attribute; falls back to R.shape[2].
  NodeProto node = MakeRNNNode("GRU", /*n_outputs=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4),
                                                                  core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3)}));
  ctx.Set("R", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim(21),
                                                                  core::symbolic::SymDim(7)}));
  onnx_shapes::shapes::nn::ComputeShapeRNN(ctx, node, "X", "R");

  const core::symbolic::SymShape &h = ctx.Get("Y_h").Shape();
  EXPECT_EQ(h[2].AsInt(), 7);
}

TEST(OnnxOptimShapesNnRNN, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Conv");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeRNN(ctx, node, "X", nullptr),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnRNN, RejectsWrongInputRank) {
  NodeProto node = MakeRNNNode("RNN", /*n_outputs=*/2, /*hidden_size=*/5);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(4), core::symbolic::SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeRNN(ctx, node, "X", nullptr),
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

core::symbolic::SymShape ShapeAttn4(int64_t a, int64_t b, int64_t c, int64_t d) {
  return core::symbolic::SymShape{core::symbolic::SymDim(a), core::symbolic::SymDim(b),
                                  core::symbolic::SymDim(c), core::symbolic::SymDim(d)};
}

void SetAttnInputs(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &q,
                   const core::symbolic::SymShape &k, const core::symbolic::SymShape &v) {
  ctx.Set("Q", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, q));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, k));
  ctx.Set("V", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, v));
}

} // namespace

TEST(OnnxOptimShapesNnAttention, BasicShape) {
  NodeProto node = MakeAttentionNode(/*n_outputs=*/1);
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, /*q=*/ShapeAttn4(2, 4, 8, 16),
                /*k=*/ShapeAttn4(2, 4, 12, 16),
                /*v=*/ShapeAttn4(2, 4, 12, 32));

  onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 8);
  EXPECT_EQ(out[3].AsInt(), 32);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnAttention, GroupedQueryAttention) {
  NodeProto node = MakeAttentionNode();
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 8, 5, 6), ShapeAttn4(1, 2, 7, 6), ShapeAttn4(1, 2, 7, 6));

  onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 8);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 6);
}

TEST(OnnxOptimShapesNnAttention, AllFourOutputs) {
  NodeProto node = MakeAttentionNode(/*n_outputs=*/4);
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(2, 4, 8, 16), ShapeAttn4(2, 4, 12, 16), ShapeAttn4(2, 4, 12, 32));

  onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

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
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 5, 8), ShapeAttn4(1, 2, 7, 8));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsInvalidGqa) {
  NodeProto node = MakeAttentionNode();
  core::shapes::ShapesContext ctx;
  // q_num_heads=5 not divisible by kv_num_heads=2.
  SetAttnInputs(ctx, ShapeAttn4(1, 5, 4, 8), ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsInvalidWindowAttribute) {
  NodeProto node = MakeAttentionNode();
  auto *attr = node.add_attribute();
  attr->set_name("left_window_size");
  attr->set_type(AttributeProto::INT);
  attr->set_i(-2);
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsWrongRank) {
  NodeProto node = MakeAttentionNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("Q", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(4),
                                                                  core::symbolic::SymDim(8)}));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         ShapeAttn4(2, 4, 8, 16)));
  ctx.Set("V", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         ShapeAttn4(2, 4, 8, 16)));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsWrongOpType) {
  NodeProto node = MakeAttentionNode();
  node.set_op_type("NotAttention");
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8), ShapeAttn4(1, 2, 4, 8));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

namespace {

void AddIntAttribute(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

core::symbolic::SymShape ShapeAttn3(int64_t a, int64_t b, int64_t c) {
  return core::symbolic::SymShape{core::symbolic::SymDim(a), core::symbolic::SymDim(b),
                                  core::symbolic::SymDim(c)};
}

} // namespace

TEST(OnnxOptimShapesNnAttention, Rank3BasicShape) {
  NodeProto node = MakeAttentionNode(/*n_outputs=*/4);
  AddIntAttribute(node, "q_num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 4);
  core::shapes::ShapesContext ctx;
  // Q=(B=2, Lq=8, Hq*D=4*16), K=(B=2, Lkv=12, Hkv*D=4*16), V=(B=2, Lkv=12, Hkv*Dv=4*32).
  SetAttnInputs(ctx, ShapeAttn3(2, 8, 64), ShapeAttn3(2, 12, 64), ShapeAttn3(2, 12, 128));

  onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  // Output 0 stays rank-3: Y=(B, Lq, Hq*Dv).
  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 3u);
  EXPECT_EQ(y[0].AsInt(), 2);
  EXPECT_EQ(y[1].AsInt(), 8);
  EXPECT_EQ(y[2].AsInt(), 128);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);

  // present_key = (B, Hkv, Lkv, D).
  const core::symbolic::SymShape &pk = ctx.Get("present_key").Shape();
  ASSERT_EQ(pk.Rank(), 4u);
  EXPECT_EQ(pk[0].AsInt(), 2);
  EXPECT_EQ(pk[1].AsInt(), 4);
  EXPECT_EQ(pk[2].AsInt(), 12);
  EXPECT_EQ(pk[3].AsInt(), 16);

  // present_value = (B, Hkv, Lkv, Dv).
  const core::symbolic::SymShape &pv = ctx.Get("present_value").Shape();
  ASSERT_EQ(pv.Rank(), 4u);
  EXPECT_EQ(pv[3].AsInt(), 32);

  // qk_matmul_output = (B, Hq, Lq, Lkv).
  const core::symbolic::SymShape &qk = ctx.Get("qk_matmul_output").Shape();
  ASSERT_EQ(qk.Rank(), 4u);
  EXPECT_EQ(qk[1].AsInt(), 4);
  EXPECT_EQ(qk[2].AsInt(), 8);
  EXPECT_EQ(qk[3].AsInt(), 12);
}

TEST(OnnxOptimShapesNnAttention, Rank3GroupedQueryAttention) {
  NodeProto node = MakeAttentionNode();
  AddIntAttribute(node, "q_num_heads", 8);
  AddIntAttribute(node, "kv_num_heads", 2);
  core::shapes::ShapesContext ctx;
  // head_size=6 -> Q last dim = 8*6=48, K last dim = 2*6=12, V (Dv=6) = 2*6=12.
  SetAttnInputs(ctx, ShapeAttn3(1, 5, 48), ShapeAttn3(1, 7, 12), ShapeAttn3(1, 7, 12));

  onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 3u);
  EXPECT_EQ(y[0].AsInt(), 1);
  EXPECT_EQ(y[1].AsInt(), 5);
  EXPECT_EQ(y[2].AsInt(), 48); // q_num_heads(8) * v_head_size(6).
}

TEST(OnnxOptimShapesNnAttention, Rank3SymbolicSequenceLengths) {
  NodeProto node = MakeAttentionNode();
  AddIntAttribute(node, "q_num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 4);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape q{core::symbolic::SymDim(2), core::symbolic::SymDim(std::string("Lq")),
                             core::symbolic::SymDim(64)};
  core::symbolic::SymShape k{core::symbolic::SymDim(2), core::symbolic::SymDim(std::string("Lkv")),
                             core::symbolic::SymDim(64)};
  core::symbolic::SymShape v{core::symbolic::SymDim(2), core::symbolic::SymDim(std::string("Lkv")),
                             core::symbolic::SymDim(128)};
  SetAttnInputs(ctx, q, k, v);

  onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V");

  const core::symbolic::SymShape &y = ctx.Get("Y").Shape();
  ASSERT_EQ(y.Rank(), 3u);
  EXPECT_EQ(y[0].AsInt(), 2);
  EXPECT_FALSE(y[1].IsInt());
  EXPECT_EQ(y[2].AsInt(), 128);
}

TEST(OnnxOptimShapesNnAttention, Rank3RejectsMissingHeadAttributes) {
  NodeProto node = MakeAttentionNode();
  core::shapes::ShapesContext ctx;
  SetAttnInputs(ctx, ShapeAttn3(2, 8, 64), ShapeAttn3(2, 12, 64), ShapeAttn3(2, 12, 128));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, Rank3RejectsIndivisibleHidden) {
  NodeProto node = MakeAttentionNode();
  AddIntAttribute(node, "q_num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 4);
  core::shapes::ShapesContext ctx;
  // Q last dim 65 not divisible by q_num_heads=4.
  SetAttnInputs(ctx, ShapeAttn3(2, 8, 65), ShapeAttn3(2, 12, 64), ShapeAttn3(2, 12, 128));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesNnAttention, RejectsMixedRanks) {
  NodeProto node = MakeAttentionNode();
  AddIntAttribute(node, "q_num_heads", 4);
  AddIntAttribute(node, "kv_num_heads", 4);
  core::shapes::ShapesContext ctx;
  ctx.Set("Q", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         ShapeAttn3(2, 8, 64)));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         ShapeAttn4(2, 4, 12, 16)));
  ctx.Set("V", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         ShapeAttn4(2, 4, 12, 32)));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeAttention(ctx, node, "Q", "K", "V"),
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
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnDeformConv, WithPaddingStrideAndDilation) {
  // 1x3x5x5 input, 2x3x3x3 kernel, pads=1, stride=2, dilation=1
  // → out spatial = floor((5 + 2 - 3) / 2) + 1 = 3. Output 1x2x3x3.
  NodeProto node = MakeDeformConvNode({3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 1});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3),
                                            core::symbolic::SymDim(5), core::symbolic::SymDim(5)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 2);
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnDeformConv, SymbolicBatchPropagates) {
  NodeProto node = MakeDeformConvNode({3, 3});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_FALSE(out[0].IsInt());
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesNnDeformConv, RejectsWrongOpType) {
  NodeProto node = MakeDeformConvNode({3, 3});
  node.set_op_type("NotDeformConv");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeDeformConv(ctx, node, "X", "W"),
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
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(4)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeConv(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnConv, SameUpperReturnsCeiledShape) {
  NodeProto node = MakeConvNode({3, 3}, {2, 2}, {}, {}, "SAME_UPPER");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(5), core::symbolic::SymDim(5)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeConv(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  // ceil(5/2) = 3.
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 3);
}

TEST(OnnxOptimShapesNnConvInteger, BasicShapeReturnsInt32) {
  NodeProto node = MakeConvIntegerNode({2, 2});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kUint8,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kUint8,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(2), core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::nn::ComputeShapeConvInteger(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[2].AsInt(), 2);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt32);
}

TEST(OnnxOptimShapesNnConvTranspose, BasicShape3x3NoPadding) {
  // 1x1x3x3 input, 1x2x3x3 weight, defaults → out spatial = 1*(3-1) + 1*3 = 5.
  NodeProto node = MakeConvTransposeNode({3, 3});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeConvTranspose(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 2); // M = W.shape[1] * group = 2.
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnConvTranspose, OutputShapeHonored) {
  NodeProto node = MakeConvTransposeNode({3, 3}, {2, 2}, {}, {}, {6, 6});
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));
  ctx.Set("W", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::nn::ComputeShapeConvTranspose(ctx, node, "X", "W");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
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

core::symbolic::SymTensor MakeIntInitializer(const std::vector<int64_t> &values) {
  core::symbolic::SymShape values_as_shape;
  for (int64_t v : values) {
    values_as_shape.PushBack(core::symbolic::SymDim(v));
  }
  core::symbolic::SymTensor t(
      nullptr, core::symbolic::TensorType::kInt64,
      core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(values.size()))});
  t.SetValueAsShape(values_as_shape);
  return t;
}

} // namespace

TEST(OnnxOptimShapesNnCol2Im, BasicShape2D) {
  // input: (1, 5, 5); image_shape=[5,5]; block_shape=[1,5] → output (1, 1, 5, 5).
  NodeProto node = MakeCol2ImNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(5),
                                                                      core::symbolic::SymDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));

  onnx_shapes::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  ASSERT_TRUE(ctx.Has("output"));
  const core::symbolic::SymShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 5);
  EXPECT_EQ(ctx.Get("output").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnCol2Im, ChannelsDivisibleByBlockProduct) {
  // input: (2, 12, 8); block_shape=[3,4] → product 12 → C = 1.
  NodeProto node = MakeCol2ImNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                      core::symbolic::SymDim(12),
                                                                      core::symbolic::SymDim(8)}));
  ctx.Set("image_shape", MakeIntInitializer({3, 4}));
  ctx.Set("block_shape", MakeIntInitializer({3, 4}));

  onnx_shapes::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  const core::symbolic::SymShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 3);
  EXPECT_EQ(out[3].AsInt(), 4);
}

TEST(OnnxOptimShapesNnCol2Im, SymbolicBatchPropagates) {
  NodeProto node = MakeCol2ImNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim("N"),
                                                                      core::symbolic::SymDim(5),
                                                                      core::symbolic::SymDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));

  onnx_shapes::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  const core::symbolic::SymShape &out = ctx.Get("output").Shape();
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
  core::shapes::ShapesContext ctx;
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(5),
                                                                      core::symbolic::SymDim(5)}));
  ctx.Set("image_shape",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("block_shape",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape");

  const core::symbolic::SymShape &out = ctx.Get("output").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_FALSE(out[1].IsInt()); // block_product unknown → C symbolic.
  EXPECT_FALSE(out[2].IsInt());
  EXPECT_FALSE(out[3].IsInt());
}

TEST(OnnxOptimShapesNnCol2Im, RejectsWrongOpType) {
  NodeProto node = MakeCol2ImNode();
  node.set_op_type("NotCol2Im");
  core::shapes::ShapesContext ctx;
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(5),
                                                                      core::symbolic::SymDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnCol2Im, RejectsWrongInputRank) {
  NodeProto node = MakeCol2ImNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(5)}));
  ctx.Set("image_shape", MakeIntInitializer({5, 5}));
  ctx.Set("block_shape", MakeIntInitializer({1, 5}));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeCol2Im(ctx, node, "input", "image_shape", "block_shape"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnFlatten, DefaultAxisFlattensAllButFirstDim) {
  NodeProto node = MakeFlattenNode("X", "Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 60);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesNnFlatten, AxisZeroMakesLeadingDimOne) {
  const int64_t axis = 0;
  NodeProto node = MakeFlattenNode("X", "Y", &axis);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                         core::symbolic::SymShape{core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3),
                                                                  core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 24);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
}

TEST(OnnxOptimShapesNnFlatten, NegativeAxisCountsFromBack) {
  const int64_t axis = -1;
  NodeProto node = MakeFlattenNode("X", "Y", &axis);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                            core::symbolic::SymDim(4), core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 24);
  EXPECT_EQ(out[1].AsInt(), 5);
}

TEST(OnnxOptimShapesNnFlatten, SymbolicDimsProduceSymbolicProduct) {
  NodeProto node = MakeFlattenNode("X", "Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(std::string("N")),
                                            core::symbolic::SymDim(3), core::symbolic::SymDim(4)}));

  onnx_shapes::shapes::nn::ComputeShapeFlatten(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_FALSE(out[0].IsInt());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_TRUE(out[1].IsInt());
  EXPECT_EQ(out[1].AsInt(), 12);
}

TEST(OnnxOptimShapesNnFlatten, OutOfRangeAxisThrows) {
  const int64_t axis = 5;
  NodeProto node = MakeFlattenNode("X", "Y", &axis);
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kFloat,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));

  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeFlatten(ctx, node, "X"), std::invalid_argument);
}

namespace {

NodeProto MakeLinearAttentionNode(int64_t q_num_heads, int64_t kv_num_heads, int n_inputs = 3,
                                  int n_outputs = 2) {
  NodeProto node;
  node.set_op_type("LinearAttention");
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  if (n_inputs >= 4) {
    node.add_input("past_state");
  }
  node.add_output("Y");
  if (n_outputs >= 2) {
    node.add_output("present_state");
  }
  {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("q_num_heads");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(q_num_heads);
  }
  {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("kv_num_heads");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(kv_num_heads);
  }
  return node;
}

core::symbolic::SymShape ShapeLin3(int64_t a, int64_t b, int64_t c) {
  return core::symbolic::SymShape{core::symbolic::SymDim(a), core::symbolic::SymDim(b),
                                  core::symbolic::SymDim(c)};
}

void SetLinAttnInputs(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &q,
                      const core::symbolic::SymShape &k, const core::symbolic::SymShape &v) {
  ctx.Set("query", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, q));
  ctx.Set("key", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, k));
  ctx.Set("value", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, v));
}

} // namespace

TEST(OnnxOptimShapesNnLinearAttention, BasicShape) {
  // B=2, T=4, H_q=H_kv=2, d_k=8, d_v=16 -> query/key (2,4,16), value (2,4,32).
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(2, 4, 16), ShapeLin3(2, 4, 16), ShapeLin3(2, 4, 32));

  onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 32); // q_num_heads * d_v = 2 * 16.
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);

  ASSERT_TRUE(ctx.Has("present_state"));
  const core::symbolic::SymShape &ps = ctx.Get("present_state").Shape();
  ASSERT_EQ(ps.Rank(), 4u);
  EXPECT_EQ(ps[0].AsInt(), 2);
  EXPECT_EQ(ps[1].AsInt(), 2);
  EXPECT_EQ(ps[2].AsInt(), 8);
  EXPECT_EQ(ps[3].AsInt(), 16);
}

TEST(OnnxOptimShapesNnLinearAttention, GroupedQueryAttention) {
  // H_q=8, H_kv=2, d_k=6, d_v=6: query (1,5,48), key/value (1,5,12).
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/8, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(1, 5, 48), ShapeLin3(1, 5, 12), ShapeLin3(1, 5, 12));

  onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 5);
  EXPECT_EQ(out[2].AsInt(), 48); // 8 * 6
  const core::symbolic::SymShape &ps = ctx.Get("present_state").Shape();
  EXPECT_EQ(ps[1].AsInt(), 2);
  EXPECT_EQ(ps[2].AsInt(), 6);
  EXPECT_EQ(ps[3].AsInt(), 6);
}

TEST(OnnxOptimShapesNnLinearAttention, WithPastStateRefinesDims) {
  // Symbolic d_v via symbolic value last dim; past_state pins d_v.
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2, /*n_inputs=*/4);
  core::shapes::ShapesContext ctx;
  ctx.Set("query", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             ShapeLin3(2, 3, 8)));
  ctx.Set("key", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                           ShapeLin3(2, 3, 8)));
  ctx.Set("value", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             ShapeLin3(2, 3, 16)));
  ctx.Set("past_state",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(2),
                                       core::symbolic::SymDim(4), core::symbolic::SymDim(8)}));

  onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value",
                                                       "past_state");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_EQ(out[2].AsInt(), 16); // 2 * d_v(=8)
  const core::symbolic::SymShape &ps = ctx.Get("present_state").Shape();
  EXPECT_EQ(ps[2].AsInt(), 4);
  EXPECT_EQ(ps[3].AsInt(), 8);
}

TEST(OnnxOptimShapesNnLinearAttention, SymbolicDimsPropagate) {
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("query", core::symbolic::SymTensor(
                       nullptr, core::symbolic::TensorType::kFloat,
                       core::symbolic::SymShape{core::symbolic::SymDim(std::string("B")),
                                                core::symbolic::SymDim(std::string("T")),
                                                core::symbolic::SymDim(16)}));
  ctx.Set("key", core::symbolic::SymTensor(
                     nullptr, core::symbolic::TensorType::kFloat,
                     core::symbolic::SymShape{core::symbolic::SymDim(std::string("B")),
                                              core::symbolic::SymDim(std::string("T")),
                                              core::symbolic::SymDim(16)}));
  ctx.Set("value", core::symbolic::SymTensor(
                       nullptr, core::symbolic::TensorType::kFloat,
                       core::symbolic::SymShape{core::symbolic::SymDim(std::string("B")),
                                                core::symbolic::SymDim(std::string("T")),
                                                core::symbolic::SymDim(32)}));

  onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  EXPECT_FALSE(out[0].IsInt());
  EXPECT_EQ(out[0].AsExpr(), "B");
  EXPECT_FALSE(out[1].IsInt());
  EXPECT_EQ(out[1].AsExpr(), "T");
  EXPECT_TRUE(out[2].IsInt());
  EXPECT_EQ(out[2].AsInt(), 32);
}

TEST(OnnxOptimShapesNnLinearAttention, RejectsMissingHeadAttributes) {
  NodeProto node;
  node.set_op_type("LinearAttention");
  node.add_input("query");
  node.add_input("key");
  node.add_input("value");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(1, 1, 4), ShapeLin3(1, 1, 4), ShapeLin3(1, 1, 4));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnLinearAttention, RejectsInvalidGqa) {
  // q_num_heads=5 not divisible by kv_num_heads=2.
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/5, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(1, 1, 20), ShapeLin3(1, 1, 8), ShapeLin3(1, 1, 8));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnLinearAttention, RejectsIndivisibleHidden) {
  // value last dim 9 not divisible by kv_num_heads=2.
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(1, 1, 8), ShapeLin3(1, 1, 8), ShapeLin3(1, 1, 9));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnLinearAttention, RejectsMismatchedHeadSize) {
  // q_d_k = 16/2 = 8, k_d_k = 12/2 = 6 -> mismatch.
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(1, 1, 16), ShapeLin3(1, 1, 12), ShapeLin3(1, 1, 16));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnLinearAttention, RejectsWrongRank) {
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("query", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                      core::symbolic::SymDim(8)}));
  ctx.Set("key", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                           ShapeLin3(1, 1, 8)));
  ctx.Set("value", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                             ShapeLin3(1, 1, 8)));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnLinearAttention, RejectsWrongOpType) {
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/2, /*kv_num_heads=*/2);
  node.set_op_type("NotLinearAttention");
  core::shapes::ShapesContext ctx;
  SetLinAttnInputs(ctx, ShapeLin3(1, 1, 8), ShapeLin3(1, 1, 8), ShapeLin3(1, 1, 8));
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesNnLinearAttention, SymbolicValueLastDimGqa) {
  // Symbolic value last dim with q_num_heads != kv_num_heads: out_last is a
  // fresh symbolic placeholder, d_v / d_k are also symbolic placeholders.
  NodeProto node = MakeLinearAttentionNode(/*q_num_heads=*/4, /*kv_num_heads=*/2);
  core::shapes::ShapesContext ctx;
  ctx.Set("query",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                       core::symbolic::SymDim(std::string("H"))}));
  ctx.Set("key", core::symbolic::SymTensor(
                     nullptr, core::symbolic::TensorType::kFloat,
                     core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                              core::symbolic::SymDim(std::string("K"))}));
  ctx.Set("value",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kFloat,
              core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(1),
                                       core::symbolic::SymDim(std::string("V"))}));

  onnx_shapes::shapes::nn::ComputeShapeLinearAttention(ctx, node, "query", "key", "value");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_FALSE(out[2].IsInt());
  const core::symbolic::SymShape &ps = ctx.Get("present_state").Shape();
  ASSERT_EQ(ps.Rank(), 4u);
  EXPECT_FALSE(ps[2].IsInt());
  EXPECT_FALSE(ps[3].IsInt());
}

} // namespace Test
