// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/quantization/shape_quantization.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeQuantizeLinearNode(bool with_zero_point, int64_t output_dtype_attr = 0) {
  NodeProto node;
  node.set_op_type("QuantizeLinear");
  node.add_input("x");
  node.add_input("y_scale");
  if (with_zero_point) {
    node.add_input("y_zero_point");
  }
  node.add_output("y");
  if (output_dtype_attr != 0) {
    AddAttribute<int64_t>(node, "output_dtype", output_dtype_attr);
  }
  return node;
}

void SetX(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &shape,
          onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, dtype, shape));
}

void SetScale(onnx_optim::shapes::ShapesContext &ctx,
              onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("y_scale", onnx_optim::OptimTensor(nullptr, dtype, onnx_optim::OptimShape{}));
}

void SetZeroPoint(onnx_optim::shapes::ShapesContext &ctx, onnx_optim::TensorType dtype) {
  ctx.Set("y_zero_point", onnx_optim::OptimTensor(nullptr, dtype, onnx_optim::OptimShape{}));
}

} // namespace

TEST(OnnxOptimShapesQuantizationQuantizeLinear, DefaultsToUint8WhenZeroPointOmitted) {
  // Mirrors the upstream ``test_quantizelinear`` node test: no
  // ``y_zero_point`` input, so the output element type defaults to uint8.
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/false);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(6)});
  SetScale(ctx);

  onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", nullptr);

  ASSERT_TRUE(ctx.Has("y"));
  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 1u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 6);
  EXPECT_EQ(out.Dtype(), onnx_optim::TensorType::kUint8);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, FollowsZeroPointDtypeInt8) {
  // Mirrors ``test_quantizelinear_int8``: an INT8 ``y_zero_point``
  // forces an INT8 output.
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(6)});
  SetScale(ctx);
  SetZeroPoint(ctx, onnx_optim::TensorType::kInt8);

  onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", "y_zero_point");

  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt8);
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 6);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, FollowsZeroPointDtypeFloat8) {
  // Mirrors ``test_quantizelinear_e4m3fn``: FLOAT8E4M3FN zero point yields
  // a FLOAT8E4M3FN output.
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  SetScale(ctx);
  SetZeroPoint(ctx, onnx_optim::TensorType::kFloat8e4m3fn);

  onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", "y_zero_point");

  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat8e4m3fn);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 2);
  EXPECT_EQ(ctx.Get("y").Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, FollowsZeroPointDtypeInt4) {
  // Mirrors ``test_quantizelinear_int4``: INT4 zero point yields INT4 output.
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(5)});
  SetScale(ctx);
  SetZeroPoint(ctx, onnx_optim::TensorType::kInt4);

  onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", "y_zero_point");

  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt4);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, UsesOutputDtypeAttribute) {
  // Mirrors the opset-23+ ``output_dtype`` attribute: when
  // ``y_zero_point`` is omitted, the attribute selects the element type.
  NodeProto node = MakeQuantizeLinearNode(
      /*with_zero_point=*/false,
      /*output_dtype_attr=*/static_cast<int64_t>(TensorProto::DataType::INT16));
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)});
  SetScale(ctx);

  onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", nullptr);

  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kInt16);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, PropagatesSymbolicShape) {
  // Per-axis quantization preserves the input shape regardless of
  // ``y_scale``/``y_zero_point`` rank.
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(8),
                                   onnx_optim::OptimDim("H"), onnx_optim::OptimDim("W")});
  SetScale(ctx);
  SetZeroPoint(ctx, onnx_optim::TensorType::kUint8);

  onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", "y_zero_point");

  const onnx_optim::OptimShape &out = ctx.Get("y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_EQ(out[1].AsInt(), 8);
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_EQ(out[2].AsExpr(), "H");
  EXPECT_TRUE(out[3].IsExpr());
  EXPECT_EQ(out[3].AsExpr(), "W");
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("DequantizeLinear");
  node.add_input("x");
  node.add_input("y_scale");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  SetScale(ctx);
  EXPECT_THROW(
      onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", nullptr),
      std::invalid_argument);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, RejectsInvalidOutputDtype) {
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/false, /*output_dtype_attr=*/9999);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  SetScale(ctx);
  EXPECT_THROW(
      onnx_optim::shapes::quantization::ComputeShapeQuantizeLinear(ctx, node, "x", nullptr),
      std::invalid_argument);
}

TEST(OnnxOptimShapesQuantizationQuantizeLinear, DispatchesViaComputeShapeNode) {
  // End-to-end: ComputeShapeNode should route QuantizeLinear through the
  // dispatch table to the per-op trampoline.
  NodeProto node = MakeQuantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetX(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(4)});
  SetScale(ctx);
  SetZeroPoint(ctx, onnx_optim::TensorType::kUint16);

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kUint16);
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 4);
}

namespace {

NodeProto MakeDequantizeLinearNode(bool with_zero_point, int64_t output_dtype_attr = 0) {
  NodeProto node;
  node.set_op_type("DequantizeLinear");
  node.add_input("x");
  node.add_input("x_scale");
  if (with_zero_point) {
    node.add_input("x_zero_point");
  }
  node.add_output("y");
  if (output_dtype_attr != 0) {
    AddAttribute<int64_t>(node, "output_dtype", output_dtype_attr);
  }
  return node;
}

void SetXDeq(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &shape,
             onnx_optim::TensorType dtype = onnx_optim::TensorType::kUint8) {
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, dtype, shape));
}

void SetScaleDeq(onnx_optim::shapes::ShapesContext &ctx,
                 onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("x_scale", onnx_optim::OptimTensor(nullptr, dtype, onnx_optim::OptimShape{}));
}

void SetZeroPointDeq(onnx_optim::shapes::ShapesContext &ctx, onnx_optim::TensorType dtype) {
  ctx.Set("x_zero_point", onnx_optim::OptimTensor(nullptr, dtype, onnx_optim::OptimShape{}));
}

} // namespace

TEST(OnnxOptimShapesQuantizationDequantizeLinear, DefaultsToScaleDtypeWhenZeroPointOmitted) {
  // Without ``output_dtype`` and without ``x_zero_point``, the output dtype
  // mirrors the ``x_scale`` dtype.
  NodeProto node = MakeDequantizeLinearNode(/*with_zero_point=*/false);
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(6)});
  SetScaleDeq(ctx);

  onnx_optim::shapes::quantization::ComputeShapeDequantizeLinear(ctx, node, "x", "x_scale");

  ASSERT_TRUE(ctx.Has("y"));
  const onnx_optim::OptimTensor &out = ctx.Get("y");
  ASSERT_EQ(out.Shape().Rank(), 1u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 6);
  EXPECT_EQ(out.Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesQuantizationDequantizeLinear, FollowsScaleDtypeFloat16) {
  NodeProto node = MakeDequantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)},
          onnx_optim::TensorType::kInt8);
  SetScaleDeq(ctx, onnx_optim::TensorType::kFloat16);
  SetZeroPointDeq(ctx, onnx_optim::TensorType::kInt8);

  onnx_optim::shapes::quantization::ComputeShapeDequantizeLinear(ctx, node, "x", "x_scale");

  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat16);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 2);
  EXPECT_EQ(ctx.Get("y").Shape()[1].AsInt(), 3);
}

TEST(OnnxOptimShapesQuantizationDequantizeLinear, UsesOutputDtypeAttribute) {
  // ``output_dtype`` overrides the ``x_scale`` dtype.
  NodeProto node = MakeDequantizeLinearNode(
      /*with_zero_point=*/false,
      /*output_dtype_attr=*/static_cast<int64_t>(TensorProto::DataType::BFLOAT16));
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)});
  SetScaleDeq(ctx);

  onnx_optim::shapes::quantization::ComputeShapeDequantizeLinear(ctx, node, "x", "x_scale");

  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kBfloat16);
}

TEST(OnnxOptimShapesQuantizationDequantizeLinear, PropagatesSymbolicShape) {
  NodeProto node = MakeDequantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(8),
                                      onnx_optim::OptimDim("H"), onnx_optim::OptimDim("W")});
  SetScaleDeq(ctx);
  SetZeroPointDeq(ctx, onnx_optim::TensorType::kUint8);

  onnx_optim::shapes::quantization::ComputeShapeDequantizeLinear(ctx, node, "x", "x_scale");

  const onnx_optim::OptimShape &out = ctx.Get("y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_EQ(out[1].AsInt(), 8);
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_EQ(out[2].AsExpr(), "H");
  EXPECT_TRUE(out[3].IsExpr());
  EXPECT_EQ(out[3].AsExpr(), "W");
}

TEST(OnnxOptimShapesQuantizationDequantizeLinear, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("QuantizeLinear");
  node.add_input("x");
  node.add_input("x_scale");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  SetScaleDeq(ctx);
  EXPECT_THROW(
      onnx_optim::shapes::quantization::ComputeShapeDequantizeLinear(ctx, node, "x", "x_scale"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesQuantizationDequantizeLinear, RejectsInvalidOutputDtype) {
  NodeProto node = MakeDequantizeLinearNode(/*with_zero_point=*/false, /*output_dtype_attr=*/9999);
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  SetScaleDeq(ctx);
  EXPECT_THROW(
      onnx_optim::shapes::quantization::ComputeShapeDequantizeLinear(ctx, node, "x", "x_scale"),
      std::invalid_argument);
}

TEST(OnnxOptimShapesQuantizationDequantizeLinear, DispatchesViaComputeShapeNode) {
  // End-to-end: ComputeShapeNode should route DequantizeLinear through the
  // dispatch table to the per-op trampoline.
  NodeProto node = MakeDequantizeLinearNode(/*with_zero_point=*/true);
  onnx_optim::shapes::ShapesContext ctx;
  SetXDeq(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(4)}, onnx_optim::TensorType::kUint8);
  SetScaleDeq(ctx);
  SetZeroPointDeq(ctx, onnx_optim::TensorType::kUint8);

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsInt(), 4);
}

TEST(OnnxOptimShapesQuantizationDynamicQuantizeLinear, AllThreeOutputsTypedAndShaped) {
  // ``DynamicQuantizeLinear`` has 1 input ``x`` (float) and 3 outputs:
  //   y           : uint8, shape == x
  //   y_scale     : scalar float
  //   y_zero_point: scalar uint8
  NodeProto node;
  node.set_op_type("DynamicQuantizeLinear");
  node.add_input("x");
  node.add_output("y");
  node.add_output("y_scale");
  node.add_output("y_zero_point");

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("x", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::quantization::ComputeShapeDynamicQuantizeLinear(ctx, node, "x");

  ASSERT_TRUE(ctx.Has("y"));
  EXPECT_EQ(ctx.Get("y").Dtype(), onnx_optim::TensorType::kUint8);
  ASSERT_EQ(ctx.Get("y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("y").Shape()[0].AsExpr(), "N");
  EXPECT_EQ(ctx.Get("y").Shape()[1].AsInt(), 4);

  ASSERT_TRUE(ctx.Has("y_scale"));
  EXPECT_EQ(ctx.Get("y_scale").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("y_scale").Shape().Rank(), 0u);

  ASSERT_TRUE(ctx.Has("y_zero_point"));
  EXPECT_EQ(ctx.Get("y_zero_point").Dtype(), onnx_optim::TensorType::kUint8);
  EXPECT_EQ(ctx.Get("y_zero_point").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapesQuantizationDynamicQuantizeLinear, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("QuantizeLinear");
  node.add_input("x");
  node.add_output("y");
  node.add_output("y_scale");
  node.add_output("y_zero_point");

  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("x", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::quantization::ComputeShapeDynamicQuantizeLinear(ctx, node, "x"),
               std::invalid_argument);
}

} // namespace Test
