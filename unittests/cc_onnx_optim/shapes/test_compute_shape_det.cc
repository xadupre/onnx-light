// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/shapes_context.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeDetNode(const std::string &input_name = "X", const std::string &output_name = "Y") {
  NodeProto node;
  node.set_op_type("Det");
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathDet, ScalarOutputFor2DInput) {
  NodeProto node = MakeDetNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimTensor &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(y.Shape().Rank(), 0u);
}

TEST(OnnxOptimShapesMathDet, DropsTrailingTwoDimensionsForBatchedInput) {
  NodeProto node = MakeDetNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("B"), onnx_optim::OptimDim(2),
                               onnx_optim::OptimDim(4), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimTensor &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), onnx_optim::TensorType::kDouble);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  EXPECT_EQ(y.Shape()[0], onnx_optim::OptimDim("B"));
  EXPECT_EQ(y.Shape()[1], onnx_optim::OptimDim(2));
}

TEST(OnnxOptimShapesMathDet, RejectsRankLessThanTwo) {
  NodeProto node = MakeDetNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathDet, RejectsMismatchedInnerDimensions) {
  NodeProto node = MakeDetNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X"), std::invalid_argument);
}

} // namespace Test
