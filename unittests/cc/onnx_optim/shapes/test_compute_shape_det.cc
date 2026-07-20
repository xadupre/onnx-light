// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_optim/shapes/math/shape_math.h"

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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(3), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(y.Shape().Rank(), 0u);
}

TEST(OnnxOptimShapesMathDet, DropsTrailingTwoDimensionsForBatchedInput) {
  NodeProto node = MakeDetNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim("B"), core::symbolic::SymDim(2),
                                 core::symbolic::SymDim(4), core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymTensor &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), core::symbolic::TensorType::kDouble);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  EXPECT_EQ(y.Shape()[0], core::symbolic::SymDim("B"));
  EXPECT_EQ(y.Shape()[1], core::symbolic::SymDim(2));
}

TEST(OnnxOptimShapesMathDet, RejectsRankLessThanTwo) {
  NodeProto node = MakeDetNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathDet, RejectsMismatchedInnerDimensions) {
  NodeProto node = MakeDetNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDet(ctx, node, "X"), std::invalid_argument);
}

} // namespace Test
