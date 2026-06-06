// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/shapes_context.h"

#include <gtest/gtest.h>

#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeShrinkNode(const std::string &input_name = "input",
                         const std::string &output_name = "output") {
  NodeProto node;
  node.set_op_type("Shrink");
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathShrink, PropagatesTypeAndShape) {
  NodeProto node = MakeShrinkNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeShrink(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathShrink, PropagatesIntegerType) {
  NodeProto node = MakeShrinkNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(2)};
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));

  onnx_optim::shapes::math::ComputeShapeShrink(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));
}

} // namespace Test
