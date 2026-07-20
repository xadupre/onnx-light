// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/onnx_shapes/shapes/math/shape_math.h"

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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)};
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::math::ComputeShapeShrink(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathShrink, PropagatesIntegerType) {
  NodeProto node = MakeShrinkNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(3), core::symbolic::SymDim(2)};
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape));

  onnx_shapes::shapes::math::ComputeShapeShrink(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape));
}

} // namespace Test
