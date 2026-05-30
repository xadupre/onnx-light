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

NodeProto MakeUnaryNode(const std::string &op_type, const std::string &input_name = "X",
                        const std::string &output_name = "Y") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathSigmoid, PropagatesTypeAndShape) {
  NodeProto node = MakeUnaryNode("Sigmoid");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeSigmoid(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathSoftmax, PropagatesTypeAndShapeWithDefaultAxisOpset13) {
  NodeProto node = MakeUnaryNode("Softmax", "input", "output");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("input", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeSoftmax(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathSoftmax, UsesOpsetDependentDefaultAxis) {
  NodeProto node = MakeUnaryNode("Softmax");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 11);
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(5)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSoftmax(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathSoftmax, RejectsOutOfRangeAxis) {
  NodeProto node = MakeUnaryNode("Softmax");
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(2);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSoftmax(ctx, node, "X"),
               std::invalid_argument);
}

} // namespace Test
