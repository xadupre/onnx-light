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
  core::symbolic::SymShape shape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeSigmoid(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathSoftmax, PropagatesTypeAndShapeWithDefaultAxisOpset13) {
  NodeProto node = MakeUnaryNode("Softmax", "input", "output");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeSoftmax(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathSoftmax, UsesOpsetDependentDefaultAxis) {
  NodeProto node = MakeUnaryNode("Softmax");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 11);
  core::symbolic::SymShape shape{core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

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
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSoftmax(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathLogSoftmax, PropagatesTypeAndShapeWithDefaultAxisOpset13) {
  NodeProto node = MakeUnaryNode("LogSoftmax", "input", "output");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeLogSoftmax(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathLogSoftmax, UsesOpsetDependentDefaultAxis) {
  NodeProto node = MakeUnaryNode("LogSoftmax");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 11);
  core::symbolic::SymShape shape{core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  // Under opset 11 the default axis is 1, which is out of range for a rank-1 input.
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeLogSoftmax(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathLogSoftmax, RejectsOutOfRangeAxis) {
  NodeProto node = MakeUnaryNode("LogSoftmax");
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(2);

  onnx_optim::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeLogSoftmax(ctx, node, "X"),
               std::invalid_argument);
}

} // namespace Test
