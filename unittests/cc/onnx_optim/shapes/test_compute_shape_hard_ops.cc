// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_optim/shapes/math/shape_math.h"

#include <gtest/gtest.h>

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

NodeProto MakeHardmaxNode(int64_t axis, const std::string &input_name = "input",
                          const std::string &output_name = "output") {
  NodeProto node;
  node.set_op_type("Hardmax");
  node.add_input(input_name);
  node.add_output(output_name);
  AttributeProto *axis_attr = node.add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::INT);
  axis_attr->set_i(axis);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathHardSigmoid, PropagatesTypeAndShape) {
  NodeProto node = MakeUnaryNode("HardSigmoid");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeHardSigmoid(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathHardSwish, PropagatesTypeAndShape) {
  NodeProto node = MakeUnaryNode("HardSwish");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeHardSwish(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathHardmax, PropagatesTypeAndShapeAndAcceptsValidAxis) {
  NodeProto node = MakeHardmaxNode(/*axis=*/-1);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeHardmax(ctx, node, "input");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output"),
            core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathHardmax, RejectsOutOfRangeAxis) {
  NodeProto node = MakeHardmaxNode(/*axis=*/5);
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeHardmax(ctx, node, "input"), std::exception);
}

} // namespace Test
