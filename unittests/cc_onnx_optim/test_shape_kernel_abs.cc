// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/abs.h"
#include "onnx_optim/shapes/shape_kernel.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeAbsNode(const std::string &domain = "") {
  NodeProto node;
  node.set_op_type("Abs");
  node.set_domain(domain);
  node.add_input("X");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathAbs, PropagatesFullyKnownShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::math::AbsShapeKernel kernel(node);

  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kFloat, shape);

  onnx_optim::OptimTensor output = kernel.Run(input);
  EXPECT_EQ(output.Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(output.Shape(), shape);
  EXPECT_TRUE(output.IsNull());
}

TEST(OnnxOptimShapesMathAbs, PropagatesSymbolicShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::math::AbsShapeKernel kernel(node);

  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kInt64, shape);

  onnx_optim::OptimTensor output = kernel.Run(input);
  EXPECT_EQ(output.Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(output.Shape().Rank(), 2u);
  EXPECT_EQ(output.Shape()[0].AsExpr(), "N");
  EXPECT_EQ(output.Shape()[1].AsInt(), 4);
}

TEST(OnnxOptimShapesMathAbs, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Neg");
  EXPECT_THROW(onnx_optim::shapes::math::AbsShapeKernel{node}, std::invalid_argument);
}

TEST(OnnxOptimShapesMathAbs, FactoryBuildsAbsKernel) {
  NodeProto node = MakeAbsNode();
  std::unique_ptr<onnx_optim::shapes::ShapeKernel> kernel =
      onnx_optim::shapes::MakeShapeKernel(node);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->OpType(), "Abs");

  onnx_optim::OptimShape shape{onnx_optim::OptimDim(5)};
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kDouble, shape);
  onnx_optim::OptimTensor output = kernel->Run(input);
  EXPECT_EQ(output.Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(output.Shape(), shape);
}

TEST(OnnxOptimShapesMathAbs, FactoryAcceptsExplicitOnnxDomain) {
  NodeProto node = MakeAbsNode("ai.onnx");
  auto kernel = onnx_optim::shapes::MakeShapeKernel(node);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->OpType(), "Abs");
  EXPECT_EQ(kernel->Domain(), "ai.onnx");
}

TEST(OnnxOptimShapesMathAbs, FactoryRejectsUnknownOp) {
  NodeProto node;
  node.set_op_type("ThisOpDoesNotExist");
  EXPECT_THROW(onnx_optim::shapes::MakeShapeKernel(node), std::runtime_error);
}

} // namespace Test
