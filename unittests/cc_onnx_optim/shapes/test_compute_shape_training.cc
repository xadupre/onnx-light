// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/training/shape_training.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// Builds an Adam node with ``n`` optimised tensors. Each optimised tensor
// ``X_i`` gets matching ``G_i``, ``V_i`` and ``H_i`` inputs, and the node
// declares the corresponding ``3 * n`` outputs.
NodeProto MakeAdamNode(int n) {
  NodeProto node;
  node.set_op_type("Adam");
  node.set_domain("ai.onnx.preview.training");
  node.add_input("R");
  node.add_input("T");
  for (int i = 0; i < n; ++i) {
    node.add_input("X" + std::to_string(i + 1));
  }
  for (int i = 0; i < n; ++i) {
    node.add_input("G" + std::to_string(i + 1));
  }
  for (int i = 0; i < n; ++i) {
    node.add_input("V" + std::to_string(i + 1));
  }
  for (int i = 0; i < n; ++i) {
    node.add_input("H" + std::to_string(i + 1));
  }
  for (int i = 0; i < n; ++i) {
    node.add_output("X" + std::to_string(i + 1) + "_new");
  }
  for (int i = 0; i < n; ++i) {
    node.add_output("V" + std::to_string(i + 1) + "_new");
  }
  for (int i = 0; i < n; ++i) {
    node.add_output("H" + std::to_string(i + 1) + "_new");
  }
  return node;
}

void SeedScalar(onnx_optim::shapes::ShapesContext &ctx, const std::string &name,
                onnx_optim::TensorType dtype) {
  ctx.Set(name, onnx_optim::OptimTensor(nullptr, dtype, onnx_optim::OptimShape{}));
}

void SeedTensor(onnx_optim::shapes::ShapesContext &ctx, const std::string &name,
                onnx_optim::TensorType dtype, onnx_optim::OptimShape shape) {
  ctx.Set(name, onnx_optim::OptimTensor(nullptr, dtype, std::move(shape)));
}

} // namespace

TEST(OnnxOptimShapeAdam, PropagatesShapesForSingleOptimizedTensor) {
  NodeProto node = MakeAdamNode(1);

  onnx_optim::shapes::ShapesContext ctx;
  SeedScalar(ctx, "R", onnx_optim::TensorType::kFloat);
  SeedScalar(ctx, "T", onnx_optim::TensorType::kInt64);
  const onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  SeedTensor(ctx, "X1", onnx_optim::TensorType::kFloat, shape);
  SeedTensor(ctx, "G1", onnx_optim::TensorType::kFloat, shape);
  SeedTensor(ctx, "V1", onnx_optim::TensorType::kFloat, shape);
  SeedTensor(ctx, "H1", onnx_optim::TensorType::kFloat, shape);

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("X1_new"));
  ASSERT_TRUE(ctx.Has("V1_new"));
  ASSERT_TRUE(ctx.Has("H1_new"));
  EXPECT_EQ(ctx.Get("X1_new").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("X1_new").Shape(), shape);
  EXPECT_EQ(ctx.Get("V1_new").Shape(), shape);
  EXPECT_EQ(ctx.Get("H1_new").Shape(), shape);
}

TEST(OnnxOptimShapeAdam, PropagatesShapesForMultipleOptimizedTensors) {
  NodeProto node = MakeAdamNode(2);

  onnx_optim::shapes::ShapesContext ctx;
  SeedScalar(ctx, "R", onnx_optim::TensorType::kFloat);
  SeedScalar(ctx, "T", onnx_optim::TensorType::kInt64);
  const onnx_optim::OptimShape shape_x1{onnx_optim::OptimDim(4)};
  const onnx_optim::OptimShape shape_x2{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)};
  SeedTensor(ctx, "X1", onnx_optim::TensorType::kFloat, shape_x1);
  SeedTensor(ctx, "X2", onnx_optim::TensorType::kDouble, shape_x2);
  SeedTensor(ctx, "G1", onnx_optim::TensorType::kFloat, shape_x1);
  SeedTensor(ctx, "G2", onnx_optim::TensorType::kDouble, shape_x2);
  SeedTensor(ctx, "V1", onnx_optim::TensorType::kFloat, shape_x1);
  SeedTensor(ctx, "V2", onnx_optim::TensorType::kDouble, shape_x2);
  SeedTensor(ctx, "H1", onnx_optim::TensorType::kFloat, shape_x1);
  SeedTensor(ctx, "H2", onnx_optim::TensorType::kDouble, shape_x2);

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  // Outputs are laid out as [X1_new, X2_new, V1_new, V2_new, H1_new, H2_new]
  // and each output mirrors the dtype and shape of its corresponding input.
  EXPECT_EQ(ctx.Get("X1_new").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("X1_new").Shape(), shape_x1);
  EXPECT_EQ(ctx.Get("X2_new").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("X2_new").Shape(), shape_x2);
  EXPECT_EQ(ctx.Get("V1_new").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("V1_new").Shape(), shape_x1);
  EXPECT_EQ(ctx.Get("V2_new").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("V2_new").Shape(), shape_x2);
  EXPECT_EQ(ctx.Get("H1_new").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("H1_new").Shape(), shape_x1);
  EXPECT_EQ(ctx.Get("H2_new").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("H2_new").Shape(), shape_x2);
}

TEST(OnnxOptimShapeAdam, RejectsInputCountNotMultipleOfFour) {
  // 2 (R, T) + 5 variadic inputs = not a multiple of 4 for the variadic part.
  NodeProto node;
  node.set_op_type("Adam");
  node.set_domain("ai.onnx.preview.training");
  node.add_input("R");
  node.add_input("T");
  for (int i = 0; i < 5; ++i) {
    node.add_input("V" + std::to_string(i));
  }
  node.add_output("Y0");
  node.add_output("Y1");
  node.add_output("Y2");

  onnx_optim::shapes::ShapesContext ctx;
  SeedScalar(ctx, "R", onnx_optim::TensorType::kFloat);
  SeedScalar(ctx, "T", onnx_optim::TensorType::kInt64);
  for (int i = 0; i < 5; ++i) {
    SeedTensor(ctx, "V" + std::to_string(i), onnx_optim::TensorType::kFloat,
               onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  }

  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeAdam, RejectsWrongOutputCount) {
  // Build a 1-optimised-tensor Adam node but declare only 2 outputs instead
  // of the expected 3 (X_new, V_new, H_new).
  NodeProto node;
  node.set_op_type("Adam");
  node.set_domain("ai.onnx.preview.training");
  node.add_input("R");
  node.add_input("T");
  node.add_input("X1");
  node.add_input("G1");
  node.add_input("V1");
  node.add_input("H1");
  node.add_output("X1_new");
  node.add_output("V1_new");

  onnx_optim::shapes::ShapesContext ctx;
  SeedScalar(ctx, "R", onnx_optim::TensorType::kFloat);
  SeedScalar(ctx, "T", onnx_optim::TensorType::kInt64);
  const onnx_optim::OptimShape shape{onnx_optim::OptimDim(3)};
  SeedTensor(ctx, "X1", onnx_optim::TensorType::kFloat, shape);
  SeedTensor(ctx, "G1", onnx_optim::TensorType::kFloat, shape);
  SeedTensor(ctx, "V1", onnx_optim::TensorType::kFloat, shape);
  SeedTensor(ctx, "H1", onnx_optim::TensorType::kFloat, shape);

  EXPECT_THROW(onnx_optim::shapes::ComputeShapeNode(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapeAdam, DirectCallRejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotAdam");
  node.add_output("Y");

  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::training::ComputeShapeAdam(ctx, node), std::invalid_argument);
}

} // namespace Test
