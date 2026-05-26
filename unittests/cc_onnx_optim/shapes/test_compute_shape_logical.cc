// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/shapes_context.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeAndNode(const std::string &a = "A", const std::string &b = "B",
                      const std::string &out = "C") {
  NodeProto node;
  node.set_op_type("And");
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesLogicalAnd, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeAndNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));

  onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalAnd, BroadcastsShapes) {
  NodeProto node = MakeAndNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_b));

  onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalAnd, SymbolicBroadcast) {
  NodeProto node = MakeAndNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("M")};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_b));

  onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalAnd, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Or");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalAnd, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("A");
  node.add_input("B");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalAnd, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeAndNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalAnd, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeAndNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::invalid_argument);
}

namespace {

NodeProto MakeBinaryLogicalNode(const std::string &op_type, const std::string &a = "A",
                                const std::string &b = "B", const std::string &out = "C") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Or
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalOr, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));

  onnx_optim::shapes::logical::ComputeShapeOr(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalOr, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_b));

  onnx_optim::shapes::logical::ComputeShapeOr(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalOr, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("And");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeOr(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalOr, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeOr(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesLogicalOr, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeOr(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Xor
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalXor, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape));

  onnx_optim::shapes::logical::ComputeShapeXor(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalXor, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, shape_b));

  onnx_optim::shapes::logical::ComputeShapeXor(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalXor, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("And");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeXor(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalXor, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeXor(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalXor, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeXor(ctx, node, "A", "B"),
               std::invalid_argument);
}

} // namespace Test
