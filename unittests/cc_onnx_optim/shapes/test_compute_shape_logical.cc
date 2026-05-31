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

NodeProto MakeWhereNode(const std::string &condition = "condition", const std::string &x = "X",
                        const std::string &y = "Y", const std::string &out = "output") {
  NodeProto node;
  node.set_op_type("Where");
  node.add_input(condition);
  node.add_input(x);
  node.add_input(y);
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

// ---------------------------------------------------------------------------
// Greater
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalGreater, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalGreater, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalGreater, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalGreater, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalGreater, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Less
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalLess, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::logical::ComputeShapeLess(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalLess, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::logical::ComputeShapeLess(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalLess, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeLess(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalLess, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeLess(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalLess, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeLess(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// GreaterOrEqual
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalGreaterOrEqual, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("GreaterOrEqual");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::logical::ComputeShapeGreaterOrEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalGreaterOrEqual, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("GreaterOrEqual");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::logical::ComputeShapeGreaterOrEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalGreaterOrEqual, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeGreaterOrEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Equal
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalEqual, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));

  onnx_optim::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalEqual, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape_b));

  onnx_optim::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalEqual, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalEqual, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalEqual, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Where
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalWhere, BroadcastsThreeInputShapesAndPropagatesDataType) {
  NodeProto node = MakeWhereNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("condition", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                               onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                                      onnx_optim::OptimDim(1)}));
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt32,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("Y", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt32,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("output").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesLogicalWhere, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Equal", "condition", "X", "output");
  node.add_input("Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("condition", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalWhere, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeWhereNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("condition", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool, {}));
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalWhere, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeWhereNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("condition", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                               onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("Y", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Bitwise operators (opset 18)
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeBinaryBitwiseNode(const std::string &op_type, const std::string &a = "A",
                                const std::string &b = "B", const std::string &out = "C") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

NodeProto MakeBitwiseNotNode(const std::string &x = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("BitwiseNot");
  node.add_input(x);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesLogicalBitwiseAnd, PropagatesShapeAndIntDtype) {
  NodeProto node = MakeBinaryBitwiseNode("BitwiseAnd");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));

  onnx_optim::shapes::logical::ComputeShapeBitwiseAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalBitwiseOr, BroadcastsShapes) {
  NodeProto node = MakeBinaryBitwiseNode("BitwiseOr");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kUint64, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kUint64, shape_b));

  onnx_optim::shapes::logical::ComputeShapeBitwiseOr(ctx, node, "A", "B");

  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kUint64);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalBitwiseXor, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryBitwiseNode("BitwiseXor");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt16,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt16,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeBitwiseXor(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalBitwiseNot, PropagatesShapeAndDtype) {
  NodeProto node = MakeBitwiseNotNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kUint8, shape));

  onnx_optim::shapes::logical::ComputeShapeBitwiseNot(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kUint8);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalBitwiseNot, ThrowsOnWrongOpType) {
  NodeProto node = MakeBitwiseNotNode();
  node.set_op_type("Not");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::logical::ComputeShapeBitwiseNot(ctx, node, "X"),
               std::invalid_argument);
}

} // namespace Test
