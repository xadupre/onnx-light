// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/shapes/logical/shape_logical.h"

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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));

  onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalAnd, BroadcastsShapes) {
  NodeProto node = MakeAndNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalAnd, SymbolicBroadcast) {
  NodeProto node = MakeAndNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim("N"), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim("M")};
  core::symbolic::SymShape expected{core::symbolic::SymDim("N"), core::symbolic::SymDim("M")};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalAnd, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Or");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalAnd, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("A");
  node.add_input("B");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalAnd, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeAndNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalAnd, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeAndNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeAnd(ctx, node, "A", "B"),
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));

  onnx_shapes::shapes::logical::ComputeShapeOr(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalOr, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeOr(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalOr, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("And");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeOr(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalOr, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeOr(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalOr, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Or");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeOr(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Xor
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalXor, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));

  onnx_shapes::shapes::logical::ComputeShapeXor(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalXor, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeXor(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalXor, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("And");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeXor(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalXor, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeXor(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalXor, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Xor");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeXor(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Greater
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalGreater, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalGreater, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalGreater, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalGreater, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalGreater, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeGreater(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Less
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalLess, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::logical::ComputeShapeLess(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalLess, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeLess(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalLess, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeLess(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalLess, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeLess(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalLess, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeLess(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// GreaterOrEqual
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalGreaterOrEqual, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("GreaterOrEqual");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::logical::ComputeShapeGreaterOrEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalGreaterOrEqual, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("GreaterOrEqual");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeGreaterOrEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalGreaterOrEqual, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeGreaterOrEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LessOrEqual
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalLessOrEqual, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("LessOrEqual");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::logical::ComputeShapeLessOrEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalLessOrEqual, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("LessOrEqual");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeLessOrEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalLessOrEqual, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Less");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeLessOrEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Equal
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalEqual, PropagatesEqualShapesWithBoolDtype) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape));

  onnx_shapes::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalEqual, BroadcastsShapes) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalEqual, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Greater");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalEqual, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalEqual, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryLogicalNode("Equal");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeEqual(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Where
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesLogicalWhere, BroadcastsThreeInputShapesAndPropagatesDataType) {
  NodeProto node = MakeWhereNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("condition",
          core::symbolic::SymTensor(
              nullptr, core::symbolic::TensorType::kBool,
              core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(1)}));
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt32,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  ctx.Set("Y", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kInt32,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(3)}));

  onnx_shapes::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y");

  ASSERT_TRUE(ctx.Has("output"));
  EXPECT_EQ(ctx.Get("output").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("output").Shape(),
            (core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
}

TEST(OnnxOptimShapesLogicalWhere, RejectsWrongOpType) {
  NodeProto node = MakeBinaryLogicalNode("Equal", "condition", "X", "output");
  node.add_input("Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("condition", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalWhere, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeWhereNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("condition", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y"),
               std::out_of_range);
}

TEST(OnnxOptimShapesLogicalWhere, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeWhereNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("condition",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                    core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("Y", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeWhere(ctx, node, "condition", "X", "Y"),
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape));

  onnx_shapes::shapes::logical::ComputeShapeBitwiseAnd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalBitwiseOr, BroadcastsShapes) {
  NodeProto node = MakeBinaryBitwiseNode("BitwiseOr");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(4)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(4)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kUint64, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kUint64, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeBitwiseOr(ctx, node, "A", "B");

  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kUint64);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesLogicalBitShift, BroadcastsSignedIntegerShapes) {
  NodeProto node = MakeBinaryBitwiseNode("BitShift");
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                   core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32, shape_b));

  onnx_shapes::shapes::logical::ComputeShapeBitShift(ctx, node, "A", "B");

  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
}

TEST(OnnxOptimShapesLogicalBitwiseXor, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryBitwiseNode("BitwiseXor");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt16,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt16,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeBitwiseXor(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalBitwiseNot, PropagatesShapeAndDtype) {
  NodeProto node = MakeBitwiseNotNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kUint8, shape));

  onnx_shapes::shapes::logical::ComputeShapeBitwiseNot(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kUint8);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalBitwiseNot, ThrowsOnWrongOpType) {
  NodeProto node = MakeBitwiseNotNode();
  node.set_op_type("Not");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt32,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeBitwiseNot(ctx, node, "X"),
               std::invalid_argument);
}

namespace {

NodeProto MakeNotNode(const std::string &x = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Not");
  node.add_input(x);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesLogicalNot, PropagatesShapeAndDtype) {
  NodeProto node = MakeNotNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(30), core::symbolic::SymDim(4),
                                 core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, shape));

  onnx_shapes::shapes::logical::ComputeShapeNot(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalNot, ThrowsOnWrongOpType) {
  NodeProto node = MakeNotNode();
  node.set_op_type("BitwiseNot");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeNot(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalNot, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeNotNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeNot(ctx, node, "X"), std::out_of_range);
}

namespace {

NodeProto MakeIsNaNNode(const std::string &x = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("IsNaN");
  node.add_input(x);
  node.add_output(out);
  return node;
}

NodeProto MakeIsInfNode(const std::string &x = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("IsInf");
  node.add_input(x);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesLogicalIsNaN, PropagatesShapeAndBoolDtype) {
  NodeProto node = MakeIsNaNNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(3), core::symbolic::SymDim("N")};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, shape));

  onnx_shapes::shapes::logical::ComputeShapeIsNaN(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalIsNaN, ThrowsOnWrongOpType) {
  NodeProto node = MakeIsNaNNode();
  node.set_op_type("IsInf");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeIsNaN(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalIsNaN, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeIsNaNNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeIsNaN(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesLogicalIsInf, PropagatesShapeAndBoolDtype) {
  NodeProto node = MakeIsInfNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(4), core::symbolic::SymDim(5)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kDouble, shape));

  onnx_shapes::shapes::logical::ComputeShapeIsInf(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesLogicalIsInf, ThrowsOnWrongOpType) {
  NodeProto node = MakeIsInfNode();
  node.set_op_type("IsNaN");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeIsInf(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesLogicalIsInf, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeIsInfNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::logical::ComputeShapeIsInf(ctx, node, "X"), std::out_of_range);
}

} // namespace Test
