// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/shape_broadcast.h"
#include "onnx_optim/shapes/shapes_context.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeAbsNode(const std::string &input_name = "X", const std::string &output_name = "Y") {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathAbs, PropagatesFullyKnownShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAbs, PropagatesSymbolicShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));
}

TEST(OnnxOptimShapesMathAbs, UsesNodeOutputNameAsKey) {
  NodeProto node = MakeAbsNode("input0", "result");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(5)};
  ctx.Set("input0", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "input0");

  EXPECT_TRUE(ctx.Has("result"));
  EXPECT_FALSE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("result"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAbs, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Neg");
  node.add_input("X");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAbs, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAbs, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesContext, OpsetVersionDefaultsToUnknown) {
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_FALSE(ctx.HasOpsetVersion("ai.onnx"));
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), onnx_optim::shapes::kUnknownOpsetVersion);
  EXPECT_EQ(ctx.OpsetVersion(""), onnx_optim::shapes::kUnknownOpsetVersion);
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx.ml"), onnx_optim::shapes::kUnknownOpsetVersion);
}

TEST(OnnxOptimShapesContext, OpsetVersionStoreAndRetrieve) {
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  ctx.SetOpsetVersion("ai.onnx.ml", 3);

  EXPECT_TRUE(ctx.HasOpsetVersion("ai.onnx"));
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 13);
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx.ml"), 3);
  EXPECT_EQ(ctx.Opsets().size(), 2u);

  // Empty domain is normalised to ai.onnx.
  EXPECT_TRUE(ctx.HasOpsetVersion(""));
  EXPECT_EQ(ctx.OpsetVersion(""), 13);

  // Replacing an existing entry overwrites the recorded version.
  ctx.SetOpsetVersion("ai.onnx", 22);
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 22);
  EXPECT_EQ(ctx.Opsets().size(), 2u);
}

TEST(OnnxOptimShapesContext, ClearAlsoClearsOpsets) {
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.SetOpsetVersion("ai.onnx", 13);
  ASSERT_FALSE(ctx.Empty());
  ASSERT_TRUE(ctx.HasOpsetVersion("ai.onnx"));

  ctx.Clear();
  EXPECT_TRUE(ctx.Empty());
  EXPECT_FALSE(ctx.HasOpsetVersion("ai.onnx"));
  EXPECT_TRUE(ctx.Opsets().empty());
}

TEST(OnnxOptimShapesMathAbs, WorksWithOpsetVersionRecordedInContext) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  // The opset entry must be left intact by ComputeShape*.
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 13);
}

namespace {

NodeProto MakeAddNode(const std::string &a = "A", const std::string &b = "B",
                      const std::string &out = "C") {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesBroadcast, EqualConcreteShapes) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape out = onnx_optim::shapes::BroadcastShapes(a, b);
  EXPECT_EQ(out, a);
}

TEST(OnnxOptimShapesBroadcast, OneIsPrefix) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                           onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape out = onnx_optim::shapes::BroadcastShapes(a, b);
  EXPECT_EQ(out, a);
  // Commutativity.
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(b, a), a);
}

TEST(OnnxOptimShapesBroadcast, OneDimIsOne) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1),
                           onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                           onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                  onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
}

TEST(OnnxOptimShapesBroadcast, ScalarBroadcastsAgainstEverything) {
  onnx_optim::OptimShape scalar{};
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(scalar, a), a);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, scalar), a);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(scalar, scalar), scalar);
}

TEST(OnnxOptimShapesBroadcast, SymbolicEqualDims) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), a);
}

TEST(OnnxOptimShapesBroadcast, SymbolicVsOnePicksSymbolic) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(1), onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("M")};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
}

TEST(OnnxOptimShapesBroadcast, SymbolicVsConcreteNonOnePicksConcrete) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N")};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(5)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(b, a), expected);
}

TEST(OnnxOptimShapesBroadcast, TwoDifferentSymbolicProducesSynthesisedExpr) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N")};
  onnx_optim::OptimShape b{onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape out = onnx_optim::shapes::BroadcastShapes(a, b);
  ASSERT_EQ(out.Rank(), 1u);
  ASSERT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "broadcast(N, M)");
}

TEST(OnnxOptimShapesBroadcast, IncompatibleConcreteThrows) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)};
  EXPECT_THROW(onnx_optim::shapes::BroadcastShapes(a, b), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAdd, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathAdd, BroadcastsShapes) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1),
                                 onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                  onnx_optim::OptimDim(4)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_b));

  onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesMathAdd, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Sub");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathAdd, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("A");
  node.add_input("B");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathAdd, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAdd, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"),
               std::invalid_argument);
}

} // namespace Test
