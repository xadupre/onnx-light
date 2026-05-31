// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_optim/shapes/tensor/shape_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeConcatNode(const std::vector<std::string> &inputs,
                         const std::optional<int64_t> &axis = std::nullopt,
                         const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Concat");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output(out);
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorConcat, ConcatenatesTwoFullyKnownInputsOnAxis0) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(7), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesTensorConcat, ConcatenatesThreeInputsOnLastAxis) {
  NodeProto node = MakeConcatNode({"A", "B", "C"}, /*axis=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(1)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(2)}));
  ctx.Set("C", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(8)}));
}

TEST(OnnxOptimShapesTensorConcat, NegativeAxisIsResolvedFromRank) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/-1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(7)}));
}

TEST(OnnxOptimShapesTensorConcat, SingleInputProducesCopyOfShape) {
  NodeProto node = MakeConcatNode({"A"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesTensorConcat, SymbolicAxisDimProducesSymbolicOutput) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], onnx_optim::OptimDim(3));
}

TEST(OnnxOptimShapesTensorConcat, MergesSymbolicNonConcatDimWithConcreteOne) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  // First input has a symbolic non-concat dim that should be replaced
  // by the concrete value coming from the second input.
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(2)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnMismatchedRank) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnMismatchedNonConcatDim) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnDtypeMismatch) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsOnAxisOutOfRange) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/5);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("A");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorConcat, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeConcatNode({"A", "B"}, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeConcat(ctx, node), std::out_of_range);
}

// ── Reshape ──────────────────────────────────────────────────────────────────

namespace {

NodeProto MakeReshapeNode(const std::string &data = "X", const std::string &shape = "S",
                          const std::optional<int64_t> &allowzero = std::nullopt,
                          const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Reshape");
  node.add_input(data);
  node.add_input(shape);
  node.add_output(out);
  if (allowzero.has_value()) {
    AddAttribute<int64_t>(node, "allowzero", *allowzero);
  }
  return node;
}

onnx_optim::OptimTensor MakeShapeInput(const std::vector<int64_t> &dims) {
  onnx_optim::OptimShape static_shape{onnx_optim::OptimDim(static_cast<int64_t>(dims.size()))};
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kInt64, std::move(static_shape));
  onnx_optim::OptimShape value;
  for (int64_t v : dims) {
    value.PushBack(onnx_optim::OptimDim(v));
  }
  t.SetValueAsShape(std::move(value));
  return t;
}

} // namespace

TEST(OnnxOptimShapesTensorReshape, AllPositiveTargetDimsAreCopied) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  ctx.Set("S", MakeShapeInput({6, 4}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(6), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorReshape, ZeroCopiesFromInputDataShape) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  ctx.Set("S", MakeShapeInput({0, 0, 4}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                    onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorReshape, AllowZeroHonoursLiteralZero) {
  NodeProto node = MakeReshapeNode("X", "S", /*allowzero=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(static_cast<int64_t>(0))}));
  ctx.Set("S", MakeShapeInput({0, 4}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(static_cast<int64_t>(0)),
                                    onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorReshape, NegativeOneIsInferredFromInputProduct) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  ctx.Set("S", MakeShapeInput({-1, 6}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(6)}));
}

TEST(OnnxOptimShapesTensorReshape, NegativeOneWithZeroResolvesAcrossInputDim) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  // 0 copies input[0]=2, -1 is inferred from product(3*4)=12.
  ctx.Set("S", MakeShapeInput({0, -1}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(12)}));
}

TEST(OnnxOptimShapesTensorReshape, NegativeOneIsSymbolicWhenInputHasSymbolicDim) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));
  ctx.Set("S", MakeShapeInput({-1, 2}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], onnx_optim::OptimDim(2));
}

TEST(OnnxOptimShapesTensorReshape, UnknownShapeValueProducesSymbolicRankFromShapeInput) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(6)}));
  // Shape input is 1-D of size 3 but its values are unknown.
  ctx.Set("S", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(static_cast<int64_t>(3))}));

  onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node);

  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnMultipleNegativeOnes) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(6)}));
  ctx.Set("S", MakeShapeInput({-1, -1}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnInvalidNegativeValue) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(6)}));
  ctx.Set("S", MakeShapeInput({-2, 3}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnZeroOutsideInputRank) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(6)}));
  ctx.Set("S", MakeShapeInput({3, 0}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsOnIncompatibleNegativeOneInference) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  // 6 elements / 4 != integer.
  ctx.Set("S", MakeShapeInput({-1, 4}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Concat");
  node.add_input("X");
  node.add_input("S");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  ctx.Set("S", MakeShapeInput({2}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorReshape, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeReshapeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeReshape(ctx, node), std::out_of_range);
}

namespace {

NodeProto MakeCastNode(int64_t to, bool with_to_attr = true) {
  NodeProto node;
  node.set_op_type("Cast");
  node.add_input("X");
  node.add_output("Y");
  if (with_to_attr) {
    AddAttribute<int64_t>(node, "to", to);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorCast, PreservesShapeAndUsesToAttributeDtype) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeCast(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesTensorCast, PreservesSymbolicDimensions) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::INT64));
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::tensor::ComputeShapeCast(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorCast, ThrowsWhenToAttributeMissing) {
  NodeProto node = MakeCastNode(0, /*with_to_attr=*/false);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCast(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCast, ThrowsOnUnsupportedToValue) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::UNDEFINED));
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCast(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCast, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::FLOAT));
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCast(ctx, node), std::out_of_range);
}

TEST(OnnxOptimShapesTensorCast, RejectsWrongOpType) {
  NodeProto node = MakeCastNode(static_cast<int64_t>(TensorProto::DataType::FLOAT));
  node.set_op_type("NotCast");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCast(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// CastLike shape inference tests.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCastLikeNode(int input_count = 2) {
  NodeProto node;
  node.set_op_type("CastLike");
  if (input_count >= 1) {
    node.add_input("X");
  }
  if (input_count >= 2) {
    node.add_input("TT");
  }
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorCastLike, PreservesShapeAndUsesTargetTypeDtype) {
  NodeProto node = MakeCastLikeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("TT", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble,
                                        onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));

  onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesTensorCastLike, PreservesSymbolicDimensions) {
  NodeProto node = MakeCastLikeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));
  ctx.Set("TT", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                        onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));

  onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorCastLike, IgnoresTargetTypeShape) {
  // Output shape must come from input(0), not from input(1).
  NodeProto node = MakeCastLikeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));
  ctx.Set("TT", onnx_optim::OptimTensor(
                    nullptr, onnx_optim::TensorType::kBool,
                    onnx_optim::OptimShape{onnx_optim::OptimDim(99), onnx_optim::OptimDim(7)}));

  onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node);

  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kBool);
  EXPECT_EQ(ctx.Get("Y").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapesTensorCastLike, ThrowsOnUndefinedTargetTypeDtype) {
  NodeProto node = MakeCastLikeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  ctx.Set("TT", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kUndefined,
                                        onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCastLike, ThrowsWhenSecondInputMissing) {
  NodeProto node = MakeCastLikeNode(/*input_count=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorCastLike, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeCastLikeNode();
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node), std::out_of_range);
}

TEST(OnnxOptimShapesTensorCastLike, RejectsWrongOpType) {
  NodeProto node = MakeCastLikeNode();
  node.set_op_type("NotCastLike");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  ctx.Set("TT", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                        onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeCastLike(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// AffineGrid shape inference tests.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeAffineGridNode(int64_t align_corners = 0) {
  NodeProto node;
  node.set_op_type("AffineGrid");
  node.add_input("theta");
  node.add_input("size");
  node.add_output("grid");
  if (align_corners != 0) {
    AddAttribute<int64_t>(node, "align_corners", align_corners);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorAffineGrid, FullyStatic2DFromConstantSize) {
  NodeProto node = MakeAffineGridNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3)}));
  // size is a 1-D INT64 tensor of length 4 whose constant value is
  // (N=2, C=3, H=5, W=6).
  onnx_optim::OptimTensor size_tensor(nullptr, onnx_optim::TensorType::kInt64,
                                      onnx_optim::OptimShape{onnx_optim::OptimDim(4)});
  size_tensor.SetValueAsShape(
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                             onnx_optim::OptimDim(5), onnx_optim::OptimDim(6)});
  ctx.Set("size", std::move(size_tensor));

  onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  ASSERT_TRUE(ctx.Has("grid"));
  const onnx_optim::OptimShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 5);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 2);
  EXPECT_EQ(ctx.Get("grid").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesTensorAffineGrid, FullyStatic3DFromConstantSize) {
  NodeProto node = MakeAffineGridNode(/*align_corners=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3),
                                                                  onnx_optim::OptimDim(4)}));
  onnx_optim::OptimTensor size_tensor(nullptr, onnx_optim::TensorType::kInt64,
                                      onnx_optim::OptimShape{onnx_optim::OptimDim(5)});
  size_tensor.SetValueAsShape(onnx_optim::OptimShape{
      onnx_optim::OptimDim(2), onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
      onnx_optim::OptimDim(5), onnx_optim::OptimDim(6)});
  ctx.Set("size", std::move(size_tensor));

  onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  const onnx_optim::OptimShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 6);
  EXPECT_EQ(out[4].AsInt(), 3);
}

TEST(OnnxOptimShapesTensorAffineGrid, SymbolicSpatialDimsWhenSizeValueMissing) {
  // theta tells us it's a 2D AffineGrid, but ``size`` has no known value;
  // the output spatial dims should be left symbolic.
  NodeProto node = MakeAffineGridNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim("N"),
                                                                  onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3)}));
  ctx.Set("size", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  const onnx_optim::OptimShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_TRUE(out[1].IsExpr()); // H
  EXPECT_TRUE(out[2].IsExpr()); // W
  EXPECT_EQ(out[3].AsInt(), 2);
}

TEST(OnnxOptimShapesTensorAffineGrid, ModeInferredFromSizeLengthWhenThetaSymbolic) {
  // theta has rank 3 but its inner dims are symbolic; the 3D vs 2D mode is
  // resolved from the static size length (here 5 → 3D).
  NodeProto node = MakeAffineGridNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(7),
                                                                  onnx_optim::OptimDim("r"),
                                                                  onnx_optim::OptimDim("c")}));
  ctx.Set("size", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node);

  const onnx_optim::OptimShape &out = ctx.Get("grid").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[0].AsInt(), 7);
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_TRUE(out[3].IsExpr());
  EXPECT_EQ(out[4].AsInt(), 3);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsInvalidThetaInnerDims) {
  NodeProto node = MakeAffineGridNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(4),
                                                                  onnx_optim::OptimDim(5)}));
  ctx.Set("size", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsThetaSizeModeMismatch) {
  // theta is 2D ((N, 2, 3)) but size has length 5 (3D).
  NodeProto node = MakeAffineGridNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3)}));
  ctx.Set("size", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsBadSizeLength) {
  NodeProto node = MakeAffineGridNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3)}));
  ctx.Set("size", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorAffineGrid, RejectsWrongOpType) {
  NodeProto node = MakeAffineGridNode();
  node.set_op_type("GridSample");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("theta", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                           onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                  onnx_optim::OptimDim(2),
                                                                  onnx_optim::OptimDim(3)}));
  ctx.Set("size", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeAffineGrid(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Expand shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeGridSampleNode() {
  NodeProto node;
  node.set_op_type("GridSample");
  node.add_input("X");
  node.add_input("grid");
  node.add_output("Y");
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorGridSample, Static2DPropagatesNCAndGridSpatial) {
  // X: (N=2, C=3, H=4, W=5), grid: (N=2, H_out=6, W_out=7, 2).
  // Expected Y shape: (2, 3, 6, 7).
  NodeProto node = MakeGridSampleNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)}));
  ctx.Set("grid", onnx_optim::OptimTensor(
                      nullptr, onnx_optim::TensorType::kFloat,
                      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(6),
                                             onnx_optim::OptimDim(7), onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::tensor::ComputeShapeGridSample(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 7);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesTensorGridSample, Static3DVolumetric) {
  // X: (1, 2, 3, 4, 5), grid: (1, 6, 7, 8, 3) → Y: (1, 2, 6, 7, 8).
  NodeProto node = MakeGridSampleNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kDouble,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(2),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                          onnx_optim::OptimDim(5)}));
  ctx.Set("grid", onnx_optim::OptimTensor(
                      nullptr, onnx_optim::TensorType::kDouble,
                      onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(6),
                                             onnx_optim::OptimDim(7), onnx_optim::OptimDim(8),
                                             onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeGridSample(ctx, node);

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 5u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 2);
  EXPECT_EQ(out[2].AsInt(), 6);
  EXPECT_EQ(out[3].AsInt(), 7);
  EXPECT_EQ(out[4].AsInt(), 8);
}

TEST(OnnxOptimShapesTensorGridSample, SymbolicBatchIsMerged) {
  // X has symbolic N, grid has concrete N=4 → output N=4.
  NodeProto node = MakeGridSampleNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3),
                                          onnx_optim::OptimDim(8), onnx_optim::OptimDim(8)}));
  ctx.Set("grid", onnx_optim::OptimTensor(
                      nullptr, onnx_optim::TensorType::kFloat,
                      onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim("H"),
                                             onnx_optim::OptimDim("W"), onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::tensor::ComputeShapeGridSample(ctx, node);

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 4);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_TRUE(out[3].IsExpr());
}

TEST(OnnxOptimShapesTensorGridSample, RejectsRankMismatch) {
  NodeProto node = MakeGridSampleNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)}));
  ctx.Set("grid", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                                 onnx_optim::OptimDim(2),
                                                                 onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeGridSample(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTensorGridSample, RejectsBadGridTrailingDim) {
  NodeProto node = MakeGridSampleNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)}));
  // 2D GridSample: last dim must be 2, not 3.
  ctx.Set("grid", onnx_optim::OptimTensor(
                      nullptr, onnx_optim::TensorType::kFloat,
                      onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(2),
                                             onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeGridSample(ctx, node),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Expand shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeExpandNode(const std::string &input = "X", const std::string &shape = "S",
                         const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Expand");
  node.add_input(input);
  node.add_input(shape);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorExpand, BroadcastsKnownShapeInputToLargerShape) {
  // input: [3, 1] → expand to [2, 3, 6] → output shape [2, 3, 6]
  NodeProto node = MakeExpandNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(1)}));
  ctx.Set("S", MakeShapeInput({2, 3, 6}));

  onnx_optim::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                    onnx_optim::OptimDim(6)}));
}

TEST(OnnxOptimShapesTensorExpand, BroadcastsDimUnchangedCase) {
  // input: [3, 1] → expand to [3, 4] → output shape [3, 4]
  NodeProto node = MakeExpandNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(1)}));
  ctx.Set("S", MakeShapeInput({3, 4}));

  onnx_optim::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorExpand, PreservesInputDtype) {
  NodeProto node = MakeExpandNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4)}));
  ctx.Set("S", MakeShapeInput({2, 4}));

  onnx_optim::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorExpand, FallsBackToSymbolicWhenShapeUnknown) {
  // shape input has no value annotation; its static shape is [3] → output rank 3, symbolic dims.
  NodeProto node = MakeExpandNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("S", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeExpand(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorExpand, RejectsWrongOpType) {
  NodeProto node = MakeExpandNode();
  node.set_op_type("Reshape");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("S", MakeShapeInput({3}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeExpand(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tile shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTileNode(const std::string &input = "X", const std::string &repeats = "R",
                       const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Tile");
  node.add_input(input);
  node.add_input(repeats);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorTile, MultipliesEachDimByRepeats) {
  // input: [2, 3] tiled by [2, 4] -> output [4, 12]
  NodeProto node = MakeTileNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("R", MakeShapeInput({2, 4}));

  onnx_optim::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(12)}));
}

TEST(OnnxOptimShapesTensorTile, RepeatsOneLeavesDimsUnchanged) {
  NodeProto node = MakeTileNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(5)}));
  ctx.Set("R", MakeShapeInput({1, 1}));

  onnx_optim::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapesTensorTile, SymbolicInputDimYieldsSymbolicOutputDim) {
  NodeProto node = MakeTileNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3)}));
  ctx.Set("R", MakeShapeInput({2, 4}));

  onnx_optim::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_EQ(ctx.Get("Y").Shape()[1], onnx_optim::OptimDim(12));
}

TEST(OnnxOptimShapesTensorTile, FallsBackToSymbolicWhenRepeatsUnknown) {
  // repeats input has no value annotation; its static shape is [3] so the
  // output rank is 3 with every dim symbolic.
  NodeProto node = MakeTileNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  ctx.Set("R", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeTile(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_TRUE(ctx.Get("Y").Shape()[0].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[1].IsExpr());
  EXPECT_TRUE(ctx.Get("Y").Shape()[2].IsExpr());
}

TEST(OnnxOptimShapesTensorTile, RejectsRepeatsLengthMismatch) {
  NodeProto node = MakeTileNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("R", MakeShapeInput({2, 3, 4}));

  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeTile(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesTensorTile, RejectsWrongOpType) {
  NodeProto node = MakeTileNode();
  node.set_op_type("Expand");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("R", MakeShapeInput({2}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeTile(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Transpose shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTransposeNode(const std::string &input = "X", const std::string &out = "Y",
                            const std::vector<int64_t> &perm = {}) {
  NodeProto node;
  node.set_op_type("Transpose");
  node.add_input(input);
  node.add_output(out);
  if (!perm.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "perm", perm);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorTranspose, DefaultsToReversePermutation) {
  NodeProto node = MakeTransposeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::tensor::ComputeShapeTranspose(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3),
                                    onnx_optim::OptimDim(2)}));
}

TEST(OnnxOptimShapesTensorTranspose, AppliesExplicitPermutation) {
  NodeProto node = MakeTransposeNode("X", "Y", {1, 0, 2});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::tensor::ComputeShapeTranspose(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(2),
                                    onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesTensorTranspose, RejectsPermutationWithDuplicateAxis) {
  NodeProto node = MakeTransposeNode("X", "Y", {0, 0});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeTranspose(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Squeeze / Unsqueeze shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeSqueezeNode(const std::string &data = "X", const std::string &axes = "A",
                          const std::string &out = "Y", bool with_axes = true) {
  NodeProto node;
  node.set_op_type("Squeeze");
  node.add_input(data);
  if (with_axes) {
    node.add_input(axes);
  }
  node.add_output(out);
  return node;
}

NodeProto MakeUnsqueezeNode(const std::string &data = "X", const std::string &axes = "A",
                            const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("Unsqueeze");
  node.add_input(data);
  node.add_input(axes);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorSqueeze, RemovesExplicitAxes) {
  NodeProto node = MakeSqueezeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(1)}));
  ctx.Set("A", MakeShapeInput({1, 3}));

  onnx_optim::shapes::tensor::ComputeShapeSqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesTensorSqueeze, WithoutAxesRemovesConcreteUnitDims) {
  NodeProto node = MakeSqueezeNode("X", "A", "Y", /*with_axes=*/false);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt64,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(2),
                                          onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::tensor::ComputeShapeSqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesTensorUnsqueeze, InsertsDimsAtGivenAxes) {
  NodeProto node = MakeUnsqueezeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("A", MakeShapeInput({0, 2}));

  onnx_optim::shapes::tensor::ComputeShapeUnsqueeze(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(2),
                                    onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)}));
}

TEST(OnnxOptimShapesTensorUnsqueeze, RejectsDuplicateAxes) {
  NodeProto node = MakeUnsqueezeNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("A", MakeShapeInput({1, 1}));

  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeUnsqueeze(ctx, node), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// NonZero shape-inference tests
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeNonZeroNode(const std::string &input = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("NonZero");
  node.add_input(input);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTensorNonZero, ProducesRank2Int64WithRankAsFirstDim) {
  NodeProto node = MakeNonZeroNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::tensor::ComputeShapeNonZero(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const auto &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), onnx_optim::TensorType::kInt64);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  ASSERT_TRUE(y.Shape()[0].IsInt());
  EXPECT_EQ(y.Shape()[0].AsInt(), 3);
  // The number of non-zero elements is a runtime value: must be symbolic.
  EXPECT_FALSE(y.Shape()[1].IsInt());
}

TEST(OnnxOptimShapesTensorNonZero, ScalarInputProducesShapeZeroByNnz) {
  NodeProto node = MakeNonZeroNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kBool,
                                       onnx_optim::OptimShape{}));

  onnx_optim::shapes::tensor::ComputeShapeNonZero(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const auto &y = ctx.Get("Y");
  EXPECT_EQ(y.Dtype(), onnx_optim::TensorType::kInt64);
  ASSERT_EQ(y.Shape().Rank(), 2u);
  ASSERT_TRUE(y.Shape()[0].IsInt());
  EXPECT_EQ(y.Shape()[0].AsInt(), 0);
  EXPECT_FALSE(y.Shape()[1].IsInt());
}

TEST(OnnxOptimShapesTensorNonZero, RejectsWrongOpType) {
  NodeProto node = MakeNonZeroNode();
  node.set_op_type("Abs");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));
  EXPECT_THROW(onnx_optim::shapes::tensor::ComputeShapeNonZero(ctx, node), std::invalid_argument);
}

} // namespace Test
