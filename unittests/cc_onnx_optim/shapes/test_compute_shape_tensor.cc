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

} // namespace Test
