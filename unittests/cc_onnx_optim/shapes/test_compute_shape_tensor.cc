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

} // namespace Test
