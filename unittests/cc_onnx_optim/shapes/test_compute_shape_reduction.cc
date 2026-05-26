// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/reduction/shape_reduction.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeReduceSumNode(const std::vector<std::string> &inputs,
                            const std::optional<int64_t> &keepdims = std::nullopt,
                            const std::optional<int64_t> &noop_with_empty_axes = std::nullopt,
                            const std::vector<int64_t> &axes_attr = {}) {
  NodeProto node;
  node.set_op_type("ReduceSum");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output("Y");
  if (keepdims.has_value()) {
    AddAttribute<int64_t>(node, "keepdims", *keepdims);
  }
  if (noop_with_empty_axes.has_value()) {
    AddAttribute<int64_t>(node, "noop_with_empty_axes", *noop_with_empty_axes);
  }
  if (!axes_attr.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "axes", axes_attr);
  }
  return node;
}

void SetData(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &shape,
             onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, dtype, shape));
}

void SetAxesValue(onnx_optim::shapes::ShapesContext &ctx, const std::vector<int64_t> &axes_values) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(static_cast<int64_t>(axes_values.size()))};
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kInt64, shape);
  onnx_optim::OptimShape vshape;
  for (int64_t v : axes_values) {
    vshape.PushBack(onnx_optim::OptimDim(v));
  }
  t.SetValueAsShape(std::move(vshape));
  ctx.Set("axes", std::move(t));
}

} // namespace

// ── opset >= 13 (axes is an input) ─────────────────────────────────────────

TEST(OnnxOptimShapesReductionReduceSum, DefaultReducesAllKeepdims) {
  // No axes input, no keepdims attribute (defaults to 1), no
  // noop_with_empty_axes -> reduce every dim and keep them as 1.
  NodeProto node = MakeReduceSumNode({"X"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceSum, DefaultReducesAllNoKeepdimsScalar) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapesReductionReduceSum, NoopWithEmptyAxesIsIdentity) {
  // noop_with_empty_axes=1 with no axes input -> identity, output == input.
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/std::nullopt, /*noop_with_empty_axes=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  const onnx_optim::OptimShape in{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  SetData(ctx, in);

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Shape(), in);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});
  SetAxesValue(ctx, {1});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputValueAsShapeNoKeepdims) {
  NodeProto node = MakeReduceSumNode({"X", "axes"}, /*keepdims=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});
  SetAxesValue(ctx, {0, 2});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputNegativeAxis) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});
  SetAxesValue(ctx, {-1});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputEmptyValueAsShapeReducesAll) {
  // axes input present but with value-as-shape of rank 0 -> "empty axes".
  // With noop_with_empty_axes=0 (default) it reduces all dims.
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{});
  t.SetValueAsShape(onnx_optim::OptimShape{});
  ctx.Set("axes", std::move(t));

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputEmptyValueAsShapeNoop) {
  NodeProto node =
      MakeReduceSumNode({"X", "axes"}, /*keepdims=*/std::nullopt, /*noop_with_empty_axes=*/1);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  const onnx_optim::OptimShape in{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  SetData(ctx, in);
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kInt64, onnx_optim::OptimShape{});
  t.SetValueAsShape(onnx_optim::OptimShape{});
  ctx.Set("axes", std::move(t));

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  EXPECT_EQ(ctx.Get("Y").Shape(), in);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputUnknownValuesKeepdimsKeepsRank) {
  // axes input present, no ValueAsShape; the rank-1 shape says how many axes
  // will be reduced but not which ones. With keepdims=1 the rank is
  // preserved; the dims become symbolic.
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});
  ctx.Set("axes", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_TRUE(out[2].IsExpr());
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputUnknownValuesNoKeepdimsReducesRank) {
  // axes input present, rank-1 shape of size 2 known but content unknown,
  // keepdims=0: output rank = 3 - 2 = 1 with symbolic dim.
  NodeProto node = MakeReduceSumNode({"X", "axes"}, /*keepdims=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});
  ctx.Set("axes", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(2)}));

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_TRUE(out[0].IsExpr());
}

TEST(OnnxOptimShapesReductionReduceSum, RejectsAxisOutOfRange) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  SetAxesValue(ctx, {5});

  EXPECT_THROW(onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceSum, RejectsTooManyAxesNoKeepdims) {
  // 4 axes but rank-3 input -> would yield negative rank.
  NodeProto node = MakeReduceSumNode({"X", "axes"}, /*keepdims=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});
  ctx.Set("axes", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));

  EXPECT_THROW(onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceSum, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  node.set_op_type("ReduceMean");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  EXPECT_THROW(onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceSum, PreservesInputDtype) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)},
          onnx_optim::TensorType::kDouble);

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kDouble);
}

TEST(OnnxOptimShapesReductionReduceSum, PropagatesSymbolicDimsThatAreNotReduced) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim("D")});
  SetAxesValue(ctx, {1});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_TRUE(out[2].IsExpr());
  EXPECT_EQ(out[2].AsExpr(), "D");
}

// ── opset < 13 (axes is an attribute) ──────────────────────────────────────

TEST(OnnxOptimShapesReductionReduceSum, AttributeAxesOpset11) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/1, /*noop_with_empty_axes=*/std::nullopt,
                                     /*axes_attr=*/{1});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapesReductionReduceSum, AttributeAxesOpset11NoKeepdims) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0, /*noop_with_empty_axes=*/std::nullopt,
                                     /*axes_attr=*/{0, 2});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                      onnx_optim::OptimDim(4)});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceSum, AttributeAxesMissingReducesAll) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});

  onnx_optim::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

} // namespace Test
