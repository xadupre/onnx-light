// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/shapes/reduction/shape_reduction.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeReduceNode(const std::string &op_type, const std::vector<std::string> &inputs,
                         const std::optional<int64_t> &keepdims = std::nullopt,
                         const std::optional<int64_t> &noop_with_empty_axes = std::nullopt,
                         const std::vector<int64_t> &axes_attr = {}) {
  NodeProto node;
  node.set_op_type(op_type);
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

NodeProto MakeReduceSumNode(const std::vector<std::string> &inputs,
                            const std::optional<int64_t> &keepdims = std::nullopt,
                            const std::optional<int64_t> &noop_with_empty_axes = std::nullopt,
                            const std::vector<int64_t> &axes_attr = {}) {
  return MakeReduceNode("ReduceSum", inputs, keepdims, noop_with_empty_axes, axes_attr);
}

void SetData(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &shape,
             core::symbolic::TensorType dtype = core::symbolic::TensorType::kFloat) {
  ctx.Set("X", core::symbolic::SymTensor(nullptr, dtype, shape));
}

void SetAxesValue(core::shapes::ShapesContext &ctx, const std::vector<int64_t> &axes_values) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(static_cast<int64_t>(axes_values.size()))};
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt64, shape);
  core::symbolic::SymShape vshape;
  for (int64_t v : axes_values) {
    vshape.PushBack(core::symbolic::SymDim(v));
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
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceSum, DefaultReducesAllNoKeepdimsScalar) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapesReductionReduceSum, NoopWithEmptyAxesIsIdentity) {
  // noop_with_empty_axes=1 with no axes input -> identity, output == input.
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/std::nullopt, /*noop_with_empty_axes=*/1);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  const core::symbolic::SymShape in{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  SetData(ctx, in);

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Shape(), in);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputValueAsShapeNoKeepdims) {
  NodeProto node = MakeReduceSumNode({"X", "axes"}, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {0, 2});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputNegativeAxis) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {-1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputEmptyValueAsShapeReducesAll) {
  // axes input present but with value-as-shape of rank 0 -> "empty axes".
  // With noop_with_empty_axes=0 (default) it reduces all dims.
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  t.SetValueAsShape(core::symbolic::SymShape{});
  ctx.Set("axes", std::move(t));

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 1);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputEmptyValueAsShapeNoop) {
  NodeProto node =
      MakeReduceSumNode({"X", "axes"}, /*keepdims=*/std::nullopt, /*noop_with_empty_axes=*/1);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  const core::symbolic::SymShape in{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  SetData(ctx, in);
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt64,
                              core::symbolic::SymShape{});
  t.SetValueAsShape(core::symbolic::SymShape{});
  ctx.Set("axes", std::move(t));

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  EXPECT_EQ(ctx.Get("Y").Shape(), in);
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputUnknownValuesKeepdimsKeepsRank) {
  // axes input present, no ValueAsShape; the rank-1 shape says how many axes
  // will be reduced but not which ones. With keepdims=1 the rank is
  // preserved; the dims become symbolic.
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  ctx.Set("axes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_TRUE(out[2].IsExpr());
}

TEST(OnnxOptimShapesReductionReduceSum, AxesInputUnknownValuesNoKeepdimsReducesRank) {
  // axes input present, rank-1 shape of size 2 known but content unknown,
  // keepdims=0: output rank = 3 - 2 = 1 with symbolic dim.
  NodeProto node = MakeReduceSumNode({"X", "axes"}, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  ctx.Set("axes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(2)}));

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_TRUE(out[0].IsExpr());
}

TEST(OnnxOptimShapesReductionReduceSum, RejectsAxisOutOfRange) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  SetAxesValue(ctx, {5});

  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceSum, RejectsTooManyAxesNoKeepdims) {
  // 4 axes but rank-3 input -> would yield negative rank.
  NodeProto node = MakeReduceSumNode({"X", "axes"}, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  ctx.Set("axes", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64,
                                            core::symbolic::SymShape{core::symbolic::SymDim(4)}));

  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceSum, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  node.set_op_type("ReduceMean");
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceSum, PreservesInputDtype) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
          core::symbolic::TensorType::kDouble);

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
}

TEST(OnnxOptimShapesReductionReduceSum, PropagatesSymbolicDimsThatAreNotReduced) {
  NodeProto node = MakeReduceSumNode({"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim("D")});
  SetAxesValue(ctx, {1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
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
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
}

TEST(OnnxOptimShapesReductionReduceSum, AttributeAxesOpset11NoKeepdims) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0, /*noop_with_empty_axes=*/std::nullopt,
                                     /*axes_attr=*/{0, 2});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceSum, AttributeAxesMissingReducesAll) {
  NodeProto node = MakeReduceSumNode({"X"}, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSum(ctx, node, "X", nullptr);

  EXPECT_EQ(ctx.Get("Y").Shape().Rank(), 0u);
}

TEST(OnnxOptimShapesReductionReduceMax, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceNode("ReduceMax", {"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceMax(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceMin, AttributeAxesNoKeepdimsOpset17) {
  NodeProto node = MakeReduceNode("ReduceMin", {"X"}, /*keepdims=*/0,
                                  /*noop_with_empty_axes=*/std::nullopt,
                                  /*axes_attr=*/{0, 2});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 17);
  SetData(ctx,
          core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                   core::symbolic::SymDim(4)},
          core::symbolic::TensorType::kDouble);

  onnx_shapes::shapes::reduction::ComputeShapeReduceMin(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
}

TEST(OnnxOptimShapesReductionReduceMax, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceMax(ctx, node, "X", nullptr),
               std::invalid_argument);
}

// ── ReduceL1 / ReduceL2 shape inference ───────────────────────────────────

TEST(OnnxOptimShapesReductionReduceL1, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceNode("ReduceL1", {"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceL1(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
  EXPECT_EQ(out[2].AsInt(), 4);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceL2, AttributeAxesNoKeepdims) {
  NodeProto node = MakeReduceNode("ReduceL2", {"X"}, /*keepdims=*/0,
                                  /*noop_with_empty_axes=*/std::nullopt,
                                  /*axes_attr=*/{0, 2});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx,
          core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                   core::symbolic::SymDim(4)},
          core::symbolic::TensorType::kDouble);

  onnx_shapes::shapes::reduction::ComputeShapeReduceL2(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kDouble);
}

TEST(OnnxOptimShapesReductionReduceL1, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceL1(ctx, node, "X", nullptr),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionReduceL2, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceL2(ctx, node, "X", nullptr),
               std::invalid_argument);
}

// ── ReduceSumSquare shape inference ───────────────────────────────────────

TEST(OnnxOptimShapesReductionReduceSumSquare, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceNode("ReduceSumSquare", {"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {-1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSumSquare(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceSumSquare, AttributeAxesNoKeepdimsOpset11) {
  NodeProto node = MakeReduceNode("ReduceSumSquare", {"X"}, /*keepdims=*/0,
                                  /*noop_with_empty_axes=*/std::nullopt,
                                  /*axes_attr=*/{0, 2});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceSumSquare(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceSumSquare, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceSumSquare(ctx, node, "X", nullptr),
               std::invalid_argument);
}

// ── ReduceProd shape inference ────────────────────────────────────────────

TEST(OnnxOptimShapesReductionReduceProd, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceNode("ReduceProd", {"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {-1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceProd(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceProd, AttributeAxesNoKeepdimsOpset11) {
  NodeProto node = MakeReduceNode("ReduceProd", {"X"}, /*keepdims=*/0,
                                  /*noop_with_empty_axes=*/std::nullopt,
                                  /*axes_attr=*/{0, 2});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceProd(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceProd, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceProd(ctx, node, "X", nullptr),
               std::invalid_argument);
}

// ── ReduceMean shape inference ────────────────────────────────────────────

TEST(OnnxOptimShapesReductionReduceMean, AxesInputValueAsShapeKeepdims) {
  NodeProto node = MakeReduceNode("ReduceMean", {"X", "axes"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 18);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});
  SetAxesValue(ctx, {-1});

  onnx_shapes::shapes::reduction::ComputeShapeReduceMean(ctx, node, "X", "axes");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesReductionReduceMean, AttributeAxesNoKeepdimsOpset11) {
  NodeProto node = MakeReduceNode("ReduceMean", {"X"}, /*keepdims=*/0,
                                  /*noop_with_empty_axes=*/std::nullopt,
                                  /*axes_attr=*/{0, 2});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 11);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeReduceMean(ctx, node, "X", nullptr);

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 1u);
  EXPECT_EQ(out[0].AsInt(), 3);
}

TEST(OnnxOptimShapesReductionReduceMean, RejectsWrongOpType) {
  NodeProto node = MakeReduceSumNode({"X"});
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeReduceMean(ctx, node, "X", nullptr),
               std::invalid_argument);
}

// ── ArgMax / ArgMin shape inference ────────────────────────────────────────

namespace {

NodeProto MakeArgReduceNode(const std::string &op_type,
                            const std::optional<int64_t> &axis = std::nullopt,
                            const std::optional<int64_t> &keepdims = std::nullopt,
                            const std::optional<int64_t> &select_last_index = std::nullopt) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("X");
  node.add_output("Y");
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  if (keepdims.has_value()) {
    AddAttribute<int64_t>(node, "keepdims", *keepdims);
  }
  if (select_last_index.has_value()) {
    AddAttribute<int64_t>(node, "select_last_index", *select_last_index);
  }
  return node;
}

} // namespace

TEST(OnnxOptimShapesReductionArgReduce, DefaultAxisKeepdimsArgMax) {
  // axis defaults to 0, keepdims defaults to 1.
  NodeProto node = MakeArgReduceNode("ArgMax");
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X");

  const core::symbolic::SymTensor &out = ctx.Get("Y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(out.Shape().Rank(), 3u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 1);
  EXPECT_EQ(out.Shape()[1].AsInt(), 3);
  EXPECT_EQ(out.Shape()[2].AsInt(), 4);
}

TEST(OnnxOptimShapesReductionArgReduce, ExplicitAxisNoKeepdimsDropsDim) {
  NodeProto node = MakeArgReduceNode("ArgMin", /*axis=*/1, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X");

  const core::symbolic::SymTensor &out = ctx.Get("Y");
  EXPECT_EQ(out.Dtype(), core::symbolic::TensorType::kInt64);
  ASSERT_EQ(out.Shape().Rank(), 2u);
  EXPECT_EQ(out.Shape()[0].AsInt(), 2);
  EXPECT_EQ(out.Shape()[1].AsInt(), 4);
}

TEST(OnnxOptimShapesReductionArgReduce, NegativeAxisKeepdims) {
  NodeProto node = MakeArgReduceNode("ArgMax", /*axis=*/-1, /*keepdims=*/1);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim(4)});

  onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 3u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 3);
  EXPECT_EQ(out[2].AsInt(), 1);
}

TEST(OnnxOptimShapesReductionArgReduce, OutputIsInt64IndependentOfInputDtype) {
  NodeProto node = MakeArgReduceNode("ArgMin", /*axis=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)},
          core::symbolic::TensorType::kDouble);

  onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X");

  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kInt64);
}

TEST(OnnxOptimShapesReductionArgReduce, PreservesSymbolicNonReducedDims) {
  NodeProto node = MakeArgReduceNode("ArgMax", /*axis=*/1, /*keepdims=*/0);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(3),
                                        core::symbolic::SymDim("D")});

  onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "N");
  EXPECT_TRUE(out[1].IsExpr());
  EXPECT_EQ(out[1].AsExpr(), "D");
}

TEST(OnnxOptimShapesReductionArgReduce, IgnoresSelectLastIndexAttribute) {
  NodeProto node = MakeArgReduceNode("ArgMax", /*axis=*/1, /*keepdims=*/1,
                                     /*select_last_index=*/1);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 2u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 1);
}

TEST(OnnxOptimShapesReductionArgReduce, RejectsAxisOutOfRange) {
  NodeProto node = MakeArgReduceNode("ArgMax", /*axis=*/5);
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});

  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionArgReduce, RejectsScalarInput) {
  NodeProto node = MakeArgReduceNode("ArgMax");
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{});

  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesReductionArgReduce, RejectsWrongOpType) {
  NodeProto node = MakeArgReduceNode("ReduceSum");
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 13);
  SetData(ctx, core::symbolic::SymShape{core::symbolic::SymDim(2)});

  EXPECT_THROW(onnx_shapes::shapes::reduction::ComputeShapeArgReduce(ctx, node, "X"),
               std::invalid_argument);
}

} // namespace Test
