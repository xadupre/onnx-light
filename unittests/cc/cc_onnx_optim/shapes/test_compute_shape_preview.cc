// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/preview/shape_preview.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeFlexAttentionNode() {
  NodeProto node;
  node.set_op_type("FlexAttention");
  node.set_domain("ai.onnx.preview");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_output("Y");
  return node;
}

void SetFlexInputs(onnx_optim::shapes::ShapesContext &ctx, const onnx_optim::OptimShape &q,
                   const onnx_optim::OptimShape &k, const onnx_optim::OptimShape &v,
                   onnx_optim::TensorType dtype = onnx_optim::TensorType::kFloat) {
  ctx.Set("Q", onnx_optim::OptimTensor(nullptr, dtype, q));
  ctx.Set("K", onnx_optim::OptimTensor(nullptr, dtype, k));
  ctx.Set("V", onnx_optim::OptimTensor(nullptr, dtype, v));
}

onnx_optim::OptimShape Shape4(int64_t a, int64_t b, int64_t c, int64_t d) {
  return onnx_optim::OptimShape{onnx_optim::OptimDim(a), onnx_optim::OptimDim(b),
                                onnx_optim::OptimDim(c), onnx_optim::OptimDim(d)};
}

} // namespace

TEST(OnnxOptimShapesPreviewFlexAttention, BasicShape) {
  // Same number of Q and KV heads, output is (B, Hq, L, Dv).
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, /*q=*/Shape4(2, 4, 8, 16),
                /*k=*/Shape4(2, 4, 12, 16),
                /*v=*/Shape4(2, 4, 12, 32));

  onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V");

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 8);
  EXPECT_EQ(out[3].AsInt(), 32);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
}

TEST(OnnxOptimShapesPreviewFlexAttention, GroupedQueryAttention) {
  // q_num_heads=8 is a multiple of kv_num_heads=2. Output[1] follows Q.
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 8, 5, 6), Shape4(1, 2, 7, 6), Shape4(1, 2, 7, 6));

  onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 1);
  EXPECT_EQ(out[1].AsInt(), 8);
  EXPECT_EQ(out[2].AsInt(), 5);
  EXPECT_EQ(out[3].AsInt(), 6);
}

TEST(OnnxOptimShapesPreviewFlexAttention, PropagatesSymbolicBatch) {
  // When the batch dim is symbolic in Q but static in K, the static
  // value wins so downstream ops see a concrete batch.
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape q{onnx_optim::OptimDim("B"), onnx_optim::OptimDim(4),
                           onnx_optim::OptimDim(8), onnx_optim::OptimDim(16)};
  SetFlexInputs(ctx, q, Shape4(2, 4, 12, 16), Shape4(2, 4, 12, 32));

  onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V");

  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_TRUE(out[0].IsInt());
  EXPECT_EQ(out[0].AsInt(), 2);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedBatch) {
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(2, 4, 8, 16), Shape4(3, 4, 12, 16), Shape4(3, 4, 12, 32));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedKvSeqLen) {
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 2, 3, 4), Shape4(1, 2, 5, 4), Shape4(1, 2, 7, 4));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedQkHeadSize) {
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 2, 3, 8), Shape4(1, 2, 3, 16), Shape4(1, 2, 3, 16));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsInvalidGqa) {
  // q_num_heads=7 is not a multiple of kv_num_heads=2.
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 7, 3, 4), Shape4(1, 2, 3, 4), Shape4(1, 2, 3, 4));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsWrongRank) {
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("Q", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                              onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3)}));
  ctx.Set("K", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                              onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3)}));
  ctx.Set("V", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1),
                                                              onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedDtype) {
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("Q",
          onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, Shape4(2, 4, 8, 16)));
  ctx.Set("K",
          onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat16, Shape4(2, 4, 12, 16)));
  ctx.Set("V",
          onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, Shape4(2, 4, 12, 32)));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(2, 4, 8, 16), Shape4(2, 4, 12, 16), Shape4(2, 4, 12, 32));
  EXPECT_THROW(onnx_optim::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, DispatchesThroughComputeShapeNode) {
  // End-to-end check: the dispatch in ComputeShapeNode picks up the
  // ai.onnx.preview:FlexAttention entry.
  NodeProto node = MakeFlexAttentionNode();
  onnx_optim::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(2, 4, 8, 16), Shape4(2, 4, 12, 16), Shape4(2, 4, 12, 32));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  const onnx_optim::OptimShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 8);
  EXPECT_EQ(out[3].AsInt(), 32);
}

} // namespace Test
