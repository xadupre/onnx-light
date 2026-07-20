// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/onnx_shapes/shapes/preview/shape_preview.h"
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

void SetFlexInputs(core::shapes::ShapesContext &ctx, const core::symbolic::SymShape &q,
                   const core::symbolic::SymShape &k, const core::symbolic::SymShape &v,
                   core::symbolic::TensorType dtype = core::symbolic::TensorType::kFloat) {
  ctx.Set("Q", core::symbolic::SymTensor(nullptr, dtype, q));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, dtype, k));
  ctx.Set("V", core::symbolic::SymTensor(nullptr, dtype, v));
}

core::symbolic::SymShape Shape4(int64_t a, int64_t b, int64_t c, int64_t d) {
  return core::symbolic::SymShape{core::symbolic::SymDim(a), core::symbolic::SymDim(b),
                                  core::symbolic::SymDim(c), core::symbolic::SymDim(d)};
}

} // namespace

TEST(OnnxOptimShapesPreviewFlexAttention, BasicShape) {
  // Same number of Q and KV heads, output is (B, Hq, L, Dv).
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, /*q=*/Shape4(2, 4, 8, 16),
                /*k=*/Shape4(2, 4, 12, 16),
                /*v=*/Shape4(2, 4, 12, 32));

  onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V");

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 8);
  EXPECT_EQ(out[3].AsInt(), 32);
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kFloat);
}

TEST(OnnxOptimShapesPreviewFlexAttention, GroupedQueryAttention) {
  // q_num_heads=8 is a multiple of kv_num_heads=2. Output[1] follows Q.
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 8, 5, 6), Shape4(1, 2, 7, 6), Shape4(1, 2, 7, 6));

  onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
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
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape q{core::symbolic::SymDim("B"), core::symbolic::SymDim(4),
                             core::symbolic::SymDim(8), core::symbolic::SymDim(16)};
  SetFlexInputs(ctx, q, Shape4(2, 4, 12, 16), Shape4(2, 4, 12, 32));

  onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V");

  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_TRUE(out[0].IsInt());
  EXPECT_EQ(out[0].AsInt(), 2);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedBatch) {
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(2, 4, 8, 16), Shape4(3, 4, 12, 16), Shape4(3, 4, 12, 32));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedKvSeqLen) {
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 2, 3, 4), Shape4(1, 2, 5, 4), Shape4(1, 2, 7, 4));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedQkHeadSize) {
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 2, 3, 8), Shape4(1, 2, 3, 16), Shape4(1, 2, 3, 16));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsInvalidGqa) {
  // q_num_heads=7 is not a multiple of kv_num_heads=2.
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(1, 7, 3, 4), Shape4(1, 2, 3, 4), Shape4(1, 2, 3, 4));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsWrongRank) {
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("Q", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3)}));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3)}));
  ctx.Set("V", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim(2),
                                                                  core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsMismatchedDtype) {
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("Q", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         Shape4(2, 4, 8, 16)));
  ctx.Set("K", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat16,
                                         Shape4(2, 4, 12, 16)));
  ctx.Set("V", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         Shape4(2, 4, 12, 32)));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Attention");
  node.add_input("Q");
  node.add_input("K");
  node.add_input("V");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(2, 4, 8, 16), Shape4(2, 4, 12, 16), Shape4(2, 4, 12, 32));
  EXPECT_THROW(onnx_shapes::shapes::preview::ComputeShapeFlexAttention(ctx, node, "Q", "K", "V"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesPreviewFlexAttention, DispatchesThroughComputeShapeNode) {
  // End-to-end check: the dispatch in ComputeShapeNode picks up the
  // ai.onnx.preview:FlexAttention entry.
  NodeProto node = MakeFlexAttentionNode();
  core::shapes::ShapesContext ctx;
  SetFlexInputs(ctx, Shape4(2, 4, 8, 16), Shape4(2, 4, 12, 16), Shape4(2, 4, 12, 32));

  ctx.ComputeShapeNode(node);

  ASSERT_TRUE(ctx.Has("Y"));
  const core::symbolic::SymShape &out = ctx.Get("Y").Shape();
  ASSERT_EQ(out.Rank(), 4u);
  EXPECT_EQ(out[0].AsInt(), 2);
  EXPECT_EQ(out[1].AsInt(), 4);
  EXPECT_EQ(out[2].AsInt(), 8);
  EXPECT_EQ(out[3].AsInt(), 32);
}

} // namespace Test
