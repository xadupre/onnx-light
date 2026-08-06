// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/preview/shape_preview.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::preview {

namespace {

// Merges two dimensions: if both are static they must agree, otherwise
// the static one (if any) wins; if both are symbolic the first one is
// kept.
SymDim MergeDim(const SymDim &a, const SymDim &b, const char *what) {
  if (a.IsInt() && b.IsInt()) {
    EXT_ENFORCE_INVALID(a.AsInt() == b.AsInt(), "ComputeShapeFlexAttention: ", what,
                        " mismatch: ", a.AsInt(), " vs ", b.AsInt(), ".");
    return a;
  }
  if (a.IsInt()) {
    return a;
  }
  if (b.IsInt()) {
    return b;
  }
  return a;
}

void RequireRank4(const SymShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 4, "ComputeShapeFlexAttention: input '", name,
                      "' must have rank 4, got ", shape.Rank(), ".");
}

} // namespace

void ComputeShapeFlexAttention(ShapesContext &ctx, const NodeProto &node, const char *q,
                               const char *k, const char *v) {
  CheckNodeOpAndOutput(node, "FlexAttention", "ComputeShapeFlexAttention");

  const SymTensor &Q = ctx.Get(q);
  const SymTensor &K = ctx.Get(k);
  const SymTensor &V = ctx.Get(v);

  EXT_ENFORCE_INVALID(!(Q.Dtype() != K.Dtype() || Q.Dtype() != V.Dtype()),
                      "ComputeShapeFlexAttention: Q, K, and V must share the same element type.");

  const SymShape &q_shape = Q.Shape();
  const SymShape &k_shape = K.Shape();
  const SymShape &v_shape = V.Shape();
  RequireRank4(q_shape, q);
  RequireRank4(k_shape, k);
  RequireRank4(v_shape, v);

  // Batch: Q[0] == K[0] == V[0] (when static).
  SymDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");

  // K and V share the same number of heads.
  EXT_ENFORCE_INVALID(
      !(k_shape[1].IsInt() && v_shape[1].IsInt() && k_shape[1].AsInt() != v_shape[1].AsInt()),
      "ComputeShapeFlexAttention: key and value must share the same head dimension (got ",
      k_shape[1].AsInt(), " vs ", v_shape[1].AsInt(), ").");

  // K and V share the same sequence length.
  EXT_ENFORCE_INVALID(
      !(k_shape[2].IsInt() && v_shape[2].IsInt() && k_shape[2].AsInt() != v_shape[2].AsInt()),
      "ComputeShapeFlexAttention: key and value must share the same sequence length (got ",
      k_shape[2].AsInt(), " vs ", v_shape[2].AsInt(), ").");

  // Q and K share the same embedding dimension.
  EXT_ENFORCE_INVALID(
      !(q_shape[3].IsInt() && k_shape[3].IsInt() && q_shape[3].AsInt() != k_shape[3].AsInt()),
      "ComputeShapeFlexAttention: query and key must share the same embedding dimension (got ",
      q_shape[3].AsInt(), " vs ", k_shape[3].AsInt(), ").");

  // Grouped Query Attention: q_num_heads must be a multiple of kv_num_heads.
  if (q_shape[1].IsInt() && k_shape[1].IsInt()) {
    const int64_t hq = q_shape[1].AsInt();
    const int64_t hkv = k_shape[1].AsInt();
    EXT_ENFORCE_INVALID(!(hq != hkv && (hkv <= 0 || (hq % hkv) != 0)),
                        "ComputeShapeFlexAttention: q_num_heads (", hq,
                        ") must be a multiple of kv_num_heads (", hkv, ") when they differ.");
  }

  // Output: (batch, q_num_heads, q_seq_len, v_head_size).
  SymShape out_shape;
  out_shape.PushBack(batch);
  out_shape.PushBack(q_shape[1]);
  out_shape.PushBack(q_shape[2]);
  out_shape.PushBack(v_shape[3]);

  ctx.Set(node.output(0), SymTensor(nullptr, Q.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::preview
