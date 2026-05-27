// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/preview/shape_preview.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace preview {

namespace {

// Merges two dimensions: if both are static they must agree, otherwise
// the static one (if any) wins; if both are symbolic the first one is
// kept.
OptimDim MergeDim(const OptimDim &a, const OptimDim &b, const char *what) {
  if (a.IsInt() && b.IsInt()) {
    if (a.AsInt() != b.AsInt()) {
      throw std::invalid_argument(std::string("ComputeShapeFlexAttention: ") + what +
                                  " mismatch: " + std::to_string(a.AsInt()) + " vs " +
                                  std::to_string(b.AsInt()) + ".");
    }
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

void RequireRank4(const OptimShape &shape, const char *name) {
  if (shape.Rank() != 4) {
    throw std::invalid_argument(std::string("ComputeShapeFlexAttention: input '") + name +
                                "' must have rank 4, got " + std::to_string(shape.Rank()) + ".");
  }
}

} // namespace

void ComputeShapeFlexAttention(ShapesContext &ctx, const NodeProto &node, const char *q,
                               const char *k, const char *v) {
  CheckNodeOpAndOutput(node, "FlexAttention", "ComputeShapeFlexAttention");

  const OptimTensor &Q = ctx.Get(q);
  const OptimTensor &K = ctx.Get(k);
  const OptimTensor &V = ctx.Get(v);

  if (Q.Dtype() != K.Dtype() || Q.Dtype() != V.Dtype()) {
    throw std::invalid_argument(
        "ComputeShapeFlexAttention: Q, K, and V must share the same element type.");
  }

  const OptimShape &q_shape = Q.Shape();
  const OptimShape &k_shape = K.Shape();
  const OptimShape &v_shape = V.Shape();
  RequireRank4(q_shape, q);
  RequireRank4(k_shape, k);
  RequireRank4(v_shape, v);

  // Batch: Q[0] == K[0] == V[0] (when static).
  OptimDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");

  // K and V share the same number of heads.
  if (k_shape[1].IsInt() && v_shape[1].IsInt() && k_shape[1].AsInt() != v_shape[1].AsInt()) {
    throw std::invalid_argument(
        "ComputeShapeFlexAttention: key and value must share the same head dimension (got " +
        std::to_string(k_shape[1].AsInt()) + " vs " + std::to_string(v_shape[1].AsInt()) + ").");
  }

  // K and V share the same sequence length.
  if (k_shape[2].IsInt() && v_shape[2].IsInt() && k_shape[2].AsInt() != v_shape[2].AsInt()) {
    throw std::invalid_argument(
        "ComputeShapeFlexAttention: key and value must share the same sequence length (got " +
        std::to_string(k_shape[2].AsInt()) + " vs " + std::to_string(v_shape[2].AsInt()) + ").");
  }

  // Q and K share the same embedding dimension.
  if (q_shape[3].IsInt() && k_shape[3].IsInt() && q_shape[3].AsInt() != k_shape[3].AsInt()) {
    throw std::invalid_argument(
        "ComputeShapeFlexAttention: query and key must share the same embedding dimension (got " +
        std::to_string(q_shape[3].AsInt()) + " vs " + std::to_string(k_shape[3].AsInt()) + ").");
  }

  // Grouped Query Attention: q_num_heads must be a multiple of kv_num_heads.
  if (q_shape[1].IsInt() && k_shape[1].IsInt()) {
    const int64_t hq = q_shape[1].AsInt();
    const int64_t hkv = k_shape[1].AsInt();
    if (hq != hkv && (hkv <= 0 || (hq % hkv) != 0)) {
      throw std::invalid_argument("ComputeShapeFlexAttention: q_num_heads (" + std::to_string(hq) +
                                  ") must be a multiple of kv_num_heads (" + std::to_string(hkv) +
                                  ") when they differ.");
    }
  }

  // Output: (batch, q_num_heads, q_seq_len, v_head_size).
  OptimShape out_shape;
  out_shape.PushBack(batch);
  out_shape.PushBack(q_shape[1]);
  out_shape.PushBack(q_shape[2]);
  out_shape.PushBack(v_shape[3]);

  ctx.Set(node.output(0), OptimTensor(nullptr, Q.Dtype(), std::move(out_shape)));
}

} // namespace preview
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
