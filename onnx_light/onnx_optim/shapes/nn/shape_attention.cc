// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

// Merges two dimensions: if both are static they must agree, otherwise
// the static one (if any) wins; if both are symbolic the first one is
// kept.
OptimDim MergeDim(const OptimDim &a, const OptimDim &b, const char *what) {
  if (a.IsInt() && b.IsInt()) {
    if (a.AsInt() != b.AsInt()) {
      throw std::invalid_argument(std::string("ComputeShapeAttention: ") + what +
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
    throw std::invalid_argument(std::string("ComputeShapeAttention: input '") + name +
                                "' must have rank 4, got " + std::to_string(shape.Rank()) + ".");
  }
}

// Returns ``a + b`` when both dims are static; otherwise returns ``a``.
OptimDim AddDims(const OptimDim &a, const OptimDim &b) {
  if (a.IsInt() && b.IsInt()) {
    return OptimDim(a.AsInt() + b.AsInt());
  }
  return a;
}

} // namespace

void ComputeShapeAttention(ShapesContext &ctx, const NodeProto &node, const char *q, const char *k,
                           const char *v, const char *past_key, const char *past_value) {
  CheckNodeOpAndOutput(node, "Attention", "ComputeShapeAttention");

  const OptimTensor &Q = ctx.Get(q);
  const OptimTensor &K = ctx.Get(k);
  const OptimTensor &V = ctx.Get(v);

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
  OptimDim kv_num_heads = MergeDim(k_shape[1], v_shape[1], "kv_num_heads");

  // K and V share the same sequence length.
  OptimDim kv_seq_len = MergeDim(k_shape[2], v_shape[2], "kv_sequence_length");

  // Q and K share the same head dimension.
  OptimDim head_size = MergeDim(q_shape[3], k_shape[3], "head_size");

  // Grouped Query Attention: q_num_heads must be a multiple of kv_num_heads.
  if (q_shape[1].IsInt() && kv_num_heads.IsInt()) {
    const int64_t hq = q_shape[1].AsInt();
    const int64_t hkv = kv_num_heads.AsInt();
    if (hq != hkv && (hkv <= 0 || (hq % hkv) != 0)) {
      throw std::invalid_argument("ComputeShapeAttention: q_num_heads (" + std::to_string(hq) +
                                  ") must be a multiple of kv_num_heads (" + std::to_string(hkv) +
                                  ") when they differ.");
    }
  }

  // total_sequence_length = past_sequence_length + kv_sequence_length when a
  // past key/value is provided; otherwise it equals kv_sequence_length.
  OptimDim total_seq_len = kv_seq_len;
  if (past_key != nullptr && ctx.Has(past_key)) {
    const OptimShape &past_k_shape = ctx.Get(past_key).Shape();
    RequireRank4(past_k_shape, past_key);
    batch = MergeDim(batch, past_k_shape[0], "batch");
    kv_num_heads = MergeDim(kv_num_heads, past_k_shape[1], "kv_num_heads");
    head_size = MergeDim(head_size, past_k_shape[3], "head_size");
    total_seq_len = AddDims(kv_seq_len, past_k_shape[2]);
  }
  if (past_value != nullptr && ctx.Has(past_value)) {
    const OptimShape &past_v_shape = ctx.Get(past_value).Shape();
    RequireRank4(past_v_shape, past_value);
    batch = MergeDim(batch, past_v_shape[0], "batch");
    kv_num_heads = MergeDim(kv_num_heads, past_v_shape[1], "kv_num_heads");
    if (past_key == nullptr || !ctx.Has(past_key)) {
      total_seq_len = AddDims(kv_seq_len, past_v_shape[2]);
    }
  }

  // Output 0: Y = (batch, q_num_heads, q_seq_len, v_head_size).
  {
    OptimShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(q_shape[1]);
    out_shape.PushBack(q_shape[2]);
    out_shape.PushBack(v_shape[3]);
    ctx.Set(node.output(0), OptimTensor(nullptr, Q.Dtype(), std::move(out_shape)));
  }

  // Output 1: present_key = (batch, kv_num_heads, total_seq_len, head_size).
  if (node.output_size() > 1 && !node.output(1).as_string().empty()) {
    OptimShape pk_shape;
    pk_shape.PushBack(batch);
    pk_shape.PushBack(kv_num_heads);
    pk_shape.PushBack(total_seq_len);
    pk_shape.PushBack(head_size);
    ctx.Set(node.output(1), OptimTensor(nullptr, Q.Dtype(), std::move(pk_shape)));
  }

  // Output 2: present_value = (batch, kv_num_heads, total_seq_len, v_head_size).
  if (node.output_size() > 2 && !node.output(2).as_string().empty()) {
    OptimShape pv_shape;
    pv_shape.PushBack(batch);
    pv_shape.PushBack(kv_num_heads);
    pv_shape.PushBack(total_seq_len);
    pv_shape.PushBack(v_shape[3]);
    ctx.Set(node.output(2), OptimTensor(nullptr, V.Dtype(), std::move(pv_shape)));
  }

  // Output 3: qk_matmul_output = (batch, q_num_heads, q_seq_len, total_seq_len).
  if (node.output_size() > 3 && !node.output(3).as_string().empty()) {
    OptimShape qk_shape;
    qk_shape.PushBack(batch);
    qk_shape.PushBack(q_shape[1]);
    qk_shape.PushBack(q_shape[2]);
    qk_shape.PushBack(total_seq_len);
    ctx.Set(node.output(3), OptimTensor(nullptr, Q.Dtype(), std::move(qk_shape)));
  }
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
