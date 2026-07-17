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
    EXT_ENFORCE_INVALID(a.AsInt() == b.AsInt(), "ComputeShapeAttention: ", what,
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

void RequireRank4(const OptimShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 4, "ComputeShapeAttention: input '", name,
                      "' must have rank 4, got ", shape.Rank(), ".");
}

// Returns ``a + b`` when both dims are static; otherwise returns ``a``.
OptimDim AddDims(const OptimDim &a, const OptimDim &b) {
  if (a.IsInt() && b.IsInt()) {
    return OptimDim(a.AsInt() + b.AsInt());
  }
  return a;
}

// Divides a (possibly symbolic) hidden dimension by ``num_heads``. When the
// dimension is static it must be a positive multiple of ``num_heads``;
// otherwise a fresh symbolic placeholder is returned.
OptimDim DivideByHeads(const OptimDim &hidden, int64_t num_heads, const char *what) {
  if (!hidden.IsInt()) {
    return OptimDim(std::string("?"));
  }
  const int64_t value = hidden.AsInt();
  EXT_ENFORCE_INVALID((value % num_heads) == 0, "ComputeShapeAttention: ", what, " (", value,
                      ") must be divisible by num_heads (", num_heads, ").");
  return OptimDim(value / num_heads);
}

// Computes ``num_heads * head`` when ``head`` is static; otherwise returns a
// fresh symbolic placeholder.
OptimDim MultiplyByHeads(int64_t num_heads, const OptimDim &head) {
  if (head.IsInt()) {
    return OptimDim(num_heads * head.AsInt());
  }
  return OptimDim(std::string("?"));
}

// Reads the optional ``q_num_heads`` / ``kv_num_heads`` attributes, which are
// required when the Q/K/V inputs are provided as rank-3 (packed) tensors.
void ReadHeadAttributes(const NodeProto &node, int64_t &q_num_heads, bool &has_q,
                        int64_t &kv_num_heads, bool &has_kv) {
  q_num_heads = 0;
  kv_num_heads = 0;
  has_q = false;
  has_kv = false;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "q_num_heads") {
      q_num_heads = attr.i();
      has_q = true;
    } else if (attr.name() == "kv_num_heads") {
      kv_num_heads = attr.i();
      has_kv = true;
    }
  }
}

// Handles the rank-3 (packed) form of Attention, where Q, K and V have shapes
// ``(batch, q_sequence_length, q_num_heads * head_size)`` and
// ``(batch, kv_sequence_length, kv_num_heads * head_size)``. The
// ``q_num_heads`` and ``kv_num_heads`` attributes are required to unpack the
// last dimension. Output 0 stays rank-3 while the optional present/qk outputs
// are rank-4.
void ComputeShapeAttentionRank3(ShapesContext &ctx, const NodeProto &node, const OptimTensor &Q,
                                const OptimTensor &K, const OptimTensor &V, const char *past_key,
                                const char *past_value) {
  int64_t q_num_heads = 0;
  int64_t kv_num_heads = 0;
  bool has_q = false;
  bool has_kv = false;
  ReadHeadAttributes(node, q_num_heads, has_q, kv_num_heads, has_kv);
  EXT_ENFORCE_INVALID(!(!has_q || !has_kv), "ComputeShapeAttention: attributes 'q_num_heads' and "
                                            "'kv_num_heads' are required for rank-3 inputs.");
  EXT_ENFORCE_INVALID(!(q_num_heads <= 0 || kv_num_heads <= 0),
                      "ComputeShapeAttention: 'q_num_heads' and 'kv_num_heads' must be positive.");
  EXT_ENFORCE_INVALID((q_num_heads % kv_num_heads) == 0, "ComputeShapeAttention: q_num_heads (",
                      q_num_heads, ") must be a multiple of kv_num_heads (", kv_num_heads, ").");

  const OptimShape &q_shape = Q.Shape();
  const OptimShape &k_shape = K.Shape();
  const OptimShape &v_shape = V.Shape();

  // Batch: Q[0] == K[0] == V[0]; q/kv sequence lengths from axis 1.
  OptimDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");
  OptimDim q_seq_len = q_shape[1];
  OptimDim kv_seq_len = MergeDim(k_shape[1], v_shape[1], "kv_sequence_length");

  // head_size = Q[2] / q_num_heads == K[2] / kv_num_heads.
  OptimDim head_size = DivideByHeads(q_shape[2], q_num_heads, "query hidden size");
  head_size =
      MergeDim(head_size, DivideByHeads(k_shape[2], kv_num_heads, "key hidden size"), "head_size");
  // v_head_size = V[2] / kv_num_heads.
  OptimDim v_head_size = DivideByHeads(v_shape[2], kv_num_heads, "value hidden size");

  // total_sequence_length = past_sequence_length + kv_sequence_length.
  OptimDim total_seq_len = kv_seq_len;
  if (past_key != nullptr && ctx.Has(past_key)) {
    const OptimShape &past_k_shape = ctx.Get(past_key).Shape();
    RequireRank4(past_k_shape, past_key);
    batch = MergeDim(batch, past_k_shape[0], "batch");
    head_size = MergeDim(head_size, past_k_shape[3], "head_size");
    total_seq_len = AddDims(kv_seq_len, past_k_shape[2]);
  }
  if (past_value != nullptr && ctx.Has(past_value)) {
    const OptimShape &past_v_shape = ctx.Get(past_value).Shape();
    RequireRank4(past_v_shape, past_value);
    batch = MergeDim(batch, past_v_shape[0], "batch");
    if (past_key == nullptr || !ctx.Has(past_key)) {
      total_seq_len = AddDims(kv_seq_len, past_v_shape[2]);
    }
  }

  // Output 0: Y = (batch, q_sequence_length, q_num_heads * v_head_size).
  {
    OptimShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(q_seq_len);
    out_shape.PushBack(MultiplyByHeads(q_num_heads, v_head_size));
    ctx.Set(node.output(0), OptimTensor(nullptr, Q.Dtype(), std::move(out_shape)));
  }

  // Output 1: present_key = (batch, kv_num_heads, total_seq_len, head_size).
  if (node.output_size() > 1 && !std::string(node.output(1)).empty()) {
    OptimShape pk_shape;
    pk_shape.PushBack(batch);
    pk_shape.PushBack(OptimDim(kv_num_heads));
    pk_shape.PushBack(total_seq_len);
    pk_shape.PushBack(head_size);
    ctx.Set(node.output(1), OptimTensor(nullptr, Q.Dtype(), std::move(pk_shape)));
  }

  // Output 2: present_value = (batch, kv_num_heads, total_seq_len, v_head_size).
  if (node.output_size() > 2 && !std::string(node.output(2)).empty()) {
    OptimShape pv_shape;
    pv_shape.PushBack(batch);
    pv_shape.PushBack(OptimDim(kv_num_heads));
    pv_shape.PushBack(total_seq_len);
    pv_shape.PushBack(v_head_size);
    ctx.Set(node.output(2), OptimTensor(nullptr, V.Dtype(), std::move(pv_shape)));
  }

  // Output 3: qk_matmul_output = (batch, q_num_heads, q_seq_len, total_seq_len).
  if (node.output_size() > 3 && !std::string(node.output(3)).empty()) {
    OptimShape qk_shape;
    qk_shape.PushBack(batch);
    qk_shape.PushBack(OptimDim(q_num_heads));
    qk_shape.PushBack(q_seq_len);
    qk_shape.PushBack(total_seq_len);
    ctx.Set(node.output(3), OptimTensor(nullptr, Q.Dtype(), std::move(qk_shape)));
  }
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

  // Q, K and V may be provided either as rank-4 ``(batch, num_heads, seq,
  // head_size)`` tensors or as rank-3 ``(batch, seq, num_heads * head_size)``
  // packed tensors. They must all share the same rank.
  EXT_ENFORCE_INVALID(
      !(q_shape.Rank() != k_shape.Rank() || q_shape.Rank() != v_shape.Rank()),
      "ComputeShapeAttention: inputs 'Q', 'K' and 'V' must share the same rank, got ",
      q_shape.Rank(), ", ", k_shape.Rank(), " and ", v_shape.Rank(), ".");
  if (q_shape.Rank() == 3) {
    ComputeShapeAttentionRank3(ctx, node, Q, K, V, past_key, past_value);
    return;
  }
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
    EXT_ENFORCE_INVALID(!(hq != hkv && (hkv <= 0 || (hq % hkv) != 0)),
                        "ComputeShapeAttention: q_num_heads (", hq,
                        ") must be a multiple of kv_num_heads (", hkv, ") when they differ.");
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
  if (node.output_size() > 1 && !std::string(node.output(1)).empty()) {
    OptimShape pk_shape;
    pk_shape.PushBack(batch);
    pk_shape.PushBack(kv_num_heads);
    pk_shape.PushBack(total_seq_len);
    pk_shape.PushBack(head_size);
    ctx.Set(node.output(1), OptimTensor(nullptr, Q.Dtype(), std::move(pk_shape)));
  }

  // Output 2: present_value = (batch, kv_num_heads, total_seq_len, v_head_size).
  if (node.output_size() > 2 && !std::string(node.output(2)).empty()) {
    OptimShape pv_shape;
    pv_shape.PushBack(batch);
    pv_shape.PushBack(kv_num_heads);
    pv_shape.PushBack(total_seq_len);
    pv_shape.PushBack(v_shape[3]);
    ctx.Set(node.output(2), OptimTensor(nullptr, V.Dtype(), std::move(pv_shape)));
  }

  // Output 3: qk_matmul_output = (batch, q_num_heads, q_seq_len, total_seq_len).
  if (node.output_size() > 3 && !std::string(node.output(3)).empty()) {
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
