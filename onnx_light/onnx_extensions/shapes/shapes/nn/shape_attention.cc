// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

namespace {

// Merges two dimensions: if both are static they must agree, otherwise
// the static one (if any) wins; if both are symbolic the first one is
// kept.
SymDim MergeDim(const SymDim &a, const SymDim &b, const char *what) {
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

void RequireRank4(const SymShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 4, "ComputeShapeAttention: input '", name,
                      "' must have rank 4, got ", shape.Rank(), ".");
}

// Returns ``a + b`` when both dims are static; otherwise returns ``a``.
SymDim AddDims(const SymDim &a, const SymDim &b) {
  if (a.IsInt() && b.IsInt()) {
    return SymDim(a.AsInt() + b.AsInt());
  }
  return a;
}

// Divides a (possibly symbolic) hidden dimension by ``num_heads``. When the
// dimension is static it must be a positive multiple of ``num_heads``;
// otherwise a fresh symbolic placeholder is returned.
SymDim DivideByHeads(const SymDim &hidden, int64_t num_heads, const char *what) {
  if (!hidden.IsInt()) {
    return SymDim(std::string("?"));
  }
  const int64_t value = hidden.AsInt();
  EXT_ENFORCE_INVALID((value % num_heads) == 0, "ComputeShapeAttention: ", what, " (", value,
                      ") must be divisible by num_heads (", num_heads, ").");
  return SymDim(value / num_heads);
}

// Computes ``num_heads * head`` when ``head`` is static; otherwise returns a
// fresh symbolic placeholder.
SymDim MultiplyByHeads(int64_t num_heads, const SymDim &head) {
  if (head.IsInt()) {
    return SymDim(num_heads * head.AsInt());
  }
  return SymDim(std::string("?"));
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

void ValidateAttention25Attributes(const NodeProto &node, int64_t input_rank) {
  bool has_q_num_heads = false;
  bool has_kv_num_heads = false;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "left_window_size" || attr.name() == "right_window_size") {
      EXT_ENFORCE_INVALID(attr.i() >= -1, "ComputeShapeAttention: '", attr.name(),
                          "' must be -1 or nonnegative.");
    } else if (attr.name() == "q_num_heads") {
      has_q_num_heads = true;
    } else if (attr.name() == "kv_num_heads") {
      has_kv_num_heads = true;
    }
  }
  EXT_ENFORCE_INVALID(input_rank != 4 || (!has_q_num_heads && !has_kv_num_heads),
                      "ComputeShapeAttention: q_num_heads and kv_num_heads must not be specified "
                      "for rank-4 inputs.");

  const bool has_past_key = node.input_size() > 4 && !node.input(4).empty();
  const bool has_past_value = node.input_size() > 5 && !node.input(5).empty();
  const bool has_nonpad = node.input_size() > 6 && !node.input(6).empty();
  const bool has_present_key = node.output_size() > 1 && !node.output(1).empty();
  const bool has_present_value = node.output_size() > 2 && !node.output(2).empty();
  EXT_ENFORCE_INVALID(has_past_key == has_past_value,
                      "ComputeShapeAttention: past_key and past_value must be provided together.");
  EXT_ENFORCE_INVALID(
      has_present_key == has_present_value,
      "ComputeShapeAttention: present_key and present_value must be requested together.");
  EXT_ENFORCE_INVALID(
      !has_nonpad || (!has_past_key && !has_present_key),
      "ComputeShapeAttention: nonpad_kv_seqlen cannot be combined with cache tensors.");
}

// Handles the rank-3 (packed) form of Attention, where Q, K and V have shapes
// ``(batch, q_sequence_length, q_num_heads * head_size)`` and
// ``(batch, kv_sequence_length, kv_num_heads * head_size)``. The
// ``q_num_heads`` and ``kv_num_heads`` attributes are required to unpack the
// last dimension. Output 0 stays rank-3 while the optional present/qk outputs
// are rank-4.
void ComputeShapeAttentionRank3(ShapesContext &ctx, const NodeProto &node, const SymTensor &Q,
                                const SymTensor &K, const SymTensor &V, const char *past_key,
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

  const SymShape &q_shape = Q.Shape();
  const SymShape &k_shape = K.Shape();
  const SymShape &v_shape = V.Shape();

  // Batch: Q[0] == K[0] == V[0]; q/kv sequence lengths from axis 1.
  SymDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");
  SymDim q_seq_len = q_shape[1];
  SymDim kv_seq_len = MergeDim(k_shape[1], v_shape[1], "kv_sequence_length");

  // head_size = Q[2] / q_num_heads == K[2] / kv_num_heads.
  SymDim head_size = DivideByHeads(q_shape[2], q_num_heads, "query hidden size");
  head_size =
      MergeDim(head_size, DivideByHeads(k_shape[2], kv_num_heads, "key hidden size"), "head_size");
  // v_head_size = V[2] / kv_num_heads.
  SymDim v_head_size = DivideByHeads(v_shape[2], kv_num_heads, "value hidden size");

  // total_sequence_length = past_sequence_length + kv_sequence_length.
  SymDim total_seq_len = kv_seq_len;
  if (past_key != nullptr && ctx.Has(past_key)) {
    const SymShape &past_k_shape = ctx.Get(past_key).Shape();
    RequireRank4(past_k_shape, past_key);
    batch = MergeDim(batch, past_k_shape[0], "batch");
    head_size = MergeDim(head_size, past_k_shape[3], "head_size");
    total_seq_len = AddDims(kv_seq_len, past_k_shape[2]);
  }
  if (past_value != nullptr && ctx.Has(past_value)) {
    const SymShape &past_v_shape = ctx.Get(past_value).Shape();
    RequireRank4(past_v_shape, past_value);
    batch = MergeDim(batch, past_v_shape[0], "batch");
    if (past_key == nullptr || !ctx.Has(past_key)) {
      total_seq_len = AddDims(kv_seq_len, past_v_shape[2]);
    }
  }

  // Output 0: Y = (batch, q_sequence_length, q_num_heads * v_head_size).
  {
    SymShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(q_seq_len);
    out_shape.PushBack(MultiplyByHeads(q_num_heads, v_head_size));
    ctx.Set(node.output(0), SymTensor(nullptr, Q.Dtype(), std::move(out_shape)));
  }

  // Output 1: present_key = (batch, kv_num_heads, total_seq_len, head_size).
  if (node.output_size() > 1 && !node.output(1).empty()) {
    SymShape pk_shape;
    pk_shape.PushBack(batch);
    pk_shape.PushBack(SymDim(kv_num_heads));
    pk_shape.PushBack(total_seq_len);
    pk_shape.PushBack(head_size);
    ctx.Set(node.output(1), SymTensor(nullptr, Q.Dtype(), std::move(pk_shape)));
  }

  // Output 2: present_value = (batch, kv_num_heads, total_seq_len, v_head_size).
  if (node.output_size() > 2 && !node.output(2).empty()) {
    SymShape pv_shape;
    pv_shape.PushBack(batch);
    pv_shape.PushBack(SymDim(kv_num_heads));
    pv_shape.PushBack(total_seq_len);
    pv_shape.PushBack(v_head_size);
    ctx.Set(node.output(2), SymTensor(nullptr, V.Dtype(), std::move(pv_shape)));
  }

  // Output 3: qk_matmul_output = (batch, q_num_heads, q_seq_len, total_seq_len).
  if (node.output_size() > 3 && !node.output(3).empty()) {
    SymShape qk_shape;
    qk_shape.PushBack(batch);
    qk_shape.PushBack(SymDim(q_num_heads));
    qk_shape.PushBack(q_seq_len);
    qk_shape.PushBack(total_seq_len);
    ctx.Set(node.output(3), SymTensor(nullptr, Q.Dtype(), std::move(qk_shape)));
  }
}

} // namespace

void ComputeShapeAttention(ShapesContext &ctx, const NodeProto &node, const char *q, const char *k,
                           const char *v, const char *past_key, const char *past_value) {
  CheckNodeOpAndOutput(node, "Attention", "ComputeShapeAttention");

  const SymTensor &Q = ctx.Get(q);
  const SymTensor &K = ctx.Get(k);
  const SymTensor &V = ctx.Get(v);

  const SymShape &q_shape = Q.Shape();
  const SymShape &k_shape = K.Shape();
  const SymShape &v_shape = V.Shape();

  // Q, K and V may be provided either as rank-4 ``(batch, num_heads, seq,
  // head_size)`` tensors or as rank-3 ``(batch, seq, num_heads * head_size)``
  // packed tensors. They must all share the same rank.
  EXT_ENFORCE_INVALID(
      !(q_shape.Rank() != k_shape.Rank() || q_shape.Rank() != v_shape.Rank()),
      "ComputeShapeAttention: inputs 'Q', 'K' and 'V' must share the same rank, got ",
      q_shape.Rank(), ", ", k_shape.Rank(), " and ", v_shape.Rank(), ".");
  ValidateAttention25Attributes(node, q_shape.Rank());
  if (q_shape.Rank() == 3) {
    ComputeShapeAttentionRank3(ctx, node, Q, K, V, past_key, past_value);
    return;
  }
  RequireRank4(q_shape, q);
  RequireRank4(k_shape, k);
  RequireRank4(v_shape, v);

  // Batch: Q[0] == K[0] == V[0] (when static).
  SymDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");

  // K and V share the same number of heads.
  SymDim kv_num_heads = MergeDim(k_shape[1], v_shape[1], "kv_num_heads");

  // K and V share the same sequence length.
  SymDim kv_seq_len = MergeDim(k_shape[2], v_shape[2], "kv_sequence_length");

  // Q and K share the same head dimension.
  SymDim head_size = MergeDim(q_shape[3], k_shape[3], "head_size");

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
  SymDim total_seq_len = kv_seq_len;
  if (past_key != nullptr && ctx.Has(past_key)) {
    const SymShape &past_k_shape = ctx.Get(past_key).Shape();
    RequireRank4(past_k_shape, past_key);
    batch = MergeDim(batch, past_k_shape[0], "batch");
    kv_num_heads = MergeDim(kv_num_heads, past_k_shape[1], "kv_num_heads");
    head_size = MergeDim(head_size, past_k_shape[3], "head_size");
    total_seq_len = AddDims(kv_seq_len, past_k_shape[2]);
  }
  if (past_value != nullptr && ctx.Has(past_value)) {
    const SymShape &past_v_shape = ctx.Get(past_value).Shape();
    RequireRank4(past_v_shape, past_value);
    batch = MergeDim(batch, past_v_shape[0], "batch");
    kv_num_heads = MergeDim(kv_num_heads, past_v_shape[1], "kv_num_heads");
    if (past_key == nullptr || !ctx.Has(past_key)) {
      total_seq_len = AddDims(kv_seq_len, past_v_shape[2]);
    }
  }

  // Output 0: Y = (batch, q_num_heads, q_seq_len, v_head_size).
  {
    SymShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(q_shape[1]);
    out_shape.PushBack(q_shape[2]);
    out_shape.PushBack(v_shape[3]);
    ctx.Set(node.output(0), SymTensor(nullptr, Q.Dtype(), std::move(out_shape)));
  }

  // Output 1: present_key = (batch, kv_num_heads, total_seq_len, head_size).
  if (node.output_size() > 1 && !node.output(1).empty()) {
    SymShape pk_shape;
    pk_shape.PushBack(batch);
    pk_shape.PushBack(kv_num_heads);
    pk_shape.PushBack(total_seq_len);
    pk_shape.PushBack(head_size);
    ctx.Set(node.output(1), SymTensor(nullptr, Q.Dtype(), std::move(pk_shape)));
  }

  // Output 2: present_value = (batch, kv_num_heads, total_seq_len, v_head_size).
  if (node.output_size() > 2 && !node.output(2).empty()) {
    SymShape pv_shape;
    pv_shape.PushBack(batch);
    pv_shape.PushBack(kv_num_heads);
    pv_shape.PushBack(total_seq_len);
    pv_shape.PushBack(v_shape[3]);
    ctx.Set(node.output(2), SymTensor(nullptr, V.Dtype(), std::move(pv_shape)));
  }

  // Output 3: qk_matmul_output = (batch, q_num_heads, q_seq_len, total_seq_len).
  if (node.output_size() > 3 && !node.output(3).empty()) {
    SymShape qk_shape;
    qk_shape.PushBack(batch);
    qk_shape.PushBack(q_shape[1]);
    qk_shape.PushBack(q_shape[2]);
    qk_shape.PushBack(total_seq_len);
    ctx.Set(node.output(3), SymTensor(nullptr, Q.Dtype(), std::move(qk_shape)));
  }
}

int64_t ComputePeakMemoryAttention(Device device, const std::vector<SymShape> &input_shapes) {
  // The score buffer accumulation type; scores are held in float32.
  constexpr int64_t kScoreElementBytes = 4;

  // Q and K carry the dimensions needed for the QK^T score buffer. Without
  // both, or without the rank-4 (batch, num_heads, sequence, head_size)
  // layout, no concrete estimate can be produced.
  (void)device;
  if (input_shapes.size() < 2) {
    return 0;
  }
  const SymShape &q_shape = input_shapes[0];
  const SymShape &k_shape = input_shapes[1];
  if (q_shape.Rank() != 4 || k_shape.Rank() != 4) {
    return 0;
  }

  // Score buffer: (batch, q_num_heads, q_sequence_length, kv_sequence_length).
  const SymDim &batch = q_shape[0];
  const SymDim &q_num_heads = q_shape[1];
  const SymDim &q_seq_len = q_shape[2];
  const SymDim &kv_seq_len = k_shape[2];
  if (!batch.IsInt() || !q_num_heads.IsInt() || !q_seq_len.IsInt() || !kv_seq_len.IsInt()) {
    return 0;
  }

  return batch.AsInt() * q_num_heads.AsInt() * q_seq_len.AsInt() * kv_seq_len.AsInt() *
         kScoreElementBytes;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
