// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <string>
#include <utility>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

namespace {

SymDim MergeDim(const SymDim &a, const SymDim &b, const char *what) {
  if (a.IsInt() && b.IsInt()) {
    EXT_ENFORCE_INVALID(a.AsInt() == b.AsInt(), "ComputeShapeLinearAttention: ", what,
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

void RequireRank3(const SymShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 3, "ComputeShapeLinearAttention: input '", name,
                      "' must have rank 3, got ", shape.Rank(), ".");
}

void RequireRank4(const SymShape &shape, const char *name) {
  EXT_ENFORCE_INVALID(shape.Rank() == 4, "ComputeShapeLinearAttention: input '", name,
                      "' must have rank 4, got ", shape.Rank(), ".");
}

} // namespace

void ComputeShapeLinearAttention(ShapesContext &ctx, const NodeProto &node, const char *query,
                                 const char *key, const char *value, const char *past_state) {
  CheckNodeOpAndOutput(node, "LinearAttention", "ComputeShapeLinearAttention");

  // Required attributes ``q_num_heads`` and ``kv_num_heads``.
  int64_t q_num_heads = 0;
  int64_t kv_num_heads = 0;
  bool has_q = false;
  bool has_kv = false;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "q_num_heads") {
      q_num_heads = attr.i();
      has_q = true;
    } else if (attr.name() == "kv_num_heads") {
      kv_num_heads = attr.i();
      has_kv = true;
    }
  }
  EXT_ENFORCE_INVALID(
      !(!has_q || !has_kv),
      "ComputeShapeLinearAttention: attributes 'q_num_heads' and 'kv_num_heads' are required.");
  EXT_ENFORCE_INVALID(
      !(q_num_heads <= 0 || kv_num_heads <= 0),
      "ComputeShapeLinearAttention: 'q_num_heads' and 'kv_num_heads' must be positive.");
  EXT_ENFORCE_INVALID((q_num_heads % kv_num_heads) == 0,
                      "ComputeShapeLinearAttention: q_num_heads (", q_num_heads,
                      ") must be a multiple of kv_num_heads (", kv_num_heads, ").");

  const SymTensor &Q = ctx.Get(query);
  const SymTensor &K = ctx.Get(key);
  const SymTensor &V = ctx.Get(value);

  const SymShape &q_shape = Q.Shape();
  const SymShape &k_shape = K.Shape();
  const SymShape &v_shape = V.Shape();
  RequireRank3(q_shape, query);
  RequireRank3(k_shape, key);
  RequireRank3(v_shape, value);

  // Batch and sequence length: shared between Q, K and V.
  SymDim batch = MergeDim(q_shape[0], k_shape[0], "batch");
  batch = MergeDim(batch, v_shape[0], "batch");
  SymDim seq_len = MergeDim(q_shape[1], k_shape[1], "sequence_length");
  seq_len = MergeDim(seq_len, v_shape[1], "sequence_length");

  // d_k = key.shape[-1] / kv_num_heads; d_v = value.shape[-1] / kv_num_heads.
  // When the value/key last dim is symbolic we fall back to a fresh symbolic
  // placeholder ("?") so the rank is correct even if the exact value is
  // unknown. ``past_state`` (if provided) can refine these below.
  SymDim d_k;
  bool d_k_resolved = false;
  if (k_shape[2].IsInt()) {
    const int64_t hidden_k = k_shape[2].AsInt();
    EXT_ENFORCE_INVALID((hidden_k % kv_num_heads) == 0,
                        "ComputeShapeLinearAttention: key last dim (", hidden_k,
                        ") must be divisible by kv_num_heads (", kv_num_heads, ").");
    if (q_shape[2].IsInt()) {
      const int64_t hidden_q = q_shape[2].AsInt();
      EXT_ENFORCE_INVALID((hidden_q % q_num_heads) == 0,
                          "ComputeShapeLinearAttention: query last dim (", hidden_q,
                          ") must be divisible by q_num_heads (", q_num_heads, ").");
      const int64_t q_d_k = hidden_q / q_num_heads;
      const int64_t k_d_k = hidden_k / kv_num_heads;
      EXT_ENFORCE_INVALID(q_d_k == k_d_k, "ComputeShapeLinearAttention: query head_size (", q_d_k,
                          ") must equal key head_size (", k_d_k, ").");
    }
    d_k = SymDim(hidden_k / kv_num_heads);
    d_k_resolved = true;
  }

  SymDim d_v;
  bool d_v_resolved = false;
  if (v_shape[2].IsInt()) {
    const int64_t hidden_v = v_shape[2].AsInt();
    EXT_ENFORCE_INVALID((hidden_v % kv_num_heads) == 0,
                        "ComputeShapeLinearAttention: value last dim (", hidden_v,
                        ") must be divisible by kv_num_heads (", kv_num_heads, ").");
    d_v = SymDim(hidden_v / kv_num_heads);
    d_v_resolved = true;
  }

  // Cross-check / refine using past_state (B, H_kv, d_k, d_v) if present.
  if (past_state != nullptr && ctx.Has(past_state)) {
    const SymShape &ps_shape = ctx.Get(past_state).Shape();
    RequireRank4(ps_shape, past_state);
    batch = MergeDim(batch, ps_shape[0], "batch");
    EXT_ENFORCE_INVALID(!(ps_shape[1].IsInt() && ps_shape[1].AsInt() != kv_num_heads),
                        "ComputeShapeLinearAttention: past_state dim 1 (", ps_shape[1].AsInt(),
                        ") must equal kv_num_heads (", kv_num_heads, ").");
    if (d_k_resolved) {
      d_k = MergeDim(d_k, ps_shape[2], "d_k");
    } else {
      d_k = ps_shape[2];
      d_k_resolved = ps_shape[2].IsInt();
    }
    if (d_v_resolved) {
      d_v = MergeDim(d_v, ps_shape[3], "d_v");
    } else {
      d_v = ps_shape[3];
      d_v_resolved = ps_shape[3].IsInt();
    }
  }

  // Compute output last dim = q_num_heads * d_v.
  SymDim out_last;
  if (d_v_resolved) {
    out_last = SymDim(q_num_heads * d_v.AsInt());
  } else if (q_num_heads == kv_num_heads && !v_shape[2].IsInt()) {
    // Special case: q_num_heads * d_v = q_num_heads * (v_shape[2] / kv_num_heads) = v_shape[2].
    out_last = v_shape[2];
  } else {
    out_last = SymDim(std::string("?"));
  }
  if (!d_k_resolved) {
    d_k = SymDim(std::string("?"));
  }
  if (!d_v_resolved) {
    d_v = SymDim(std::string("?"));
  }

  // Output 0: output = (B, T, q_num_heads * d_v).
  {
    SymShape out_shape;
    out_shape.PushBack(batch);
    out_shape.PushBack(seq_len);
    out_shape.PushBack(out_last);
    ctx.Set(node.output(0), SymTensor(nullptr, Q.Dtype(), std::move(out_shape)));
  }

  // Output 1: present_state = (B, kv_num_heads, d_k, d_v).
  if (node.output_size() > 1 && !node.output(1).empty()) {
    SymShape ps_shape;
    ps_shape.PushBack(batch);
    ps_shape.PushBack(SymDim(kv_num_heads));
    ps_shape.PushBack(d_k);
    ps_shape.PushBack(d_v);
    // Dtype: prefer past_state dtype when available, otherwise inherit from Q.
    TensorType state_dtype = Q.Dtype();
    if (past_state != nullptr && ctx.Has(past_state)) {
      state_dtype = ctx.Get(past_state).Dtype();
    }
    ctx.Set(node.output(1), SymTensor(nullptr, state_dtype, std::move(ps_shape)));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
