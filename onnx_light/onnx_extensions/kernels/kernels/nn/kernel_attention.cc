// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Validates that ``t`` is a rank-4 tensor whose element type is supported
// by the Attention kernel (FLOAT, FLOAT16, or BFLOAT16). The caller is
// identified by ``label`` for clearer error messages.
void CheckRank4Float(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT || t.data_type == DataType::FLOAT16 ||
                          t.data_type == DataType::BFLOAT16,
                      "kernel::Attention: '", label,
                      "' must be a FLOAT, FLOAT16 or BFLOAT16 tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4, "kernel::Attention: '", label,
                      "' must be a rank-4 tensor.");
  for (int64_t d : t.shape) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::Attention: '", label, "' has a negative dimension.");
  }
}

// Promotes a rank-3 fused ``(batch, seq, num_heads * head_size)`` FLOAT tensor
// into the rank-4 layout ``(batch, num_heads, seq, head_size)`` used by the
// internal kernel. Returns the promoted tensor; ``num_heads`` must divide
// ``hidden_size``.
Tensor PromoteRank3(const Tensor &t, int64_t num_heads, const char *label,
                    RuntimeContext *rt = nullptr) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, "kernel::Attention: '", label,
                      "' must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 3, "kernel::Attention: '", label, "' must be rank-3.");
  const int64_t batch = t.shape[0];
  const int64_t seq = t.shape[1];
  const int64_t hidden = t.shape[2];
  EXT_ENFORCE_INVALID(num_heads > 0, "kernel::Attention: '", label,
                      "' needs a positive ``num_heads`` to promote rank-3.");
  EXT_ENFORCE_INVALID(hidden % num_heads == 0, "kernel::Attention: '", label,
                      "' hidden_size must be a multiple of ``num_heads``.");
  const int64_t head_size = hidden / num_heads;
  const size_t out_n_bytes = t.size_bytes();
  Tensor out =
      rt ? rt->MakeTemporaryTensor(DataType::FLOAT, {batch, num_heads, seq, head_size}, out_n_bytes)
         : MakeOutputTensor(DataType::FLOAT, {batch, num_heads, seq, head_size}, out_n_bytes,
                            nullptr);
  const float *src = t.AsFloat();
  float *dst = out.AsFloat();
  // src strides: (seq*hidden, hidden, 1) over (batch, seq, hidden).
  // hidden indexed as (h * head_size + d).
  // dst strides: (num_heads*seq*head_size, seq*head_size, head_size, 1)
  // over (batch, h, seq, d).
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t d = 0; d < head_size; ++d) {
          dst[((b * num_heads + h) * seq + s) * head_size + d] =
              src[(b * seq + s) * hidden + h * head_size + d];
        }
      }
    }
  }
  return out;
}

// Inverse of ``PromoteRank3``: collapses a rank-4 ``(batch, num_heads, seq,
// head_size)`` tensor into the rank-3 fused layout
// ``(batch, seq, num_heads * head_size)``.
Tensor CollapseToRank3(const Tensor &t, RuntimeContext *rt = nullptr) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, "kernel::Attention: must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4, "kernel::Attention: must be rank-4.");
  const int64_t batch = t.shape[0];
  const int64_t num_heads = t.shape[1];
  const int64_t seq = t.shape[2];
  const int64_t head_size = t.shape[3];
  const int64_t hidden = num_heads * head_size;
  const size_t out_n_bytes = t.size_bytes();
  Tensor out = rt ? rt->MakeOutputTensor(0, DataType::FLOAT, {batch, seq, hidden}, out_n_bytes)
                  : MakeOutputTensor(DataType::FLOAT, {batch, seq, hidden}, out_n_bytes, nullptr);
  const float *src = t.AsFloat();
  float *dst = out.AsFloat();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < num_heads; ++h) {
      for (int64_t s = 0; s < seq; ++s) {
        for (int64_t d = 0; d < head_size; ++d) {
          dst[(b * seq + s) * hidden + h * head_size + d] =
              src[((b * num_heads + h) * seq + s) * head_size + d];
        }
      }
    }
  }
  return out;
}

// Concatenates two rank-4 FLOAT tensors along axis 2 (the sequence axis).
// Returns a new tensor whose shape equals ``a.shape`` with axis 2 = ``a[2] +
// b[2]``; axes 0, 1, 3 must match between ``a`` and ``b``.
Tensor AllocateResult(RuntimeContext *rt, int output_slot, int32_t data_type, const Shape &shape,
                      size_t n_bytes) {
  if (rt == nullptr) {
    return MakeOutputTensor(data_type, shape, n_bytes, nullptr);
  }
  const auto &roles = rt->output_slot_io_roles();
  if (!roles.empty() && static_cast<size_t>(output_slot) >= roles.size()) {
    return rt->MakeTemporaryTensor(data_type, shape, n_bytes);
  }
  return rt->MakeOutputTensor(output_slot, data_type, shape, n_bytes);
}

bool HasRecordedOutputSlot(const RuntimeContext *rt, int output_slot) {
  return rt != nullptr && !rt->output_slot_io_roles().empty() && output_slot >= 0 &&
         static_cast<size_t>(output_slot) < rt->output_slot_io_roles().size();
}

Tensor ConcatAxis2(const Tensor &a, const Tensor &b, int output_slot,
                   RuntimeContext *rt = nullptr) {
  EXT_ENFORCE_INVALID(a.shape.size() == 4 && b.shape.size() == 4,
                      "kernel::Attention: concat inputs must be rank-4.");
  EXT_ENFORCE_INVALID(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] &&
                          a.shape[3] == b.shape[3],
                      "kernel::Attention: past KV shape must match current K/V on axes 0, 1, 3.");
  const int64_t batch = a.shape[0];
  const int64_t heads = a.shape[1];
  const int64_t la = a.shape[2];
  const int64_t lb = b.shape[2];
  const int64_t d = a.shape[3];
  const int64_t lc = la + lb;
  const size_t out_n_bytes = static_cast<size_t>(batch * heads * lc * d) * sizeof(float);
  Tensor out = AllocateResult(rt, output_slot, DataType::FLOAT, {batch, heads, lc, d}, out_n_bytes);
  const float *pa = a.AsFloat();
  const float *pb = b.AsFloat();
  float *po = out.AsFloat();
  for (int64_t bi = 0; bi < batch; ++bi) {
    for (int64_t h = 0; h < heads; ++h) {
      // copy ``a`` slice
      const int64_t off_a = (bi * heads + h) * la * d;
      const int64_t off_o = (bi * heads + h) * lc * d;
      std::copy(pa + off_a, pa + off_a + la * d, po + off_o);
      const int64_t off_b = (bi * heads + h) * lb * d;
      std::copy(pb + off_b, pb + off_b + lb * d, po + off_o + la * d);
    }
  }
  return out;
}

Tensor CopyOutput(const Tensor &src, int output_slot, RuntimeContext *rt) {
  Tensor out = AllocateResult(rt, output_slot, src.data_type, src.shape, src.size_bytes());
  if (src.size_bytes() != 0) {
    std::memcpy(out.mutable_bytes(), src.bytes(), src.size_bytes());
  }
  return out;
}

// Returns the value of a FLOAT or BOOL ``attn_mask`` at the broadcasted index
// ``(b, h, i, j)``. The mask is treated as if its shape were left-padded
// with leading 1s up to rank 4. Each axis must either equal 1 (broadcast) or
// the corresponding output dimension. BOOL masks return ``0`` for true (no
// penalty) and ``-inf`` for false (mask out).
double BroadcastedMaskValue(const Tensor &mask, int64_t batch_size, int64_t q_num_heads,
                            int64_t q_seq_len, int64_t kv_seq_len, int64_t b, int64_t h, int64_t i,
                            int64_t j) {
  const int rank = static_cast<int>(mask.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1 && rank <= 4,
                      "kernel::Attention: 'attn_mask' must have rank between 1 and 4.");
  int64_t shape4[4] = {1, 1, 1, 1};
  for (int k = 0; k < rank; ++k) {
    shape4[4 - rank + k] = mask.shape[static_cast<size_t>(k)];
  }
  const int64_t out_dims[4] = {batch_size, q_num_heads, q_seq_len, kv_seq_len};
  for (int k = 0; k < 4; ++k) {
    EXT_ENFORCE_INVALID(shape4[k] == 1 || shape4[k] == out_dims[k],
                        "kernel::Attention: 'attn_mask' is not broadcastable to (batch_size, "
                        "q_num_heads, q_seq_len, kv_seq_len).");
  }
  const int64_t idx[4] = {b, h, i, j};
  int64_t linear = 0;
  int64_t stride = 1;
  for (int k = 3; k >= 0; --k) {
    const int64_t coord = shape4[k] == 1 ? 0 : idx[k];
    linear += coord * stride;
    stride *= shape4[k];
  }
  if (mask.data_type == DataType::FLOAT) {
    return static_cast<double>(mask.AsFloat()[linear]);
  }
  if (mask.data_type == DataType::BOOL) {
    return mask.AsBool()[linear] != 0 ? 0.0 : -std::numeric_limits<double>::infinity();
  }
  EXT_ENFORCE_INVALID(false, "kernel::Attention: 'attn_mask' must be FLOAT or BOOL.");
  return 0.0;
}

// Returns the mask value at ``(b, h, i, j)`` allowing the mask to be
// shorter than ``total_kv_seq_len`` along its last axis: missing positions
// are treated as ``-inf`` (FLOAT) / ``false`` (BOOL), matching the upstream
// ``np.pad(..., constant_values=-inf)`` behaviour. Returns ``0.0`` when
// ``mask == nullptr``.
double MaskValuePadded(const Tensor *mask, int64_t batch_size, int64_t q_num_heads,
                       int64_t q_seq_len, int64_t /*kv_seq_len*/, int64_t b, int64_t h, int64_t i,
                       int64_t j) {
  if (mask == nullptr) {
    return 0.0;
  }
  const int rank = static_cast<int>(mask->shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::Attention: 'attn_mask' must have rank >= 1.");
  const int64_t mask_kv = mask->shape[static_cast<size_t>(rank - 1)];
  if (j >= mask_kv) {
    return -std::numeric_limits<double>::infinity();
  }
  return BroadcastedMaskValue(*mask, batch_size, q_num_heads, q_seq_len, mask_kv, b, h, i, j);
}

// Core rank-4 Attention. ``Q4``/``K4``/``V4`` are already in the internal
// ``(batch, num_heads, seq, head_size)`` layout. Builds ``present_key`` /
// ``present_value`` (concatenating the optional past tensors), runs the
// scaled dot-product attention and returns a rank-4 ``Y`` together with the
// present tensors and the mode-dependent ``qk_matmul_output``.
Attention::Result ComputeAttentionRank4(const Tensor &Q4, const Tensor &K4, const Tensor &V4,
                                        const Attention::Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value,
                                        const Tensor *nonpad_kv_seqlen, RuntimeContext *rt,
                                        bool temporary_y = false) {
  CheckRank4Float(Q4, "Q");
  CheckRank4Float(K4, "K");
  CheckRank4Float(V4, "V");

  const int64_t batch_size = Q4.shape[0];
  const int64_t q_num_heads = Q4.shape[1];
  const int64_t q_seq_len = Q4.shape[2];
  const int64_t head_size = Q4.shape[3];
  const int64_t kv_num_heads = K4.shape[1];

  EXT_ENFORCE_INVALID(K4.shape[0] == batch_size && V4.shape[0] == batch_size,
                      "kernel::Attention: Q, K, V must share the same batch size.");
  EXT_ENFORCE_INVALID(K4.shape[3] == head_size,
                      "kernel::Attention: K head_size must match Q head_size.");
  EXT_ENFORCE_INVALID(V4.shape[1] == kv_num_heads,
                      "kernel::Attention: V num_heads must match K num_heads.");
  EXT_ENFORCE_INVALID(V4.shape[2] == K4.shape[2],
                      "kernel::Attention: V kv_seq_len must match K kv_seq_len.");
  EXT_ENFORCE_INVALID(
      kv_num_heads > 0 && q_num_heads % kv_num_heads == 0,
      "kernel::Attention: q_num_heads must be a positive multiple of kv_num_heads.");

  // ----- Build present_key / present_value -------------------------------
  // The upstream operator concatenates past_key/past_value with K/V along
  // the sequence axis. When neither is supplied, present == K/V.
  Tensor present_key = past_key != nullptr
                           ? ConcatAxis2(*past_key, K4, 1, rt)
                           : (HasRecordedOutputSlot(rt, 1) ? CopyOutput(K4, 1, rt) : K4);
  Tensor present_value = past_value != nullptr
                             ? ConcatAxis2(*past_value, V4, 2, rt)
                             : (HasRecordedOutputSlot(rt, 2) ? CopyOutput(V4, 2, rt) : V4);
  const int64_t total_kv_seq_len = present_key.shape[2];
  const int64_t past_kv_seq_len = past_key != nullptr ? past_key->shape[2] : 0;
  const int64_t v_head_size = present_value.shape[3];

  // ----- Resolve nonpad_kv_seqlen ----------------------------------------
  // Optional 1-D INT64 tensor of length ``batch_size``. Positions
  // ``j >= nonpad_kv_seqlen[b]`` along the key/value sequence axis are
  // padding and masked out with ``-inf`` (mirroring the upstream
  // ``padding_mask`` derived from ``nonpad_kv_seqlen``).
  const int64_t *nonpad_lengths = nullptr;
  if (nonpad_kv_seqlen != nullptr && nonpad_kv_seqlen->size_bytes() != 0) {
    EXT_ENFORCE_INVALID(nonpad_kv_seqlen->data_type == DataType::INT64,
                        "kernel::Attention: 'nonpad_kv_seqlen' must be INT64.");
    EXT_ENFORCE_INVALID(
        nonpad_kv_seqlen->shape.size() == 1 && nonpad_kv_seqlen->shape[0] == batch_size,
        "kernel::Attention: 'nonpad_kv_seqlen' must be a 1-D tensor of length batch_size.");
    nonpad_lengths = nonpad_kv_seqlen->AsInt64();
  }

  // ----- Resolve scale ---------------------------------------------------
  const float scale =
      attrs.has_scale ? attrs.scale : 1.0f / std::sqrt(static_cast<float>(head_size));

  // ----- Allocate outputs ------------------------------------------------
  const int64_t out_count_y = batch_size * q_num_heads * q_seq_len * v_head_size;
  const size_t Y_n_bytes = static_cast<size_t>(out_count_y) * sizeof(float);
  const Shape y_shape{batch_size, q_num_heads, q_seq_len, v_head_size};
  Tensor Y = temporary_y && rt ? rt->MakeTemporaryTensor(DataType::FLOAT, y_shape, Y_n_bytes)
                               : AllocateResult(rt, 0, DataType::FLOAT, y_shape, Y_n_bytes);
  const int64_t qk_count = batch_size * q_num_heads * q_seq_len * total_kv_seq_len;
  const size_t qk_out_n_bytes = static_cast<size_t>(qk_count) * sizeof(float);
  Tensor qk_out =
      AllocateResult(rt, 3, DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, total_kv_seq_len},
                     qk_out_n_bytes);

  // ----- Compute ---------------------------------------------------------
  const int64_t group_size = q_num_heads / kv_num_heads;
  const float *pQ = Q4.AsFloat();
  const float *pK = present_key.AsFloat();
  const float *pV = present_value.AsFloat();
  float *pY = Y.AsFloat();
  float *pQK = qk_out.AsFloat();

  const int64_t q_head_stride = q_seq_len * head_size;
  const int64_t q_batch_stride = q_num_heads * q_head_stride;
  const int64_t k_head_stride = total_kv_seq_len * head_size;
  const int64_t k_batch_stride = kv_num_heads * k_head_stride;
  const int64_t v_head_stride = total_kv_seq_len * v_head_size;
  const int64_t v_batch_stride = kv_num_heads * v_head_stride;
  const int64_t y_head_stride = q_seq_len * v_head_size;
  const int64_t y_batch_stride = q_num_heads * y_head_stride;
  const int64_t qk_head_stride = q_seq_len * total_kv_seq_len;
  const int64_t qk_batch_stride = q_num_heads * qk_head_stride;

  // Upstream's _compute_attention multiplies Q and K by sqrt(scale)
  // separately and then matmuls, which is numerically equivalent to a
  // single ``* scale`` after the matmul.
  // The per-row scratch buffers are acquired from the runtime allocator (when
  // one is provided) so no working memory is allocated outside it; they fall
  // back to inline storage when ``allocator`` is null.
  const size_t scratch_n_bytes = static_cast<size_t>(total_kv_seq_len) * sizeof(double);
  Tensor scores_buf =
      rt ? rt->MakeTemporaryTensor(DataType::DOUBLE, {total_kv_seq_len}, scratch_n_bytes)
         : MakeOutputTensor(DataType::DOUBLE, {total_kv_seq_len}, scratch_n_bytes, nullptr);
  Tensor bias_buf =
      rt ? rt->MakeTemporaryTensor(DataType::DOUBLE, {total_kv_seq_len}, scratch_n_bytes)
         : MakeOutputTensor(DataType::DOUBLE, {total_kv_seq_len}, scratch_n_bytes, nullptr);
  Tensor qkraw_buf =
      rt ? rt->MakeTemporaryTensor(DataType::DOUBLE, {total_kv_seq_len}, scratch_n_bytes)
         : MakeOutputTensor(DataType::DOUBLE, {total_kv_seq_len}, scratch_n_bytes, nullptr);
  double *scores = scores_buf.AsDouble();
  double *bias = bias_buf.AsDouble();
  double *qkraw = qkraw_buf.AsDouble();
  for (int64_t b = 0; b < batch_size; ++b) {
    // Bottom-right / offset-aware causal frontier (mirrors onnx/onnx#8068):
    // a query at in-block index ``i`` attends key ``j`` iff ``j <= i + offset``,
    // where ``offset`` is the number of valid keys that precede this query block:
    //   * past_key present (internal cache):    offset = past_kv_seq_len
    //   * nonpad_kv_seqlen present, no past_key (external/static cache):
    //                                           offset = nonpad_kv_seqlen[b] - q_seq_len
    //   * neither:                              offset = 0 (ordinary top-left causal)
    // ``offset`` is intentionally not clamped to ``>= 0``: a negative offset
    // (out-of-contract over-long query block) fully masks the affected rows,
    // which the fully-masked-row guard below then zeroes.
    int64_t causal_offset = past_kv_seq_len;
    if (past_key == nullptr && nonpad_lengths != nullptr) {
      causal_offset = nonpad_lengths[b] - q_seq_len;
    }
    for (int64_t h = 0; h < q_num_heads; ++h) {
      const int64_t kv_h = h / group_size;
      const float *Qbh = pQ + b * q_batch_stride + h * q_head_stride;
      const float *Kbh = pK + b * k_batch_stride + kv_h * k_head_stride;
      const float *Vbh = pV + b * v_batch_stride + kv_h * v_head_stride;
      float *Ybh = pY + b * y_batch_stride + h * y_head_stride;
      float *QKbh = pQK + b * qk_batch_stride + h * qk_head_stride;

      for (int64_t i = 0; i < q_seq_len; ++i) {
        // Compute raw scaled QK scores and assemble bias.
        for (int64_t j = 0; j < total_kv_seq_len; ++j) {
          double s = 0.0;
          for (int64_t d = 0; d < head_size; ++d) {
            s += static_cast<double>(Qbh[i * head_size + d]) *
                 static_cast<double>(Kbh[j * head_size + d]);
          }
          s *= static_cast<double>(scale);
          qkraw[static_cast<size_t>(j)] = s;

          double b_val = MaskValuePadded(attn_mask, batch_size, q_num_heads, q_seq_len,
                                         total_kv_seq_len, b, h, i, j);
          if (attrs.is_causal) {
            // Bottom-right / offset-aware causal mask: key ``j`` is masked for
            // query ``i`` iff ``j > i + causal_offset``.
            if (j > i + causal_offset) {
              b_val = -std::numeric_limits<double>::infinity();
            }
          }
          if (nonpad_lengths != nullptr && j >= nonpad_lengths[b]) {
            // Padding position: suppressed regardless of the supplied mask.
            b_val = -std::numeric_limits<double>::infinity();
          }
          bias[static_cast<size_t>(j)] = b_val;
        }
        // qk_matmul_output_mode 0: raw QK^T * scale (no softcap, no bias).
        if (attrs.qk_matmul_output_mode == 0) {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            QKbh[i * total_kv_seq_len + j] = static_cast<float>(qkraw[static_cast<size_t>(j)]);
          }
        }
        // Apply softcap to raw QK BEFORE adding mask/bias. This matches
        // upstream's ordering so that ``-inf`` mask values survive into
        // softmax (yielding zero probability on masked positions). If
        // softcap were applied after the mask, ``sc * tanh(-inf / sc)``
        // would saturate to ``-sc`` (finite) and leak probability.
        if (attrs.softcap > 0.0f) {
          const double sc = static_cast<double>(attrs.softcap);
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            scores[static_cast<size_t>(j)] = sc * std::tanh(qkraw[static_cast<size_t>(j)] / sc);
          }
        } else {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            scores[static_cast<size_t>(j)] = qkraw[static_cast<size_t>(j)];
          }
        }
        // qk_matmul_output_mode 1: softcap output (before mask), or raw
        // when no softcap is requested.
        if (attrs.qk_matmul_output_mode == 1) {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            QKbh[i * total_kv_seq_len + j] = static_cast<float>(scores[static_cast<size_t>(j)]);
          }
        }
        // Add mask/bias to the (possibly softcapped) scores.
        double max_score = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < total_kv_seq_len; ++j) {
          const double s = scores[static_cast<size_t>(j)] + bias[static_cast<size_t>(j)];
          scores[static_cast<size_t>(j)] = s;
          if (s > max_score) {
            max_score = s;
          }
        }
        // qk_matmul_output_mode 2: includes attention mask and softcap.
        if (attrs.qk_matmul_output_mode == 2) {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            QKbh[i * total_kv_seq_len + j] = static_cast<float>(scores[static_cast<size_t>(j)]);
          }
        }
        // Softmax over last axis.
        double denom = 0.0;
        for (int64_t j = 0; j < total_kv_seq_len; ++j) {
          const double e = std::exp(scores[static_cast<size_t>(j)] - max_score);
          scores[static_cast<size_t>(j)] = e;
          denom += e;
        }
        const double inv_denom = denom != 0.0 ? 1.0 / denom : 0.0;
        for (int64_t j = 0; j < total_kv_seq_len; ++j) {
          scores[static_cast<size_t>(j)] *= inv_denom;
        }
        // A fully-masked row (every key masked out) has no attendable key: by
        // convention it softmaxes to all-zero probabilities. Detect it on the
        // additive bias (not the possibly-NaN logits) and zero the row BEFORE
        // capturing the mode-3 output so the exposed ``qk_matmul_output`` row is
        // also zeroed, consistent with the primary output ``Y`` (both 0). This
        // mirrors upstream's guard, which runs before the mode-3 capture.
        bool row_fully_masked = true;
        for (int64_t j = 0; j < total_kv_seq_len; ++j) {
          const double b = bias[static_cast<size_t>(j)];
          if (!(std::isinf(b) && b < 0.0)) {
            row_fully_masked = false;
            break;
          }
        }
        if (row_fully_masked) {
          std::fill(scores, scores + total_kv_seq_len, 0.0);
        }
        // qk_matmul_output_mode 3: after softmax (with the fully-masked-row
        // guard applied above).
        if (attrs.qk_matmul_output_mode == 3) {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            QKbh[i * total_kv_seq_len + j] = static_cast<float>(scores[static_cast<size_t>(j)]);
          }
        }
        // Y[i, dv] = sum_j probs[j] * V[j, dv]
        for (int64_t dv = 0; dv < v_head_size; ++dv) {
          double y = 0.0;
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            y += scores[static_cast<size_t>(j)] * static_cast<double>(Vbh[j * v_head_size + dv]);
          }
          Ybh[i * v_head_size + dv] = static_cast<float>(y);
        }
      }
    }
  }

  Attention::Result r;
  r.Y = std::move(Y);
  r.present_key = std::move(present_key);
  r.present_value = std::move(present_value);
  r.qk_matmul_output = std::move(qk_out);
  return r;
}

// Rank-3 Attention. ``Q``/``K``/``V`` use the fused
// ``(batch, seq, num_heads * head_size)`` layout. The inputs are promoted to
// the rank-4 layout, delegated to ``ComputeAttentionRank4`` and the primary
// output ``Y`` is collapsed back to rank-3. The present/qk auxiliary outputs
// remain rank-4, matching the upstream operator.
Attention::Result ComputeAttentionRank3(const Tensor &Q, const Tensor &K, const Tensor &V,
                                        const Attention::Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value,
                                        const Tensor *nonpad_kv_seqlen, RuntimeContext *rt) {
  Tensor Q4 = PromoteRank3(Q, attrs.q_num_heads, "Q", rt);
  Tensor K4 = PromoteRank3(K, attrs.kv_num_heads, "K", rt);
  Tensor V4 = PromoteRank3(V, attrs.kv_num_heads, "V", rt);
  Attention::Result r = ComputeAttentionRank4(Q4, K4, V4, attrs, attn_mask, past_key, past_value,
                                              nonpad_kv_seqlen, rt, /*temporary_y=*/true);
  r.Y = CollapseToRank3(r.Y, rt);
  return r;
}

} // namespace

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                             RuntimeContext *rt) const {
  CheckRank4Float(Q, "Q");
  const int64_t head_size = Q.shape[3];
  EXT_ENFORCE_INVALID(head_size > 0, "kernel::Attention: 'head_size' must be positive.");
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  return (*this)(Q, K, V, scale, rt);
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                             RuntimeContext *rt) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  return (*this)(Q, K, V, attrs, nullptr, nullptr, nullptr, nullptr, rt).Y;
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                             const Tensor &attn_mask, RuntimeContext *rt) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  const Tensor *const mask_ptr =
      attn_mask.shape.empty() && attn_mask.size_bytes() == 0 ? nullptr : &attn_mask;
  return (*this)(Q, K, V, attrs, mask_ptr, nullptr, nullptr, nullptr, rt).Y;
}

void Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                           const Tensor *attn_mask, Tensor &output) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  Result r = (*this)(Q, K, V, attrs, attn_mask);
  EXT_ENFORCE_INVALID(output.data_type == Q.data_type,
                      "kernel::Attention preallocated output must share Q's element type.");
  EXT_ENFORCE_INVALID(output.shape == r.Y.shape,
                      "kernel::Attention preallocated output shape must be (batch_size, "
                      "q_num_heads, q_seq_len, v_head_size).");
  EXT_ENFORCE_INVALID(output.size_bytes() == r.Y.size_bytes(),
                      "kernel::Attention preallocated output buffer has unexpected size in bytes.");
  std::memcpy(output.mutable_bytes(), r.Y.bytes(), r.Y.size_bytes());
}

Attention::Result Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                                        const Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value,
                                        const Tensor *nonpad_kv_seqlen, RuntimeContext *rt) const {
  // ----- Half-precision fast path ----------------------------------------
  // FLOAT16 / BFLOAT16 inputs are promoted to FLOAT32 here, the reference
  // implementation runs in float32, and the result tensors are demoted
  // back to the original element type. ``attn_mask`` may be FLOAT or BOOL
  // (or a matching half-precision dtype); BOOL masks are forwarded as-is.
  if (IsHalfPrecision(Q.data_type)) {
    EXT_ENFORCE_INVALID(K.data_type == Q.data_type && V.data_type == Q.data_type,
                        "kernel::Attention: Q, K, V must share the same dtype.");
    const int32_t target_dtype = Q.data_type;
    RuntimeContext scratch_rt(
        rt ? rt->kernel_ctx() : ctx_,
        RuntimeContextOptions{.allocator = rt ? rt->execution_allocator() : nullptr});
    RuntimeContext *compute_rt = rt ? &scratch_rt : nullptr;
    const Tensor Q_f = PromoteToFloat32(Q, compute_rt);
    const Tensor K_f = PromoteToFloat32(K, compute_rt);
    const Tensor V_f = PromoteToFloat32(V, compute_rt);
    Tensor attn_mask_f;
    const Tensor *attn_mask_ptr = attn_mask;
    if (attn_mask != nullptr && IsHalfPrecision(attn_mask->data_type)) {
      attn_mask_f = PromoteToFloat32(*attn_mask, compute_rt);
      attn_mask_ptr = &attn_mask_f;
    }
    Tensor past_key_f;
    const Tensor *past_key_ptr = past_key;
    if (past_key != nullptr && IsHalfPrecision(past_key->data_type)) {
      past_key_f = PromoteToFloat32(*past_key, compute_rt);
      past_key_ptr = &past_key_f;
    }
    Tensor past_value_f;
    const Tensor *past_value_ptr = past_value;
    if (past_value != nullptr && IsHalfPrecision(past_value->data_type)) {
      past_value_f = PromoteToFloat32(*past_value, compute_rt);
      past_value_ptr = &past_value_f;
    }
    Result r_f = (*this)(Q_f, K_f, V_f, attrs, attn_mask_ptr, past_key_ptr, past_value_ptr,
                         nonpad_kv_seqlen, compute_rt);
    Result r;
    constexpr int64_t kSerialDemotion = std::numeric_limits<int64_t>::max();
    auto demote_result = [&](const Tensor &value, int output_slot) {
      const bool has_recorded_roles = rt != nullptr && !rt->output_slot_io_roles().empty();
      const bool is_node_output = HasRecordedOutputSlot(rt, output_slot);
      RuntimeContext *destination_rt = !has_recorded_roles || is_node_output ? rt : compute_rt;
      const int destination_slot = is_node_output ? output_slot : 0;
      return DemoteFromFloat32(value, target_dtype, destination_rt, kSerialDemotion,
                               destination_slot);
    };
    r.Y = demote_result(r_f.Y, 0);
    r.present_key = demote_result(r_f.present_key, 1);
    r.present_value = demote_result(r_f.present_value, 2);
    r.qk_matmul_output = demote_result(r_f.qk_matmul_output, 3);
    return r;
  }

  // ----- Dispatch on rank -----------------------------------------------
  // Q/K/V may be provided either as rank-4 ``(batch, num_heads, seq,
  // head_size)`` tensors or as rank-3 ``(batch, seq, num_heads * head_size)``
  // packed tensors. They must all share the same rank; each case is handled
  // by its own dedicated implementation.
  EXT_ENFORCE_INVALID(Q.shape.size() == K.shape.size() && Q.shape.size() == V.shape.size(),
                      "kernel::Attention: Q, K, V must all share the same rank.");
  EXT_ENFORCE_INVALID(Q.shape.size() == 3 || Q.shape.size() == 4,
                      "kernel::Attention: Q, K, V must be rank-3 or rank-4.");
  if (Q.shape.size() == 3) {
    return ComputeAttentionRank3(Q, K, V, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                                 rt);
  }
  return ComputeAttentionRank4(Q, K, V, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                               rt);
}

void Attention::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 7), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 7 input(s), got ", node.input_size(),
                      ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 4), "RunNode: op '",
                      node.op_type(), "' expects between 1 and 4 output(s), got ",
                      node.output_size(), ".");
  const Tensor &q = GetInput(node, 0, rt.tensors());
  const Tensor &k = GetInput(node, 1, rt.tensors());
  const Tensor &v = GetInput(node, 2, rt.tensors());
  const Tensor *attn_mask = GetOptionalInput(node, 3, rt.tensors());
  const Tensor *past_key = GetOptionalInput(node, 4, rt.tensors());
  const Tensor *past_value = GetOptionalInput(node, 5, rt.tensors());
  const Tensor *nonpad_kv_seqlen = GetOptionalInput(node, 6, rt.tensors());

  onnx_kernels::kernel::Attention::Attributes attrs;
  if (FindAttribute(node, "scale") != nullptr) {
    attrs.has_scale = true;
    attrs.scale = GetAttributeFloatOrDefault(node, "scale", 0.0f);
  }
  attrs.is_causal = GetAttributeIntOrDefault(node, "is_causal", 0) != 0;
  attrs.softcap = GetAttributeFloatOrDefault(node, "softcap", 0.0f);
  attrs.qk_matmul_output_mode =
      static_cast<int>(GetAttributeIntOrDefault(node, "qk_matmul_output_mode", 0));
  attrs.q_num_heads = GetAttributeIntOrDefault(node, "q_num_heads", 0);
  attrs.kv_num_heads = GetAttributeIntOrDefault(node, "kv_num_heads", 0);

  onnx_kernels::kernel::Attention kernel(rt.kernel_ctx());
  onnx_kernels::kernel::Attention::Result result =
      kernel(q, k, v, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen, &rt);
  SetOutput(node, 0, std::move(result.Y), rt);

  auto set_optional_output = [&node, &rt](int index, Tensor output) {
    if (index >= node.output_size()) {
      return;
    }
    const std::string &name = node.output(index);
    if (name.empty()) {
      return;
    }
    output.name = name;
    rt.Put(name, std::move(output), RuntimeEventKind::kIntermediate);
  };
  set_optional_output(1, std::move(result.present_key));
  set_optional_output(2, std::move(result.present_value));
  set_optional_output(3, std::move(result.qk_matmul_output));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
