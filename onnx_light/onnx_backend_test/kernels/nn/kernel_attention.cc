// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Validates that ``t`` is a rank-4 FLOAT tensor. The caller is identified by
// ``label`` for clearer error messages.
void CheckRank4Float(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT,
                      std::string("kernel::Attention: '") + label + "' must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4,
                      std::string("kernel::Attention: '") + label + "' must be a rank-4 tensor.");
  for (int64_t d : t.shape) {
    EXT_ENFORCE_INVALID(d >= 0, std::string("kernel::Attention: '") + label +
                                    "' has a negative dimension.");
  }
}

// Promotes a rank-3 fused ``(batch, seq, num_heads * head_size)`` FLOAT tensor
// into the rank-4 layout ``(batch, num_heads, seq, head_size)`` used by the
// internal kernel. Returns the promoted tensor; ``num_heads`` must divide
// ``hidden_size``.
Tensor PromoteRank3(const Tensor &t, int64_t num_heads, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT,
                      std::string("kernel::Attention: '") + label + "' must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 3,
                      std::string("kernel::Attention: '") + label + "' must be rank-3.");
  const int64_t batch = t.shape[0];
  const int64_t seq = t.shape[1];
  const int64_t hidden = t.shape[2];
  EXT_ENFORCE_INVALID(num_heads > 0, std::string("kernel::Attention: '") + label +
                                         "' needs a positive ``num_heads`` to promote rank-3.");
  EXT_ENFORCE_INVALID(hidden % num_heads == 0,
                      std::string("kernel::Attention: '") + label +
                          "' hidden_size must be a multiple of ``num_heads``.");
  const int64_t head_size = hidden / num_heads;
  Tensor out("", DataType::FLOAT, {batch, num_heads, seq, head_size},
             std::vector<uint8_t>(t.data.size()));
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
Tensor CollapseToRank3(const Tensor &t) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, "kernel::Attention: must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4, "kernel::Attention: must be rank-4.");
  const int64_t batch = t.shape[0];
  const int64_t num_heads = t.shape[1];
  const int64_t seq = t.shape[2];
  const int64_t head_size = t.shape[3];
  const int64_t hidden = num_heads * head_size;
  Tensor out("", DataType::FLOAT, {batch, seq, hidden}, std::vector<uint8_t>(t.data.size()));
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
Tensor ConcatAxis2(const Tensor &a, const Tensor &b) {
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
  Tensor out("", DataType::FLOAT, {batch, heads, lc, d},
             std::vector<uint8_t>(static_cast<size_t>(batch * heads * lc * d) * sizeof(float)));
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
                       int64_t q_seq_len, int64_t kv_seq_len, int64_t b, int64_t h, int64_t i,
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

} // namespace

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V) const {
  CheckRank4Float(Q, "Q");
  const int64_t head_size = Q.shape[3];
  EXT_ENFORCE_INVALID(head_size > 0, "kernel::Attention: 'head_size' must be positive.");
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  return (*this)(Q, K, V, scale);
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  return (*this)(Q, K, V, attrs).Y;
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                             const Tensor &attn_mask) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  const Tensor *const mask_ptr =
      attn_mask.shape.empty() && attn_mask.data.empty() ? nullptr : &attn_mask;
  return (*this)(Q, K, V, attrs, mask_ptr).Y;
}

void Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                           const Tensor *attn_mask, Tensor &output) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  Result r = (*this)(Q, K, V, attrs, attn_mask);
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::Attention preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == r.Y.shape,
                      "kernel::Attention preallocated output shape must be (batch_size, "
                      "q_num_heads, q_seq_len, v_head_size).");
  EXT_ENFORCE_INVALID(output.data.size() == r.Y.data.size(),
                      "kernel::Attention preallocated output buffer has unexpected size in bytes.");
  std::copy(r.Y.data.begin(), r.Y.data.end(), output.data.begin());
}

Attention::Result Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                                        const Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value) const {
  // ----- Normalize Q/K/V to rank-4 ---------------------------------------
  EXT_ENFORCE_INVALID(Q.shape.size() == K.shape.size() && Q.shape.size() == V.shape.size(),
                      "kernel::Attention: Q, K, V must all share the same rank.");
  EXT_ENFORCE_INVALID(Q.shape.size() == 3 || Q.shape.size() == 4,
                      "kernel::Attention: Q, K, V must be rank-3 or rank-4.");
  const bool rank3 = Q.shape.size() == 3;

  Tensor Q4 = rank3 ? PromoteRank3(Q, attrs.q_num_heads, "Q") : Q;
  Tensor K4 = rank3 ? PromoteRank3(K, attrs.kv_num_heads, "K") : K;
  Tensor V4 = rank3 ? PromoteRank3(V, attrs.kv_num_heads, "V") : V;
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
  Tensor present_key = past_key != nullptr ? ConcatAxis2(*past_key, K4) : K4;
  Tensor present_value = past_value != nullptr ? ConcatAxis2(*past_value, V4) : V4;
  const int64_t total_kv_seq_len = present_key.shape[2];
  const int64_t past_kv_seq_len = past_key != nullptr ? past_key->shape[2] : 0;
  const int64_t v_head_size = present_value.shape[3];

  // ----- Resolve scale ---------------------------------------------------
  const float scale =
      attrs.has_scale ? attrs.scale : 1.0f / std::sqrt(static_cast<float>(head_size));

  // ----- Allocate outputs ------------------------------------------------
  const int64_t out_count_y = batch_size * q_num_heads * q_seq_len * v_head_size;
  Tensor Y("", DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, v_head_size},
           std::vector<uint8_t>(static_cast<size_t>(out_count_y) * sizeof(float)));
  const int64_t qk_count = batch_size * q_num_heads * q_seq_len * total_kv_seq_len;
  Tensor qk_out("", DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, total_kv_seq_len},
                std::vector<uint8_t>(static_cast<size_t>(qk_count) * sizeof(float)));

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
  std::vector<double> scores(static_cast<size_t>(total_kv_seq_len));
  std::vector<double> bias(static_cast<size_t>(total_kv_seq_len));
  std::vector<double> qkraw(static_cast<size_t>(total_kv_seq_len));
  for (int64_t b = 0; b < batch_size; ++b) {
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
            // Upper-triangular causal mask anchored at the upper-left of
            // the new-token sub-block ``[:, past_kv_seq_len:]``.
            if (j >= past_kv_seq_len && (j - past_kv_seq_len) > i) {
              b_val = -std::numeric_limits<double>::infinity();
            }
          }
          bias[static_cast<size_t>(j)] = b_val;
        }
        // Build pre-softcap scores = raw + bias.
        double max_score = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < total_kv_seq_len; ++j) {
          double s = qkraw[static_cast<size_t>(j)] + bias[static_cast<size_t>(j)];
          scores[static_cast<size_t>(j)] = s;
          if (s > max_score) {
            max_score = s;
          }
        }
        // qk_matmul_output_mode 0: raw; 1: with bias (pre-softcap).
        if (attrs.qk_matmul_output_mode == 0) {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            QKbh[i * total_kv_seq_len + j] = static_cast<float>(qkraw[static_cast<size_t>(j)]);
          }
        } else if (attrs.qk_matmul_output_mode == 1) {
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            QKbh[i * total_kv_seq_len + j] = static_cast<float>(scores[static_cast<size_t>(j)]);
          }
        }
        // Apply softcap if requested.
        if (attrs.softcap > 0.0f) {
          const double sc = static_cast<double>(attrs.softcap);
          max_score = -std::numeric_limits<double>::infinity();
          for (int64_t j = 0; j < total_kv_seq_len; ++j) {
            // ``sc * tanh(s / sc)`` saturates large magnitudes to ``±sc``;
            // for ``s == -inf`` the limit is ``-sc`` (a finite value),
            // matching upstream's ``np.tanh`` behaviour exactly.
            const double s = sc * std::tanh(scores[static_cast<size_t>(j)] / sc);
            scores[static_cast<size_t>(j)] = s;
            if (s > max_score) {
              max_score = s;
            }
          }
        }
        // qk_matmul_output_mode 2: after softcap.
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
        // qk_matmul_output_mode 3: after softmax.
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

  Result r;
  r.Y = rank3 ? CollapseToRank3(Y) : std::move(Y);
  r.present_key = std::move(present_key);
  r.present_value = std::move(present_value);
  r.qk_matmul_output = std::move(qk_out);
  return r;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
