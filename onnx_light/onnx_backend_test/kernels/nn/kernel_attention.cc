// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Validates that ``t`` is a rank-4 FLOAT tensor and returns its shape as
// ``(B, H, L, D)``. The caller is identified by ``label`` for clearer error
// messages.
void CheckRank4Float(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == TensorProto::DataType::FLOAT,
                      std::string("kernel::Attention: '") + label + "' must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4,
                      std::string("kernel::Attention: '") + label + "' must be a rank-4 tensor.");
  for (int64_t d : t.shape) {
    EXT_ENFORCE_INVALID(d >= 0, std::string("kernel::Attention: '") + label +
                                    "' has a negative dimension.");
  }
}

// Returns the value of ``attn_mask`` at the broadcasted index
// ``(b, h, i, j)``. ``mask`` is assumed FLOAT, with up to rank 4 and each
// dimension equal to either 1 (broadcast) or the corresponding output
// dimension.
double BroadcastedMaskValue(const Tensor &mask, int64_t batch_size, int64_t q_num_heads,
                            int64_t q_seq_len, int64_t kv_seq_len, int64_t b, int64_t h, int64_t i,
                            int64_t j) {
  const int rank = static_cast<int>(mask.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1 && rank <= 4,
                      "kernel::Attention: 'attn_mask' must have rank between 1 and 4.");
  // Promote to rank-4 by prepending 1's on the leading axes.
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
  return static_cast<double>(mask.AsFloat()[linear]);
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
  CheckRank4Float(Q, "Q");
  CheckRank4Float(V, "V");
  const int64_t batch_size = Q.shape[0];
  const int64_t q_num_heads = Q.shape[1];
  const int64_t q_seq_len = Q.shape[2];
  const int64_t v_head_size = V.shape[3];
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  Tensor out("", TensorProto::DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, v_head_size},
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float)));
  (*this)(Q, K, V, scale, /*attn_mask=*/nullptr, out);
  return out;
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                             const Tensor &attn_mask) const {
  CheckRank4Float(Q, "Q");
  CheckRank4Float(V, "V");
  const int64_t batch_size = Q.shape[0];
  const int64_t q_num_heads = Q.shape[1];
  const int64_t q_seq_len = Q.shape[2];
  const int64_t v_head_size = V.shape[3];
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  Tensor out("", TensorProto::DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, v_head_size},
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float)));
  const Tensor *const mask_ptr =
      attn_mask.shape.empty() && attn_mask.data.empty() ? nullptr : &attn_mask;
  (*this)(Q, K, V, scale, mask_ptr, out);
  return out;
}

void Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                           const Tensor *attn_mask, Tensor &output) const {
  CheckRank4Float(Q, "Q");
  CheckRank4Float(K, "K");
  CheckRank4Float(V, "V");

  const int64_t batch_size = Q.shape[0];
  const int64_t q_num_heads = Q.shape[1];
  const int64_t q_seq_len = Q.shape[2];
  const int64_t head_size = Q.shape[3];

  const int64_t k_batch = K.shape[0];
  const int64_t kv_num_heads = K.shape[1];
  const int64_t kv_seq_len = K.shape[2];
  const int64_t k_head_size = K.shape[3];

  const int64_t v_batch = V.shape[0];
  const int64_t v_num_heads = V.shape[1];
  const int64_t v_seq_len = V.shape[2];
  const int64_t v_head_size = V.shape[3];

  EXT_ENFORCE_INVALID(batch_size == k_batch && batch_size == v_batch,
                      "kernel::Attention: 'Q', 'K', 'V' must share the same batch size.");
  EXT_ENFORCE_INVALID(k_head_size == head_size,
                      "kernel::Attention: 'K' head_size must match 'Q' head_size.");
  EXT_ENFORCE_INVALID(v_num_heads == kv_num_heads,
                      "kernel::Attention: 'V' num_heads must match 'K' num_heads.");
  EXT_ENFORCE_INVALID(v_seq_len == kv_seq_len,
                      "kernel::Attention: 'V' kv_seq_len must match 'K' kv_seq_len.");
  EXT_ENFORCE_INVALID(
      kv_num_heads > 0 && q_num_heads % kv_num_heads == 0,
      "kernel::Attention: 'q_num_heads' must be a positive multiple of 'kv_num_heads'.");

  EXT_ENFORCE_INVALID(output.data_type == TensorProto::DataType::FLOAT,
                      "kernel::Attention preallocated output must be a FLOAT tensor.");
  const std::vector<int64_t> expected_out_shape = {batch_size, q_num_heads, q_seq_len, v_head_size};
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::Attention preallocated output shape must be (batch_size, "
                      "q_num_heads, q_seq_len, v_head_size).");
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
                      "kernel::Attention preallocated output buffer has unexpected size in bytes.");

  if (attn_mask != nullptr) {
    EXT_ENFORCE_INVALID(attn_mask->data_type == TensorProto::DataType::FLOAT,
                        "kernel::Attention: 'attn_mask' must be a FLOAT tensor in this reference "
                        "kernel (bool/integer masks are not supported).");
  }

  const int64_t group_size = q_num_heads / kv_num_heads;
  const float *pQ = Q.AsFloat();
  const float *pK = K.AsFloat();
  const float *pV = V.AsFloat();
  float *pY = output.AsFloat();

  // Strides (row-major):
  // Q: (q_num_heads * q_seq_len * head_size, q_seq_len * head_size, head_size, 1)
  // K: (kv_num_heads * kv_seq_len * head_size, kv_seq_len * head_size, head_size, 1)
  // V: (kv_num_heads * kv_seq_len * v_head_size, kv_seq_len * v_head_size, v_head_size, 1)
  // Y: (q_num_heads * q_seq_len * v_head_size, q_seq_len * v_head_size, v_head_size, 1)
  const int64_t q_head_stride = q_seq_len * head_size;
  const int64_t q_batch_stride = q_num_heads * q_head_stride;
  const int64_t k_head_stride = kv_seq_len * head_size;
  const int64_t k_batch_stride = kv_num_heads * k_head_stride;
  const int64_t v_head_stride = kv_seq_len * v_head_size;
  const int64_t v_batch_stride = kv_num_heads * v_head_stride;
  const int64_t y_head_stride = q_seq_len * v_head_size;
  const int64_t y_batch_stride = q_num_heads * y_head_stride;

  std::vector<double> scores(static_cast<size_t>(kv_seq_len));
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < q_num_heads; ++h) {
      const int64_t kv_h = h / group_size;
      const float *Qbh = pQ + b * q_batch_stride + h * q_head_stride;
      const float *Kbh = pK + b * k_batch_stride + kv_h * k_head_stride;
      const float *Vbh = pV + b * v_batch_stride + kv_h * v_head_stride;
      float *Ybh = pY + b * y_batch_stride + h * y_head_stride;

      for (int64_t i = 0; i < q_seq_len; ++i) {
        double max_score = 0.0;
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          double s = 0.0;
          for (int64_t d = 0; d < head_size; ++d) {
            s += static_cast<double>(Qbh[i * head_size + d]) *
                 static_cast<double>(Kbh[j * head_size + d]);
          }
          s *= static_cast<double>(scale);
          if (attn_mask != nullptr) {
            s += BroadcastedMaskValue(*attn_mask, batch_size, q_num_heads, q_seq_len, kv_seq_len, b,
                                      h, i, j);
          }
          scores[static_cast<size_t>(j)] = s;
          if (j == 0 || s > max_score) {
            max_score = s;
          }
        }
        // softmax over the last axis (kv_seq_len).
        double denom = 0.0;
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          const double e = std::exp(scores[static_cast<size_t>(j)] - max_score);
          scores[static_cast<size_t>(j)] = e;
          denom += e;
        }
        const double inv_denom = denom != 0.0 ? 1.0 / denom : 0.0;
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          scores[static_cast<size_t>(j)] *= inv_denom;
        }
        // Y[i, dv] = sum_j probs[j] * V[j, dv]
        for (int64_t dv = 0; dv < v_head_size; ++dv) {
          double y = 0.0;
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            y += scores[static_cast<size_t>(j)] * static_cast<double>(Vbh[j * v_head_size + dv]);
          }
          Ybh[i * v_head_size + dv] = static_cast<float>(y);
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
