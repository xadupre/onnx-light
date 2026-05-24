// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/preview/include_preview_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Validates that ``t`` is a rank-4 FLOAT tensor and returns its shape as
// ``(B, H, L, D)``. The caller is identified by ``label`` for clearer error
// messages.
void CheckRank4Float(const Tensor &t, const char *label) {
  if (t.data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument(std::string("kernel::FlexAttention: '") + label +
                                "' must be a FLOAT tensor.");
  }
  if (t.shape.size() != 4) {
    throw std::invalid_argument(std::string("kernel::FlexAttention: '") + label +
                                "' must be a rank-4 tensor.");
  }
  for (int64_t d : t.shape) {
    if (d < 0) {
      throw std::invalid_argument(std::string("kernel::FlexAttention: '") + label +
                                  "' has a negative dimension.");
    }
  }
}

} // namespace

Tensor FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V) const {
  CheckRank4Float(Q, "Q");
  const int64_t head_size = Q.shape[3];
  if (head_size <= 0) {
    throw std::invalid_argument("kernel::FlexAttention: 'head_size' must be positive.");
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  return (*this)(Q, K, V, scale);
}

Tensor FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                                 float scale) const {
  CheckRank4Float(Q, "Q");
  CheckRank4Float(V, "V");
  const int64_t batch_size = Q.shape[0];
  const int64_t q_num_heads = Q.shape[1];
  const int64_t q_seq_len = Q.shape[2];
  const int64_t v_head_size = V.shape[3];
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  Tensor out("", TensorProto::DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, v_head_size},
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float)));
  (*this)(Q, K, V, scale, out);
  return out;
}

void FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                               Tensor &output) const {
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

  if (batch_size != k_batch || batch_size != v_batch) {
    throw std::invalid_argument(
        "kernel::FlexAttention: 'Q', 'K', 'V' must share the same batch size.");
  }
  if (k_head_size != head_size) {
    throw std::invalid_argument("kernel::FlexAttention: 'K' head_size must match 'Q' head_size.");
  }
  if (v_num_heads != kv_num_heads) {
    throw std::invalid_argument("kernel::FlexAttention: 'V' num_heads must match 'K' num_heads.");
  }
  if (v_seq_len != kv_seq_len) {
    throw std::invalid_argument("kernel::FlexAttention: 'V' kv_seq_len must match 'K' kv_seq_len.");
  }
  if (kv_num_heads <= 0 || q_num_heads % kv_num_heads != 0) {
    throw std::invalid_argument(
        "kernel::FlexAttention: 'q_num_heads' must be a positive multiple of 'kv_num_heads'.");
  }

  if (output.data_type != TensorProto::DataType::FLOAT) {
    throw std::invalid_argument(
        "kernel::FlexAttention preallocated output must be a FLOAT tensor.");
  }
  const std::vector<int64_t> expected_out_shape = {batch_size, q_num_heads, q_seq_len, v_head_size};
  if (output.shape != expected_out_shape) {
    throw std::invalid_argument(
        "kernel::FlexAttention preallocated output shape must be (batch_size, q_num_heads, "
        "q_seq_len, v_head_size).");
  }
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  if (output.data.size() != static_cast<size_t>(out_count) * sizeof(float)) {
    throw std::invalid_argument(
        "kernel::FlexAttention preallocated output buffer has unexpected size in bytes.");
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
        // scores[j] = sum_d Q[i, d] * K[j, d] * scale
        double max_score = 0.0;
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          double s = 0.0;
          for (int64_t d = 0; d < head_size; ++d) {
            s += static_cast<double>(Qbh[i * head_size + d]) *
                 static_cast<double>(Kbh[j * head_size + d]);
          }
          s *= static_cast<double>(scale);
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
