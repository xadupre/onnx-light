// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Verifies that the cos/sin caches have the expected shapes given the
// input layout and the (possibly partial) rotation dimension.
void CheckCacheShape(const Tensor &cache, const char *which, int64_t batch, int64_t sequence_length,
                     int64_t rotary_dim_half, bool has_position_ids) {
  EXT_ENFORCE_INVALID(cache.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::RotaryEmbedding: ") + which + " must be FLOAT.");
  if (has_position_ids) {
    EXT_ENFORCE_INVALID(cache.shape.size() == 2,
                        std::string("kernel::RotaryEmbedding: ") + which +
                            " must be rank-2 when position_ids is provided.");
    EXT_ENFORCE_INVALID(cache.shape[1] == rotary_dim_half,
                        std::string("kernel::RotaryEmbedding: last dim of ") + which + " (" +
                            std::to_string(cache.shape[1]) +
                            ") must equal rotary_embedding_dim/2 (" +
                            std::to_string(rotary_dim_half) + ").");
  } else {
    EXT_ENFORCE_INVALID(cache.shape.size() == 3,
                        std::string("kernel::RotaryEmbedding: ") + which +
                            " must be rank-3 when position_ids is omitted.");
    EXT_ENFORCE_INVALID(cache.shape[0] == batch && cache.shape[1] == sequence_length &&
                            cache.shape[2] == rotary_dim_half,
                        std::string("kernel::RotaryEmbedding: ") + which +
                            " has unexpected shape; expected (batch, seq, rotary_dim/2).");
  }
  (void)batch;
  (void)sequence_length;
}

} // namespace

Tensor RotaryEmbedding::operator()(const Tensor &X, const Tensor &cos_cache,
                                   const Tensor &sin_cache, const Tensor &position_ids,
                                   const Attributes &attrs) const {
  Tensor output;
  output.data_type = X.data_type;
  output.shape = X.shape;
  output.data.assign(X.data.size(), 0);
  const Tensor *pos =
      position_ids.shape.empty() && position_ids.size_bytes() == 0 && position_ids.data_type == 0
          ? nullptr
          : &position_ids;
  (*this)(X, cos_cache, sin_cache, pos, attrs, output);
  return output;
}

void RotaryEmbedding::operator()(const Tensor &X, const Tensor &cos_cache, const Tensor &sin_cache,
                                 const Tensor *position_ids, const Attributes &attrs,
                                 Tensor &output) const {
  EXT_ENFORCE_INVALID(X.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RotaryEmbedding: X must be FLOAT.");
  EXT_ENFORCE_INVALID(X.shape.size() == 3 || X.shape.size() == 4,
                      "kernel::RotaryEmbedding: X must have rank 3 or 4.");

  const bool original_is_3d = X.shape.size() == 3;
  int64_t batch = X.shape[0];
  int64_t sequence_length = 0;
  int64_t num_heads = 0;
  int64_t head_size = 0;

  if (original_is_3d) {
    // (batch, sequence_length, hidden_size) -> infer head_size.
    sequence_length = X.shape[1];
    EXT_ENFORCE_INVALID(attrs.num_heads > 0,
                        "kernel::RotaryEmbedding: num_heads attribute is required for rank-3 X.");
    num_heads = attrs.num_heads;
    const int64_t hidden_size = X.shape[2];
    EXT_ENFORCE_INVALID(hidden_size % num_heads == 0,
                        "kernel::RotaryEmbedding: hidden_size must be divisible by num_heads.");
    head_size = hidden_size / num_heads;
  } else {
    // (batch, num_heads, sequence_length, head_size).
    num_heads = X.shape[1];
    sequence_length = X.shape[2];
    head_size = X.shape[3];
  }
  EXT_ENFORCE_INVALID(head_size % 2 == 0, "kernel::RotaryEmbedding: head_size must be even.");

  int64_t rotary_dim = attrs.rotary_embedding_dim;
  if (rotary_dim == 0) {
    rotary_dim = head_size;
  }
  EXT_ENFORCE_INVALID(
      rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= head_size,
      "kernel::RotaryEmbedding: rotary_embedding_dim must be even and <= head_size.");
  const int64_t rotary_dim_half = rotary_dim / 2;

  const bool has_position_ids = position_ids != nullptr && !position_ids->shape.empty();
  if (has_position_ids) {
    EXT_ENFORCE_INVALID(position_ids->data_type == static_cast<int32_t>(DataType::INT64),
                        "kernel::RotaryEmbedding: position_ids must be INT64.");
    EXT_ENFORCE_INVALID(position_ids->shape.size() == 2 && position_ids->shape[0] == batch &&
                            position_ids->shape[1] == sequence_length,
                        "kernel::RotaryEmbedding: position_ids must have shape (batch, seq).");
  }
  CheckCacheShape(cos_cache, "cos_cache", batch, sequence_length, rotary_dim_half,
                  has_position_ids);
  CheckCacheShape(sin_cache, "sin_cache", batch, sequence_length, rotary_dim_half,
                  has_position_ids);

  EXT_ENFORCE_INVALID(output.data_type == X.data_type && output.shape == X.shape,
                      "kernel::RotaryEmbedding: output buffer has mismatched type or shape.");
  const size_t expected_bytes = static_cast<size_t>(X.element_count()) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::RotaryEmbedding: output buffer has wrong byte size.");

  const float *px = X.AsFloat();
  float *py = reinterpret_cast<float *>(output.data.data());
  const float *pcos = cos_cache.AsFloat();
  const float *psin = sin_cache.AsFloat();
  const int64_t *ppos = has_position_ids ? position_ids->AsInt64() : nullptr;

  // Strides for the canonical (batch, num_heads, sequence_length, head_size)
  // layout. When original_is_3d the actual storage layout is
  // (batch, sequence_length, num_heads, head_size); we account for this with
  // a permuted index computation rather than copying the buffer.
  const int64_t cache_max_pos = has_position_ids ? cos_cache.shape[0] : 0;

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < num_heads; ++h) {
      for (int64_t s = 0; s < sequence_length; ++s) {
        // Source/destination base index for (b, h, s, :) in the actual
        // storage layout.
        int64_t base;
        if (original_is_3d) {
          // Storage: (batch, seq, hidden_size = num_heads * head_size).
          base = ((b * sequence_length + s) * num_heads + h) * head_size;
        } else {
          base = ((b * num_heads + h) * sequence_length + s) * head_size;
        }

        // Locate the cos/sin row for (b, s).
        const float *cos_row;
        const float *sin_row;
        if (has_position_ids) {
          const int64_t pos = ppos[b * sequence_length + s];
          EXT_ENFORCE_INVALID(pos >= 0 && pos < cache_max_pos,
                              "kernel::RotaryEmbedding: position_ids value out of range.");
          cos_row = pcos + pos * rotary_dim_half;
          sin_row = psin + pos * rotary_dim_half;
        } else {
          cos_row = pcos + (b * sequence_length + s) * rotary_dim_half;
          sin_row = psin + (b * sequence_length + s) * rotary_dim_half;
        }

        // Rotate the first ``rotary_dim`` channels.
        for (int64_t i = 0; i < rotary_dim_half; ++i) {
          float x1;
          float x2;
          int64_t idx1;
          int64_t idx2;
          if (attrs.interleaved) {
            idx1 = base + 2 * i;
            idx2 = base + 2 * i + 1;
          } else {
            idx1 = base + i;
            idx2 = base + rotary_dim_half + i;
          }
          x1 = px[idx1];
          x2 = px[idx2];
          const float c = cos_row[i];
          const float s_v = sin_row[i];
          py[idx1] = c * x1 - s_v * x2;
          py[idx2] = s_v * x1 + c * x2;
        }
        // Pass through the remaining ``head_size - rotary_dim`` channels.
        for (int64_t i = rotary_dim; i < head_size; ++i) {
          py[base + i] = px[base + i];
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
