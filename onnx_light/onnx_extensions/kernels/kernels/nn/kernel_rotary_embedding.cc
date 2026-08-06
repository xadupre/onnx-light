// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/float16_promote.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Verifies that the cos/sin caches have the expected shapes given the
// input layout and the (possibly partial) rotation dimension. The dtype
// must equal ``x_dtype`` (the dtype of ``X``); the kernel supports FLOAT,
// FLOAT16, and BFLOAT16, but cos and sin must match ``X``.
void CheckCacheShape(const Tensor &cache, const char *which, int64_t batch, int64_t sequence_length,
                     int64_t rotary_dim_half, bool has_position_ids, int32_t x_dtype) {
  EXT_ENFORCE_INVALID(cache.data_type == x_dtype, "kernel::RotaryEmbedding: ", which,
                      " must have the same dtype as X.");
  if (has_position_ids) {
    EXT_ENFORCE_INVALID(cache.shape.size() == 2, "kernel::RotaryEmbedding: ", which,
                        " must be rank-2 when position_ids is provided.");
    EXT_ENFORCE_INVALID(cache.shape[1] == rotary_dim_half, "kernel::RotaryEmbedding: last dim of ",
                        which, " (", std::to_string(cache.shape[1]),
                        ") must equal rotary_embedding_dim/2 (", std::to_string(rotary_dim_half),
                        ").");
  } else {
    EXT_ENFORCE_INVALID(cache.shape.size() == 3, "kernel::RotaryEmbedding: ", which,
                        " must be rank-3 when position_ids is omitted.");
    EXT_ENFORCE_INVALID(cache.shape[0] == batch && cache.shape[1] == sequence_length &&
                            cache.shape[2] == rotary_dim_half,
                        "kernel::RotaryEmbedding: ", which,
                        " has unexpected shape; expected (batch, seq, rotary_dim/2).");
  }
  (void)batch;
  (void)sequence_length;
}

} // namespace

Tensor RotaryEmbedding::operator()(const Tensor &X, const Tensor &cos_cache,
                                   const Tensor &sin_cache, const Tensor &position_ids,
                                   const Attributes &attrs, RuntimeContext *rt) const {
  Tensor output;
  output.data_type = X.data_type;
  output.shape = X.shape;
  output.data.assign(X.size_bytes(), 0);
  const Tensor *pos =
      position_ids.shape.empty() && position_ids.size_bytes() == 0 && position_ids.data_type == 0
          ? nullptr
          : &position_ids;
  (*this)(X, cos_cache, sin_cache, pos, attrs, output, rt);
  return output;
}

void RotaryEmbedding::operator()(const Tensor &X, const Tensor &cos_cache, const Tensor &sin_cache,
                                 const Tensor *position_ids, const Attributes &attrs,
                                 Tensor &output, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(X.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          X.data_type == static_cast<int32_t>(DataType::FLOAT16) ||
                          X.data_type == static_cast<int32_t>(DataType::BFLOAT16),
                      "kernel::RotaryEmbedding: X must be FLOAT, FLOAT16 or BFLOAT16.");

  // Half-precision fast path: promote inputs to FLOAT32, compute, then
  // demote the result back into ``output``.
  if (IsHalfPrecision(X.data_type)) {
    EXT_ENFORCE_INVALID(output.data_type == X.data_type && output.shape == X.shape,
                        "kernel::RotaryEmbedding: output buffer has mismatched type or shape.");
    const int32_t target_dtype = X.data_type;
    const Tensor X_f = PromoteToFloat32(X);
    const Tensor cos_f = PromoteToFloat32(cos_cache);
    const Tensor sin_f = PromoteToFloat32(sin_cache);
    const size_t out_f_n_bytes = static_cast<size_t>(X.element_count()) * sizeof(float);
    RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
    Tensor out_f =
        MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), X.shape, out_f_n_bytes, allocator);
    (*this)(X_f, cos_f, sin_f, position_ids, attrs, out_f, rt);
    Tensor demoted = DemoteFromFloat32(out_f, target_dtype);
    EXT_ENFORCE_INVALID(output.size_bytes() == demoted.size_bytes(),
                        "kernel::RotaryEmbedding: output buffer has wrong byte size.");
    std::memcpy(output.mutable_bytes(), demoted.bytes(), demoted.size_bytes());
    return;
  }

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
  CheckCacheShape(cos_cache, "cos_cache", batch, sequence_length, rotary_dim_half, has_position_ids,
                  X.data_type);
  CheckCacheShape(sin_cache, "sin_cache", batch, sequence_length, rotary_dim_half, has_position_ids,
                  X.data_type);

  EXT_ENFORCE_INVALID(output.data_type == X.data_type && output.shape == X.shape,
                      "kernel::RotaryEmbedding: output buffer has mismatched type or shape.");
  const size_t expected_bytes = static_cast<size_t>(X.element_count()) * sizeof(float);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::RotaryEmbedding: output buffer has wrong byte size.");

  const float *px = X.AsFloat();
  float *py = reinterpret_cast<float *>(output.mutable_bytes());
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

void RotaryEmbedding::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 4), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 4 input(s), got ", node.input_size(),
                      ".");
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &cos_cache = GetInput(node, 1, rt.tensors());
  const Tensor &sin_cache = GetInput(node, 2, rt.tensors());
  const Tensor *position_ids = GetOptionalInput(node, 3, rt.tensors());

  onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
  attrs.interleaved = GetAttributeIntOrDefault(node, "interleaved", 0) != 0;
  attrs.rotary_embedding_dim = GetAttributeIntOrDefault(node, "rotary_embedding_dim", 0);
  attrs.num_heads = GetAttributeIntOrDefault(node, "num_heads", 0);

  onnx_kernels::kernel::RotaryEmbedding kernel(rt.kernel_ctx());
  const Tensor empty;
  const Tensor &pos = (position_ids != nullptr) ? *position_ids : empty;
  SetOutput(node, 0, kernel(x, cos_cache, sin_cache, pos, attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
