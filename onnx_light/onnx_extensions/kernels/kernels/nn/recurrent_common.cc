// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/recurrent_common.h"

#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {
namespace recurrent {

int64_t RecurrentNumDirections(const char *op, const std::string &direction) {
  if (direction == "forward" || direction == "reverse") {
    return 1;
  }
  if (direction == "bidirectional") {
    return 2;
  }
  EXT_ENFORCE_INVALID("kernel::", op, ": direction must be 'forward', 'reverse' or 'bidirectional'",
                      ", got '", direction, "'.");
  return 1; // unreachable
}

Tensor RecurrentTransposeInitialState(const Tensor &state, int64_t num_directions,
                                      RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(state.shape.size() == 3u && state.shape[1] == num_directions,
                      "kernel: initial state must have shape [batch_size, num_directions, "
                      "hidden_size] for layout=1.");
  const int64_t batch_size = state.shape[0];
  const int64_t hidden_size = state.shape[2];
  const size_t n_bytes =
      static_cast<size_t>(num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT),
                                onnx_kernels::Shape{num_directions, batch_size, hidden_size},
                                n_bytes, allocator);
  float *dst = out.AsFloat();
  const float *src = state.AsFloat();
  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t d = 0; d < num_directions; ++d) {
      for (int64_t h = 0; h < hidden_size; ++h) {
        dst[static_cast<size_t>((d * batch_size + n) * hidden_size + h)] =
            src[static_cast<size_t>((n * num_directions + d) * hidden_size + h)];
      }
    }
  }
  return out;
}

Tensor RecurrentPermuteYLayout1(const Tensor &y, int64_t seq_length, int64_t num_directions,
                                int64_t batch_size, int64_t hidden_size,
                                RawBufferAllocator *allocator) {
  const size_t n_bytes =
      static_cast<size_t>(seq_length * num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor out = MakeOutputTensor(
      static_cast<int32_t>(DataType::FLOAT),
      onnx_kernels::Shape{batch_size, seq_length, num_directions, hidden_size}, n_bytes, allocator);
  float *dst = out.AsFloat();
  const float *src = y.AsFloat();
  for (int64_t s = 0; s < seq_length; ++s) {
    for (int64_t d = 0; d < num_directions; ++d) {
      for (int64_t n = 0; n < batch_size; ++n) {
        for (int64_t h = 0; h < hidden_size; ++h) {
          const size_t src_idx =
              static_cast<size_t>(((s * num_directions + d) * batch_size + n) * hidden_size + h);
          const size_t dst_idx =
              static_cast<size_t>(((n * seq_length + s) * num_directions + d) * hidden_size + h);
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
  return out;
}

Tensor RecurrentPermuteStateLayout1(const Tensor &state, int64_t num_directions,
                                    int64_t batch_size, int64_t hidden_size,
                                    RawBufferAllocator *allocator) {
  const size_t n_bytes =
      static_cast<size_t>(num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT),
                                onnx_kernels::Shape{batch_size, num_directions, hidden_size},
                                n_bytes, allocator);
  float *dst = out.AsFloat();
  const float *src = state.AsFloat();
  for (int64_t d = 0; d < num_directions; ++d) {
    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t h = 0; h < hidden_size; ++h) {
        dst[static_cast<size_t>((n * num_directions + d) * hidden_size + h)] =
            src[static_cast<size_t>((d * batch_size + n) * hidden_size + h)];
      }
    }
  }
  return out;
}

} // namespace recurrent
} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
