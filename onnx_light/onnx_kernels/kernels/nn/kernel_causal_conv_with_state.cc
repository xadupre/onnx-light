// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

std::pair<Tensor, Tensor> CausalConvWithState::operator()(const Tensor &input, const Tensor &weight,
                                                          const Tensor &bias,
                                                          const Tensor &past_state,
                                                          const Attributes &attrs) const {
  Tensor output;
  output.data_type = input.data_type;
  output.shape = input.shape;
  output.data.assign(input.data.size(), 0);

  // present_state shape = (B, C, K - 1).
  const int64_t K = weight.shape.size() >= 3 ? weight.shape[2] : 0;
  const int64_t Km1 = K > 0 ? K - 1 : 0;
  Tensor present_state;
  present_state.data_type = input.data_type;
  present_state.shape = {input.shape.empty() ? 0 : input.shape[0],
                         input.shape.size() > 1 ? input.shape[1] : 0, Km1};
  present_state.data.assign(static_cast<size_t>(present_state.shape[0]) *
                                static_cast<size_t>(present_state.shape[1]) *
                                static_cast<size_t>(Km1) * sizeof(float),
                            0);

  const Tensor *bias_ptr =
      bias.shape.empty() && bias.size_bytes() == 0 && bias.data_type == 0 ? nullptr : &bias;
  const Tensor *past_ptr =
      past_state.shape.empty() && past_state.size_bytes() == 0 && past_state.data_type == 0
          ? nullptr
          : &past_state;
  (*this)(input, weight, bias_ptr, past_ptr, attrs, output, present_state);
  return {std::move(output), std::move(present_state)};
}

void CausalConvWithState::operator()(const Tensor &input, const Tensor &weight, const Tensor *bias,
                                     const Tensor *past_state, const Attributes &attrs,
                                     Tensor &output, Tensor &present_state) const {
  EXT_ENFORCE_INVALID(input.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::CausalConvWithState: input must be FLOAT.");
  EXT_ENFORCE_INVALID(weight.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::CausalConvWithState: weight must be FLOAT.");
  EXT_ENFORCE_INVALID(input.shape.size() == 3,
                      "kernel::CausalConvWithState: input must be rank-3 (B, C, L).");
  EXT_ENFORCE_INVALID(weight.shape.size() == 3 && weight.shape[1] == 1,
                      "kernel::CausalConvWithState: weight must be rank-3 (C, 1, K).");
  EXT_ENFORCE_INVALID(weight.shape[0] == input.shape[1],
                      "kernel::CausalConvWithState: weight C must equal input C.");

  const int64_t B = input.shape[0];
  const int64_t C = input.shape[1];
  const int64_t L = input.shape[2];
  const int64_t K = weight.shape[2];
  EXT_ENFORCE_INVALID(K >= 1, "kernel::CausalConvWithState: kernel size K must be >= 1.");
  const int64_t Km1 = K - 1;

  if (bias != nullptr) {
    EXT_ENFORCE_INVALID(bias->data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::CausalConvWithState: bias must be FLOAT.");
    EXT_ENFORCE_INVALID(bias->shape.size() == 1 && bias->shape[0] == C,
                        "kernel::CausalConvWithState: bias must have shape (C).");
  }
  if (past_state != nullptr) {
    EXT_ENFORCE_INVALID(past_state->data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::CausalConvWithState: past_state must be FLOAT.");
    EXT_ENFORCE_INVALID(past_state->shape.size() == 3 && past_state->shape[0] == B &&
                            past_state->shape[1] == C && past_state->shape[2] == Km1,
                        "kernel::CausalConvWithState: past_state must have shape (B, C, K-1).");
  }

  EXT_ENFORCE_INVALID(output.data_type == input.data_type && output.shape == input.shape,
                      "kernel::CausalConvWithState: output buffer has mismatched type or shape.");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(B * C * L) * sizeof(float),
                      "kernel::CausalConvWithState: output buffer has wrong byte size.");
  EXT_ENFORCE_INVALID(present_state.data_type == input.data_type &&
                          present_state.shape.size() == 3 && present_state.shape[0] == B &&
                          present_state.shape[1] == C && present_state.shape[2] == Km1,
                      "kernel::CausalConvWithState: present_state has mismatched type or shape.");
  EXT_ENFORCE_INVALID(present_state.data.size() == static_cast<size_t>(B * C * Km1) * sizeof(float),
                      "kernel::CausalConvWithState: present_state buffer has wrong byte size.");

  const bool use_silu = attrs.activation == "silu" || attrs.activation == "swish";
  EXT_ENFORCE_INVALID(use_silu || attrs.activation == "none",
                      "kernel::CausalConvWithState: unsupported activation '" + attrs.activation +
                          "'. Allowed: 'none', 'silu', 'swish'.");

  const float *px = input.AsFloat();
  const float *pw = weight.AsFloat();
  const float *pb = bias != nullptr ? bias->AsFloat() : nullptr;
  const float *pp = past_state != nullptr ? past_state->AsFloat() : nullptr;
  float *py = output.AsFloat();
  float *ps = Km1 > 0 ? present_state.AsFloat() : nullptr;

  // Build the padded view length: PadLen = L + (K - 1).
  // For each output position l (0..L-1), the convolution reads positions
  // [l, l+1, ..., l+K-1] from the padded sequence. The padded sequence is
  // [past_state || input] (concat along axis 2). When past_state is absent
  // the first K-1 positions are implicit zeros.
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t c = 0; c < C; ++c) {
      const float bias_val = pb != nullptr ? pb[c] : 0.0f;
      const float *w_row = pw + c * K;
      for (int64_t l = 0; l < L; ++l) {
        float acc = bias_val;
        for (int64_t k = 0; k < K; ++k) {
          // Index into the padded sequence.
          const int64_t pidx = l + k;
          float v;
          if (pidx < Km1) {
            // Inside the past_state (or implicit zero pad) region.
            v = pp != nullptr ? pp[(b * C + c) * Km1 + pidx] : 0.0f;
          } else {
            // Inside the actual input region.
            v = px[(b * C + c) * L + (pidx - Km1)];
          }
          acc += v * w_row[k];
        }
        if (use_silu) {
          const float sig = 1.0f / (1.0f + std::exp(-acc));
          acc = acc * sig;
        }
        py[(b * C + c) * L + l] = acc;
      }
      // Compute present_state[b, c, :] = last (K - 1) values of the padded
      // sequence. That is positions [L, L+1, ..., L+K-2] of the padded
      // sequence — equivalently the last (K - 1) values of the concatenated
      // [past_state || input] sequence.
      for (int64_t j = 0; j < Km1; ++j) {
        const int64_t pidx = L + j; // 0..K-2 mapped into padded indexing.
        float v;
        if (pidx < Km1) {
          v = pp != nullptr ? pp[(b * C + c) * Km1 + pidx] : 0.0f;
        } else {
          v = px[(b * C + c) * L + (pidx - Km1)];
        }
        ps[(b * C + c) * Km1 + j] = v;
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
