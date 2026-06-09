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

namespace {

constexpr int32_t kFloat32ExponentBias = 127;
constexpr int32_t kFloat16ExponentBias = 15;

uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e32 = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t m32 = u & 0x007fffffu;
  if (e32 == 0xff) {
    return static_cast<uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const int32_t e = e32 - kFloat32ExponentBias + kFloat16ExponentBias;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t m = (m32 | 0x00800000u) >> static_cast<uint32_t>(1 - e);
    const uint32_t round_bit = (m >> 12) & 1u;
    const uint32_t sticky = m & 0x00000fffu;
    uint16_t h = static_cast<uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<uint16_t>(h + 1);
    }
    return h;
  }
  const uint32_t low = m32 & 0x1fffu;
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<uint16_t>(h + 1);
  }
  return h;
}

float Float16BitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 0x1u;
  const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
  const uint32_t mant = static_cast<uint32_t>(h) & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      uint32_t m = mant;
      int32_t e = -1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      f = (sign << 31) | (static_cast<uint32_t>(e + kFloat32ExponentBias + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) |
        (static_cast<uint32_t>(exp - kFloat16ExponentBias + kFloat32ExponentBias) << 23) |
        (mant << 13);
  }
  float fv;
  std::memcpy(&fv, &f, sizeof(float));
  return fv;
}

} // namespace

std::pair<Tensor, Tensor> CausalConvWithState::operator()(const Tensor &input, const Tensor &weight,
                                                          const Tensor &bias,
                                                          const Tensor &past_state,
                                                          const Attributes &attrs) const {
  Tensor output;
  output.data_type = input.data_type;
  output.shape = input.shape;
  output.data.assign(static_cast<size_t>(output.element_count()) * output.element_size(), 0);

  // present_state shape = (B, C, K - 1).
  const int64_t K = weight.shape.size() >= 3 ? weight.shape[2] : 0;
  const int64_t Km1 = K > 0 ? K - 1 : 0;
  Tensor present_state;
  present_state.data_type = input.data_type;
  present_state.shape = {input.shape.empty() ? 0 : input.shape[0],
                         input.shape.size() > 1 ? input.shape[1] : 0, Km1};
  present_state.data.assign(static_cast<size_t>(present_state.shape[0]) *
                                static_cast<size_t>(present_state.shape[1]) *
                                static_cast<size_t>(Km1) * present_state.element_size(),
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
  const int32_t dtype = input.data_type;
  const bool is_float = dtype == static_cast<int32_t>(DataType::FLOAT);
  const bool is_float16 = dtype == static_cast<int32_t>(DataType::FLOAT16);
  EXT_ENFORCE_INVALID(is_float || is_float16,
                      "kernel::CausalConvWithState: input must be FLOAT or FLOAT16.");
  EXT_ENFORCE_INVALID(weight.data_type == dtype,
                      "kernel::CausalConvWithState: weight must match input dtype.");
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
    EXT_ENFORCE_INVALID(bias->data_type == dtype,
                        "kernel::CausalConvWithState: bias must match input dtype.");
    EXT_ENFORCE_INVALID(bias->shape.size() == 1 && bias->shape[0] == C,
                        "kernel::CausalConvWithState: bias must have shape (C).");
  }
  if (past_state != nullptr) {
    EXT_ENFORCE_INVALID(past_state->data_type == dtype,
                        "kernel::CausalConvWithState: past_state must match input dtype.");
    EXT_ENFORCE_INVALID(past_state->shape.size() == 3 && past_state->shape[0] == B &&
                            past_state->shape[1] == C && past_state->shape[2] == Km1,
                        "kernel::CausalConvWithState: past_state must have shape (B, C, K-1).");
  }

  EXT_ENFORCE_INVALID(output.data_type == input.data_type && output.shape == input.shape,
                      "kernel::CausalConvWithState: output buffer has mismatched type or shape.");
  const size_t elem_size = output.element_size();
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(B * C * L) * elem_size,
                      "kernel::CausalConvWithState: output buffer has wrong byte size.");
  EXT_ENFORCE_INVALID(present_state.data_type == input.data_type &&
                          present_state.shape.size() == 3 && present_state.shape[0] == B &&
                          present_state.shape[1] == C && present_state.shape[2] == Km1,
                      "kernel::CausalConvWithState: present_state has mismatched type or shape.");
  EXT_ENFORCE_INVALID(present_state.data.size() == static_cast<size_t>(B * C * Km1) * elem_size,
                      "kernel::CausalConvWithState: present_state buffer has wrong byte size.");

  const bool use_silu = attrs.activation == "silu" || attrs.activation == "swish";
  EXT_ENFORCE_INVALID(use_silu || attrs.activation == "none",
                      "kernel::CausalConvWithState: unsupported activation '" + attrs.activation +
                          "'. Allowed: 'none', 'silu', 'swish'.");

  const float *px_f = is_float ? input.AsFloat() : nullptr;
  const float *pw_f = is_float ? weight.AsFloat() : nullptr;
  const float *pb_f = (is_float && bias != nullptr) ? bias->AsFloat() : nullptr;
  const float *pp_f = (is_float && past_state != nullptr) ? past_state->AsFloat() : nullptr;
  float *py_f = is_float ? output.AsFloat() : nullptr;
  float *ps_f = (is_float && Km1 > 0) ? present_state.AsFloat() : nullptr;

  const uint16_t *px_h = is_float16 ? reinterpret_cast<const uint16_t *>(input.bytes()) : nullptr;
  const uint16_t *pw_h = is_float16 ? reinterpret_cast<const uint16_t *>(weight.bytes()) : nullptr;
  const uint16_t *pb_h =
      (is_float16 && bias != nullptr) ? reinterpret_cast<const uint16_t *>(bias->bytes()) : nullptr;
  const uint16_t *pp_h = (is_float16 && past_state != nullptr)
                             ? reinterpret_cast<const uint16_t *>(past_state->bytes())
                             : nullptr;
  uint16_t *py_h =
      is_float16 ? reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(output.bytes())) : nullptr;
  uint16_t *ps_h = (is_float16 && Km1 > 0)
                       ? reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(present_state.bytes()))
                       : nullptr;

  auto load = [&](const float *pf, const uint16_t *ph, int64_t idx) -> float {
    return is_float ? pf[idx] : Float16BitsToFloat(ph[idx]);
  };
  auto store = [&](float *pf, uint16_t *ph, int64_t idx, float value) {
    if (is_float) {
      pf[idx] = value;
    } else {
      ph[idx] = FloatToFloat16Bits(value);
    }
  };

  // Build the padded view length: PadLen = L + (K - 1).
  // For each output position l (0..L-1), the convolution reads positions
  // [l, l+1, ..., l+K-1] from the padded sequence. The padded sequence is
  // [past_state || input] (concat along axis 2). When past_state is absent
  // the first K-1 positions are implicit zeros.
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t c = 0; c < C; ++c) {
      const float bias_val = bias != nullptr ? load(pb_f, pb_h, c) : 0.0f;
      for (int64_t l = 0; l < L; ++l) {
        float acc = bias_val;
        for (int64_t k = 0; k < K; ++k) {
          // Index into the padded sequence.
          const int64_t pidx = l + k;
          float v;
          if (pidx < Km1) {
            // Inside the past_state (or implicit zero pad) region.
            v = past_state != nullptr ? load(pp_f, pp_h, (b * C + c) * Km1 + pidx) : 0.0f;
          } else {
            // Inside the actual input region.
            v = load(px_f, px_h, (b * C + c) * L + (pidx - Km1));
          }
          acc += v * load(pw_f, pw_h, c * K + k);
        }
        if (use_silu) {
          const float sig = 1.0f / (1.0f + std::exp(-acc));
          acc = acc * sig;
        }
        store(py_f, py_h, (b * C + c) * L + l, acc);
      }
      // Compute present_state[b, c, :] = last (K - 1) values of the padded
      // sequence. That is positions [L, L+1, ..., L+K-2] of the padded
      // sequence — equivalently the last (K - 1) values of the concatenated
      // [past_state || input] sequence.
      for (int64_t j = 0; j < Km1; ++j) {
        const int64_t pidx = L + j; // 0..K-2 mapped into padded indexing.
        float v;
        if (pidx < Km1) {
          v = past_state != nullptr ? load(pp_f, pp_h, (b * C + c) * Km1 + pidx) : 0.0f;
        } else {
          v = load(px_f, px_h, (b * C + c) * L + (pidx - Km1));
        }
        store(ps_f, ps_h, (b * C + c) * Km1 + j, v);
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
