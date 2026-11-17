// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

size_t CausalConvBytes(const Shape &shape, int32_t dtype) {
  int64_t count = 1;
  for (const int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "kernel::CausalConvWithState: negative dimension.");
    EXT_ENFORCE_INVALID(dim == 0 || count <= std::numeric_limits<int64_t>::max() / dim,
                        "kernel::CausalConvWithState: tensor element count overflows int64.");
    count *= dim;
  }
  return core::runtime::PackedByteSize(dtype, count);
}

int64_t CausalConvHistory(const Tensor &input, const Tensor &weight,
                          const CausalConvWithState::Attributes &attrs) {
  EXT_ENFORCE_INVALID(attrs.dilation >= 1, "kernel::CausalConvWithState: dilation must be >= 1.");
  EXT_ENFORCE_INVALID(input.shape.size() == 3 && weight.shape.size() == 3,
                      "kernel::CausalConvWithState: input and weight must have rank 3.");
  EXT_ENFORCE_INVALID(weight.shape[1] == 1 && weight.shape[0] == input.shape[1],
                      "kernel::CausalConvWithState: weight must have shape (C, 1, K).");
  const int64_t K = weight.shape[2];
  EXT_ENFORCE_INVALID(K >= 1, "kernel::CausalConvWithState: kernel size K must be >= 1.");
  EXT_ENFORCE_INVALID(K - 1 <= (std::numeric_limits<int64_t>::max() - 1) / attrs.dilation,
                      "kernel::CausalConvWithState: effective kernel size overflows int64.");
  const int64_t history = (K - 1) * attrs.dilation;
  EXT_ENFORCE_INVALID(input.shape[2] >= 0 &&
                          input.shape[2] <= std::numeric_limits<int64_t>::max() - history,
                      "kernel::CausalConvWithState: padded length overflows int64.");
  EXT_ENFORCE_INVALID(input.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          input.data_type == static_cast<int32_t>(DataType::FLOAT16),
                      "kernel::CausalConvWithState: input must be FLOAT or FLOAT16.");
  EXT_ENFORCE_INVALID(weight.data_type == input.data_type,
                      "kernel::CausalConvWithState: weight must match input dtype.");
  EXT_ENFORCE_INVALID(input.size_bytes() == CausalConvBytes(input.shape, input.data_type) &&
                          weight.size_bytes() == CausalConvBytes(weight.shape, weight.data_type),
                      "kernel::CausalConvWithState: input or weight has wrong byte size.");
  CausalConvBytes({input.shape[0], input.shape[1], history}, input.data_type);
  return history;
}

} // namespace

// IEEE-754 binary16 ↔ binary32 conversions used by the FLOAT16 dispatch
// path are provided by ``onnx_core/runtime/kernels/cast_helper.h``.

std::pair<Tensor, Tensor> CausalConvWithState::operator()(const Tensor &input, const Tensor &weight,
                                                          const Tensor &bias,
                                                          const Tensor &past_state,
                                                          const Attributes &attrs,
                                                          RuntimeContext *rt) const {
  const int64_t history = CausalConvHistory(input, weight, attrs);
  const size_t output_n_bytes = CausalConvBytes(input.shape, input.data_type);
  Tensor output = rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, output_n_bytes)
                     : MakeOutputTensor(input.data_type, input.shape, output_n_bytes, nullptr);

  const Shape present_shape{input.shape[0], input.shape[1], history};
  const size_t present_n_bytes = CausalConvBytes(present_shape, input.data_type);
  Tensor present_state =
      rt ? rt->MakeOutputTensor(1, input.data_type, present_shape, present_n_bytes)
         : MakeOutputTensor(input.data_type, present_shape, present_n_bytes, nullptr);

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
  const int64_t history = CausalConvHistory(input, weight, attrs);
  const int32_t dtype = input.data_type;
  const bool is_float = dtype == static_cast<int32_t>(DataType::FLOAT);
  const bool is_float16 = dtype == static_cast<int32_t>(DataType::FLOAT16);

  const int64_t B = input.shape[0];
  const int64_t C = input.shape[1];
  const int64_t L = input.shape[2];
  const int64_t K = weight.shape[2];

  if (bias != nullptr) {
    EXT_ENFORCE_INVALID(bias->data_type == dtype,
                        "kernel::CausalConvWithState: bias must match input dtype.");
    EXT_ENFORCE_INVALID(bias->shape.size() == 1 && bias->shape[0] == C,
                        "kernel::CausalConvWithState: bias must have shape (C).");
    EXT_ENFORCE_INVALID(bias->size_bytes() == CausalConvBytes(bias->shape, dtype),
                        "kernel::CausalConvWithState: bias has wrong byte size.");
  }
  if (past_state != nullptr) {
    EXT_ENFORCE_INVALID(past_state->data_type == dtype,
                        "kernel::CausalConvWithState: past_state must match input dtype.");
    EXT_ENFORCE_INVALID(
        past_state->shape.size() == 3 && past_state->shape[0] == B && past_state->shape[1] == C &&
            past_state->shape[2] == history,
        "kernel::CausalConvWithState: past_state must have shape (B, C, (K-1)*dilation).");
    EXT_ENFORCE_INVALID(past_state->size_bytes() == CausalConvBytes(past_state->shape, dtype),
                        "kernel::CausalConvWithState: past_state has wrong byte size.");
  }

  EXT_ENFORCE_INVALID(output.data_type == input.data_type && output.shape == input.shape,
                      "kernel::CausalConvWithState: output buffer has mismatched type or shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == CausalConvBytes(input.shape, dtype),
                      "kernel::CausalConvWithState: output buffer has wrong byte size.");
  EXT_ENFORCE_INVALID(present_state.data_type == input.data_type &&
                          present_state.shape.size() == 3 && present_state.shape[0] == B &&
                          present_state.shape[1] == C && present_state.shape[2] == history,
                      "kernel::CausalConvWithState: present_state has mismatched type or shape.");
  EXT_ENFORCE_INVALID(present_state.size_bytes() == CausalConvBytes(present_state.shape, dtype),
                      "kernel::CausalConvWithState: present_state buffer has wrong byte size.");

  const bool use_silu = attrs.activation == "silu" || attrs.activation == "swish";
  EXT_ENFORCE_INVALID(use_silu || attrs.activation == "none",
                      "kernel::CausalConvWithState: unsupported activation '", attrs.activation,
                      "'. Allowed: 'none', 'silu', 'swish'.");

  const float *px_f = is_float ? input.AsFloat() : nullptr;
  const float *pw_f = is_float ? weight.AsFloat() : nullptr;
  const float *pb_f = (is_float && bias != nullptr) ? bias->AsFloat() : nullptr;
  const float *pp_f = (is_float && past_state != nullptr) ? past_state->AsFloat() : nullptr;
  float *py_f = is_float ? output.AsFloat() : nullptr;
  float *ps_f = (is_float && history > 0) ? present_state.AsFloat() : nullptr;

  const uint16_t *px_h = is_float16 ? reinterpret_cast<const uint16_t *>(input.bytes()) : nullptr;
  const uint16_t *pw_h = is_float16 ? reinterpret_cast<const uint16_t *>(weight.bytes()) : nullptr;
  const uint16_t *pb_h =
      (is_float16 && bias != nullptr) ? reinterpret_cast<const uint16_t *>(bias->bytes()) : nullptr;
  const uint16_t *pp_h = (is_float16 && past_state != nullptr)
                             ? reinterpret_cast<const uint16_t *>(past_state->bytes())
                             : nullptr;
  uint16_t *py_h =
      is_float16 ? reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(output.bytes())) : nullptr;
  uint16_t *ps_h = (is_float16 && history > 0)
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

  // The padded view is [past_state || input], with implicit zero history
  // when past_state is absent. Only convolution taps are spaced by dilation.
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t c = 0; c < C; ++c) {
      const float bias_val = bias != nullptr ? load(pb_f, pb_h, c) : 0.0f;
      for (int64_t l = 0; l < L; ++l) {
        float acc = bias_val;
        for (int64_t k = 0; k < K; ++k) {
          // Index into the padded sequence.
          const int64_t pidx = l + k * attrs.dilation;
          float v;
          if (pidx < history) {
            // Inside the past_state (or implicit zero pad) region.
            v = past_state != nullptr ? load(pp_f, pp_h, (b * C + c) * history + pidx) : 0.0f;
          } else {
            // Inside the actual input region.
            v = load(px_f, px_h, (b * C + c) * L + (pidx - history));
          }
          acc += v * load(pw_f, pw_h, c * K + k);
        }
        if (use_silu) {
          const float sig = 1.0f / (1.0f + std::exp(-acc));
          acc = acc * sig;
        }
        store(py_f, py_h, (b * C + c) * L + l, acc);
      }
      // Retains consecutive raw samples, including those skipped by taps.
      for (int64_t j = 0; j < history; ++j) {
        const int64_t pidx = L + j;
        float v;
        if (pidx < history) {
          v = past_state != nullptr ? load(pp_f, pp_h, (b * C + c) * history + pidx) : 0.0f;
        } else {
          v = load(px_f, px_h, (b * C + c) * L + (pidx - history));
        }
        store(ps_f, ps_h, (b * C + c) * history + j, v);
      }
    }
  }
}

void CausalConvWithState::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 2 || node.input_size() > 4), "RunNode: op '",
                      node.op_type(), "' expects between 2 and 4 input(s), got ", node.input_size(),
                      ".");
  RequireOutputCount(node, 2);
  const Tensor &input = GetInput(node, 0, rt);
  const Tensor &weight = GetInput(node, 1, rt);
  const Tensor *bias = GetOptionalInput(node, 2, rt);
  const Tensor *past_state = GetOptionalInput(node, 3, rt);

  onnx_kernels::kernel::CausalConvWithState::Attributes attrs;
  attrs.activation = GetAttributeStringOrDefault(node, "activation", "none");
  attrs.dilation = GetAttributeIntOrDefault(node, "dilation", 1);

  onnx_kernels::kernel::CausalConvWithState kernel(rt.kernel_ctx());
  auto [output, present_state] = kernel(input, weight, bias != nullptr ? *bias : Tensor{},
                                        past_state != nullptr ? *past_state : Tensor{}, attrs, &rt);
  SetOutput(node, 0, std::move(output), rt);
  SetOutput(node, 1, std::move(present_state), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
