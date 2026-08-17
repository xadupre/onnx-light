// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Validates a 3D packed FLOAT tensor and returns its shape components.
void Check3DFloat(const Tensor &t, const char *label, int64_t &B, int64_t &T, int64_t &last) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, "kernel::LinearAttention: '", label,
                      "' must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 3, "kernel::LinearAttention: '", label,
                      "' must be rank-3.");
  B = t.shape[0];
  T = t.shape[1];
  last = t.shape[2];
}

} // namespace

Tensor LinearAttention::operator()(const Tensor &query, const Tensor &key, const Tensor &value,
                                   RuntimeContext *rt) const {
  Attributes attrs;
  attrs.update_rule = "linear";
  attrs.q_num_heads = 0; // will be inferred below
  attrs.kv_num_heads = 0;
  // Infer heads from query/key: assume q_num_heads == kv_num_heads and
  // d_k from key. For default overload, require same hidden dim.
  EXT_ENFORCE_INVALID(query.shape.size() == 3 && key.shape.size() == 3,
                      "kernel::LinearAttention: default overload requires rank-3 inputs.");
  EXT_ENFORCE_INVALID(key.shape[2] > 0, "kernel::LinearAttention: key last dim must be positive.");
  // Assume kv_num_heads = q_num_heads = key.shape[2] / d_k is ambiguous;
  // default to 1 head.
  attrs.q_num_heads = 1;
  attrs.kv_num_heads = 1;
  return (*this)(query, key, value, attrs, nullptr, nullptr, nullptr, rt).output;
}

LinearAttention::Result LinearAttention::operator()(const Tensor &query, const Tensor &key,
                                                    const Tensor &value, const Attributes &attrs,
                                                    const Tensor *past_state, const Tensor *decay,
                                                    const Tensor *beta, RuntimeContext *rt) const {
  // Half-precision fast path: promote FLOAT16/BFLOAT16 inputs to FLOAT32,
  // compute the recurrence in float32, then demote the outputs back to the
  // activation dtype. This mirrors the ONNX reference, which accumulates in
  // float32 and casts ``output`` (and ``present_state``) back afterwards.
  if (IsHalfPrecision(query.data_type)) {
    const int32_t out_dtype = query.data_type;
    const int32_t state_dtype = (past_state != nullptr) ? past_state->data_type : query.data_type;
    RuntimeContext scratch_rt(
        rt ? rt->kernel_ctx() : ctx_,
        RuntimeContextOptions{.allocator = rt ? rt->execution_allocator() : nullptr});
    RuntimeContext *compute_rt = rt ? &scratch_rt : nullptr;
    const Tensor query_f = PromoteToFloat32(query, compute_rt);
    const Tensor key_f = PromoteToFloat32(key, compute_rt);
    const Tensor value_f = PromoteToFloat32(value, compute_rt);
    Tensor past_f, decay_f, beta_f;
    const Tensor *past_ptr = nullptr;
    const Tensor *decay_ptr_in = nullptr;
    const Tensor *beta_ptr_in = nullptr;
    if (past_state != nullptr) {
      past_f = PromoteToFloat32(*past_state, compute_rt);
      past_ptr = &past_f;
    }
    if (decay != nullptr) {
      decay_f = PromoteToFloat32(*decay, compute_rt);
      decay_ptr_in = &decay_f;
    }
    if (beta != nullptr) {
      beta_f = PromoteToFloat32(*beta, compute_rt);
      beta_ptr_in = &beta_f;
    }
    Result result_f =
        (*this)(query_f, key_f, value_f, attrs, past_ptr, decay_ptr_in, beta_ptr_in, compute_rt);
    Result result;
    Tensor output_demoted = IsHalfPrecision(out_dtype)
                                ? DemoteFromFloat32(result_f.output, out_dtype, compute_rt)
                                : std::move(result_f.output);
    Tensor state_demoted = IsHalfPrecision(state_dtype)
                               ? DemoteFromFloat32(result_f.present_state, state_dtype, compute_rt)
                               : std::move(result_f.present_state);
    result.output = rt ? rt->MakeOutputTensor(0, output_demoted.data_type, output_demoted.shape,
                                              output_demoted.size_bytes())
                       : MakeOutputTensor(output_demoted.data_type, output_demoted.shape,
                                          output_demoted.size_bytes(), nullptr);
    const bool has_state_output = rt == nullptr || rt->output_slot_io_roles().empty() ||
                                  rt->output_slot_io_roles().size() > 1;
    result.present_state =
        rt ? (!has_state_output
                  ? rt->MakeTemporaryTensor(state_demoted.data_type, state_demoted.shape,
                                            state_demoted.size_bytes())
                  : rt->MakeOutputTensor(1, state_demoted.data_type, state_demoted.shape,
                                         state_demoted.size_bytes()))
           : MakeOutputTensor(state_demoted.data_type, state_demoted.shape,
                              state_demoted.size_bytes(), nullptr);
    if (output_demoted.size_bytes() != 0) {
      std::memcpy(result.output.mutable_bytes(), output_demoted.bytes(),
                  output_demoted.size_bytes());
    }
    if (state_demoted.size_bytes() != 0) {
      std::memcpy(result.present_state.mutable_bytes(), state_demoted.bytes(),
                  state_demoted.size_bytes());
    }
    return result;
  }

  // Validate inputs
  int64_t B_q, T_q, hidden_q;
  Check3DFloat(query, "query", B_q, T_q, hidden_q);
  int64_t B_k, T_k, hidden_k;
  Check3DFloat(key, "key", B_k, T_k, hidden_k);
  int64_t B_v, T_v, hidden_v;
  Check3DFloat(value, "value", B_v, T_v, hidden_v);

  EXT_ENFORCE_INVALID(B_q == B_k && B_q == B_v, "kernel::LinearAttention: batch sizes must match.");
  EXT_ENFORCE_INVALID(T_q == T_k && T_q == T_v,
                      "kernel::LinearAttention: sequence lengths must match.");

  const int64_t B = B_q;
  const int64_t T = T_q;
  const int64_t q_num_heads = attrs.q_num_heads;
  const int64_t kv_num_heads = attrs.kv_num_heads;

  EXT_ENFORCE_INVALID(q_num_heads > 0 && kv_num_heads > 0,
                      "kernel::LinearAttention: q_num_heads and kv_num_heads must be positive.");
  EXT_ENFORCE_INVALID(q_num_heads % kv_num_heads == 0,
                      "kernel::LinearAttention: q_num_heads must be a multiple of kv_num_heads.");

  EXT_ENFORCE_INVALID(hidden_q % q_num_heads == 0,
                      "kernel::LinearAttention: query hidden dim not divisible by q_num_heads.");
  EXT_ENFORCE_INVALID(hidden_k % kv_num_heads == 0,
                      "kernel::LinearAttention: key hidden dim not divisible by kv_num_heads.");
  EXT_ENFORCE_INVALID(hidden_v % kv_num_heads == 0,
                      "kernel::LinearAttention: value hidden dim not divisible by kv_num_heads.");

  const int64_t d_k = hidden_k / kv_num_heads;
  const int64_t d_v = hidden_v / kv_num_heads;
  const int64_t q_d_k = hidden_q / q_num_heads;
  EXT_ENFORCE_INVALID(q_d_k == d_k,
                      "kernel::LinearAttention: query head_size must equal key head_size.");

  const int64_t heads_per_kv = q_num_heads / kv_num_heads;

  // Compute scale: the schema's ``scale`` attribute defaults to 0.0f, which
  // means "derive 1/sqrt(d_k)". ``has_scale`` short-circuits the same default
  // when the caller did not set the attribute at all.
  float scale;
  if (attrs.has_scale && attrs.scale != 0.0f) {
    scale = attrs.scale;
  } else {
    scale = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  // Determine update rule
  const std::string &rule = attrs.update_rule;
  const bool use_decay = (rule == "gated" || rule == "gated_delta");
  const bool use_beta = (rule == "delta" || rule == "gated_delta");

  // Validate decay
  int64_t decay_last = 0;
  const float *decay_ptr = nullptr;
  if (use_decay) {
    EXT_ENFORCE_INVALID(decay != nullptr,
                        "kernel::LinearAttention: decay required for gated/gated_delta.");
    EXT_ENFORCE_INVALID(decay->data_type == DataType::FLOAT,
                        "kernel::LinearAttention: decay must be FLOAT.");
    EXT_ENFORCE_INVALID(decay->shape.size() == 3 && decay->shape[0] == B && decay->shape[1] == T,
                        "kernel::LinearAttention: decay must be (B, T, ...).");
    decay_last = decay->shape[2];
    EXT_ENFORCE_INVALID(decay_last == kv_num_heads * d_k || decay_last == kv_num_heads,
                        "kernel::LinearAttention: decay last dim must be H_kv*d_k or H_kv.");
    decay_ptr = decay->AsFloat();
  }

  // Validate beta
  int64_t beta_last = 0;
  const float *beta_ptr = nullptr;
  if (use_beta) {
    EXT_ENFORCE_INVALID(beta != nullptr,
                        "kernel::LinearAttention: beta required for delta/gated_delta.");
    EXT_ENFORCE_INVALID(beta->data_type == DataType::FLOAT,
                        "kernel::LinearAttention: beta must be FLOAT.");
    EXT_ENFORCE_INVALID(beta->shape.size() == 3 && beta->shape[0] == B && beta->shape[1] == T,
                        "kernel::LinearAttention: beta must be (B, T, ...).");
    beta_last = beta->shape[2];
    EXT_ENFORCE_INVALID(beta_last == kv_num_heads || beta_last == 1,
                        "kernel::LinearAttention: beta last dim must be H_kv or 1.");
    beta_ptr = beta->AsFloat();
  }

  // Initialize state: (B, H_kv, d_k, d_v). The state buffer is drawn from the
  // runtime allocator (via ``MakeOutputTensor``) so it is allocator-backed and
  // becomes the ``present_state`` result without an extra copy. The recurrence
  // accumulates into it, so it is explicitly zero-initialized (or seeded from
  // ``past_state``) rather than relying on the allocator returning zeroed bytes.
  const int64_t state_size = B * kv_num_heads * d_k * d_v;
  const Shape state_shape{B, kv_num_heads, d_k, d_v};
  const size_t state_n_bytes = static_cast<size_t>(state_size) * sizeof(float);
  const bool has_state_output =
      rt == nullptr || rt->output_slot_io_roles().empty() || rt->output_slot_io_roles().size() > 1;
  Tensor state_tensor =
      rt ? (!has_state_output ? rt->MakeTemporaryTensor(static_cast<int32_t>(DataType::FLOAT),
                                                        state_shape, state_n_bytes)
                              : rt->MakeOutputTensor(1, static_cast<int32_t>(DataType::FLOAT),
                                                     state_shape, state_n_bytes))
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), state_shape, state_n_bytes,
                            nullptr);
  float *state = state_tensor.AsFloat();
  if (past_state != nullptr) {
    EXT_ENFORCE_INVALID(past_state->data_type == DataType::FLOAT,
                        "kernel::LinearAttention: past_state must be FLOAT.");
    EXT_ENFORCE_INVALID(past_state->shape.size() == 4 && past_state->shape[0] == B &&
                            past_state->shape[1] == kv_num_heads && past_state->shape[2] == d_k &&
                            past_state->shape[3] == d_v,
                        "kernel::LinearAttention: past_state shape must be (B, H_kv, d_k, d_v).");
    const float *ps = past_state->AsFloat();
    for (int64_t i = 0; i < state_size; ++i) {
      state[static_cast<size_t>(i)] = ps[i];
    }
  } else {
    std::fill(state, state + state_size, 0.0f);
  }

  // Output buffer: (B, T, H_q * d_v), likewise drawn from the runtime allocator.
  const int64_t out_hidden = q_num_heads * d_v;
  const int64_t output_size = B * T * out_hidden;
  Tensor output_tensor =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), {B, T, out_hidden},
                                static_cast<size_t>(output_size) * sizeof(float))
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {B, T, out_hidden},
                            static_cast<size_t>(output_size) * sizeof(float), nullptr);
  float *output = output_tensor.AsFloat();

  const float *q_ptr = query.AsFloat();
  const float *k_ptr = key.AsFloat();
  const float *v_ptr = value.AsFloat();

  // Sequential recurrence over time steps
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      // Process each KV head
      for (int64_t h_kv = 0; h_kv < kv_num_heads; ++h_kv) {
        // State offset: state[b, h_kv, :, :] is at b*H_kv*d_k*d_v + h_kv*d_k*d_v
        float *S = state + static_cast<size_t>((b * kv_num_heads + h_kv) * d_k * d_v);

        // Key vector for this head at this time step
        // k[b, t, h_kv*d_k : (h_kv+1)*d_k]
        const float *k_t = k_ptr + static_cast<size_t>((b * T + t) * hidden_k + h_kv * d_k);
        // Value vector for this head
        const float *v_t = v_ptr + static_cast<size_t>((b * T + t) * hidden_v + h_kv * d_v);

        // Get decay for this head/timestep
        // decay_last == kv_num_heads*d_k: per-key-dim
        // decay_last == kv_num_heads: per-head scalar
        detail::TemporaryTypedBuffer<float> g_exp_buf(static_cast<std::size_t>(d_k),
                                                      rt ? rt->execution_allocator() : nullptr,
                                                      "kernel::LinearAttention g_exp");
        float *g_exp = g_exp_buf.data();
        std::fill(g_exp, g_exp + d_k, 1.0f);
        if (use_decay) {
          if (decay_last == kv_num_heads * d_k) {
            const float *g_ptr =
                decay_ptr + static_cast<size_t>((b * T + t) * decay_last + h_kv * d_k);
            for (int64_t i = 0; i < d_k; ++i) {
              g_exp[static_cast<size_t>(i)] = std::exp(g_ptr[i]);
            }
          } else {
            // Per-head scalar: broadcast over d_k
            float g_val = decay_ptr[static_cast<size_t>((b * T + t) * decay_last + h_kv)];
            float eg = std::exp(g_val);
            for (int64_t i = 0; i < d_k; ++i) {
              g_exp[static_cast<size_t>(i)] = eg;
            }
          }
        }

        // Get beta for this head/timestep
        float beta_val = 1.0f;
        if (use_beta) {
          if (beta_last == kv_num_heads) {
            beta_val = beta_ptr[static_cast<size_t>((b * T + t) * beta_last + h_kv)];
          } else {
            // beta_last == 1: broadcast
            beta_val = beta_ptr[static_cast<size_t>((b * T + t) * beta_last)];
          }
        }

        // Apply update rule
        if (rule == "linear") {
          // S_t = S_{t-1} + k_t ⊗ v_t
          for (int64_t i = 0; i < d_k; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
              S[static_cast<size_t>(i * d_v + j)] += k_t[i] * v_t[j];
            }
          }
        } else if (rule == "gated") {
          // S_t = exp(g_t) * S_{t-1} + k_t ⊗ v_t
          for (int64_t i = 0; i < d_k; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
              S[static_cast<size_t>(i * d_v + j)] =
                  g_exp[static_cast<size_t>(i)] * S[static_cast<size_t>(i * d_v + j)] +
                  k_t[i] * v_t[j];
            }
          }
        } else if (rule == "delta") {
          // S_t = S_{t-1} + β_t * k_t ⊗ (v_t - S_{t-1}^T k_t)
          // S_{t-1}^T k_t = sum_i(S[i,:] * k_t[i]) => vector of d_v
          detail::TemporaryTypedBuffer<float> Sk_buf(static_cast<std::size_t>(d_v),
                                                     rt ? rt->execution_allocator() : nullptr,
                                                     "kernel::LinearAttention Sk");
          float *Sk = Sk_buf.data();
          std::fill(Sk, Sk + d_v, 0.0f);
          for (int64_t i = 0; i < d_k; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
              Sk[static_cast<size_t>(j)] += S[static_cast<size_t>(i * d_v + j)] * k_t[i];
            }
          }
          for (int64_t i = 0; i < d_k; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
              float diff = v_t[j] - Sk[static_cast<size_t>(j)];
              S[static_cast<size_t>(i * d_v + j)] += beta_val * k_t[i] * diff;
            }
          }
        } else if (rule == "gated_delta") {
          // S_t = exp(g_t)*S_{t-1} + β_t * k_t ⊗ (v_t - exp(g_t)*S_{t-1}^T k_t)
          // First compute exp(g_t)*S_{t-1}^T k_t
          detail::TemporaryTypedBuffer<float> gSk_buf(static_cast<std::size_t>(d_v),
                                                      rt ? rt->execution_allocator() : nullptr,
                                                      "kernel::LinearAttention gSk");
          float *gSk = gSk_buf.data();
          std::fill(gSk, gSk + d_v, 0.0f);
          for (int64_t i = 0; i < d_k; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
              gSk[static_cast<size_t>(j)] +=
                  g_exp[static_cast<size_t>(i)] * S[static_cast<size_t>(i * d_v + j)] * k_t[i];
            }
          }
          // Apply decay to state and add delta
          for (int64_t i = 0; i < d_k; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
              float decayed = g_exp[static_cast<size_t>(i)] * S[static_cast<size_t>(i * d_v + j)];
              float diff = v_t[j] - gSk[static_cast<size_t>(j)];
              S[static_cast<size_t>(i * d_v + j)] = decayed + beta_val * k_t[i] * diff;
            }
          }
        } else {
          EXT_ENFORCE_INVALID(false, "kernel::LinearAttention: unknown update_rule '", rule, "'.");
        }

        // Compute output for each query head that shares this KV head
        for (int64_t qg = 0; qg < heads_per_kv; ++qg) {
          int64_t h_q = h_kv * heads_per_kv + qg;
          // q[b, t, h_q*d_k : (h_q+1)*d_k]
          const float *q_t = q_ptr + static_cast<size_t>((b * T + t) * hidden_q + h_q * d_k);
          // o_t = scale * q_t^T S_t => vector of d_v
          float *out_t = output + static_cast<size_t>((b * T + t) * out_hidden + h_q * d_v);
          for (int64_t j = 0; j < d_v; ++j) {
            float dot = 0.0f;
            for (int64_t i = 0; i < d_k; ++i) {
              dot += q_t[i] * S[static_cast<size_t>(i * d_v + j)];
            }
            out_t[j] = scale * dot;
          }
        }
      }
    }
  }

  Result result;
  result.output = std::move(output_tensor);
  result.present_state = std::move(state_tensor);
  return result;
}

void LinearAttention::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 6),
                      "RunNode: op 'LinearAttention' expects between 3 and 6 "
                      "input(s), got ",
                      node.input_size(), ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 2),
                      "RunNode: op 'LinearAttention' expects 1 or 2 output(s), "
                      "got ",
                      node.output_size(), ".");
  const Tensor &query = GetInput(node, 0, rt.tensors());
  const Tensor &key = GetInput(node, 1, rt.tensors());
  const Tensor &value = GetInput(node, 2, rt.tensors());
  const Tensor *past_state = GetOptionalInput(node, 3, rt.tensors());
  const Tensor *decay = GetOptionalInput(node, 4, rt.tensors());
  const Tensor *beta = GetOptionalInput(node, 5, rt.tensors());

  onnx_kernels::kernel::LinearAttention::Attributes attrs;
  attrs.update_rule = GetAttributeStringOrDefault(node, "update_rule", "gated_delta");
  if (FindAttribute(node, "scale") != nullptr) {
    attrs.has_scale = true;
    attrs.scale = GetAttributeFloatOrDefault(node, "scale", 0.0f);
  }
  attrs.q_num_heads = GetAttributeIntOrDefault(node, "q_num_heads", 0);
  attrs.kv_num_heads = GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  attrs.chunk_size = GetAttributeIntOrDefault(node, "chunk_size", 64);

  onnx_kernels::kernel::LinearAttention k(rt.kernel_ctx());
  onnx_kernels::kernel::LinearAttention::Result result =
      k(query, key, value, attrs, past_state, decay, beta, &rt);
  SetOutput(node, 0, std::move(result.output), rt.tensors());

  if (node.output_size() >= 2) {
    const std::string present_name = node.output(1);
    if (!present_name.empty()) {
      result.present_state.name = present_name;
      rt.tensors()[present_name] = std::move(result.present_state);
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
