// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/preview/include_preview_kernels.h"

#include "onnx_core/runtime/float16_promote.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Validates that ``t`` is a rank-4 tensor with a supported element type
// (FLOAT, DOUBLE, FLOAT16, or BFLOAT16). The caller is identified by ``label``
// for clearer error messages.
void CheckRank4Float(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT || t.data_type == DataType::DOUBLE ||
                          t.data_type == DataType::FLOAT16 || t.data_type == DataType::BFLOAT16,
                      "kernel::FlexAttention: '", label,
                      "' must be a FLOAT, DOUBLE, FLOAT16 or BFLOAT16 tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4, "kernel::FlexAttention: '", label,
                      "' must be a rank-4 tensor.");
  for (int64_t d : t.shape) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::FlexAttention: '", label, "' has a negative dimension.");
  }
}

// Returns the per-element byte size for a FLOAT/DOUBLE/FLOAT16/BFLOAT16 element
// type.
size_t ElementBytes(int32_t dtype) {
  if (dtype == static_cast<int32_t>(DataType::FLOAT)) {
    return sizeof(float);
  }
  if (dtype == static_cast<int32_t>(DataType::DOUBLE)) {
    return sizeof(double);
  }
  return sizeof(uint16_t);
}

// Reference implementation of FlexAttention for a single floating-point storage
// type ``T`` (``float`` or ``double``). The pre-softmax scores and post-softmax
// probabilities are materialized in a ``T`` tensor of shape
// ``(batch_size, q_num_heads, q_seq_len, kv_seq_len)`` so the optional
// ``score_mod`` / ``prob_mod`` callbacks can rewrite them in place; all
// accumulation uses ``double`` regardless of ``T``. ``output`` must already be a
// ``T`` tensor of shape ``(batch_size, q_num_heads, q_seq_len, v_head_size)``.
template <typename T>
void ComputeFlexAttentionTyped(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                               const FlexAttention::ScoreModFn &score_mod,
                               const FlexAttention::ProbModFn &prob_mod, Tensor &output,
                               RawBufferAllocator *allocator = nullptr) {
  constexpr int32_t kElementType = TensorElementType<T>::value;

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
                      "kernel::FlexAttention: 'Q', 'K', 'V' must share the same batch size.");
  EXT_ENFORCE_INVALID(k_head_size == head_size,
                      "kernel::FlexAttention: 'K' head_size must match 'Q' head_size.");
  EXT_ENFORCE_INVALID(v_num_heads == kv_num_heads,
                      "kernel::FlexAttention: 'V' num_heads must match 'K' num_heads.");
  EXT_ENFORCE_INVALID(v_seq_len == kv_seq_len,
                      "kernel::FlexAttention: 'V' kv_seq_len must match 'K' kv_seq_len.");
  EXT_ENFORCE_INVALID(
      kv_num_heads > 0 && q_num_heads % kv_num_heads == 0,
      "kernel::FlexAttention: 'q_num_heads' must be a positive multiple of 'kv_num_heads'.");

  EXT_ENFORCE_INVALID(output.data_type == kElementType,
                      "kernel::FlexAttention preallocated output must share Q's element type.");
  const onnx_kernels::Shape expected_out_shape = {batch_size, q_num_heads, q_seq_len, v_head_size};
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::FlexAttention preallocated output shape must be (batch_size, q_num_heads, "
      "q_seq_len, v_head_size).");
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * sizeof(T),
      "kernel::FlexAttention preallocated output buffer has unexpected size in bytes.");

  const int64_t group_size = q_num_heads / kv_num_heads;
  const T *pQ = Q.As<T>();
  const T *pK = K.As<T>();
  const T *pV = V.As<T>();
  T *pY = output.As<T>();

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

  // Allocate the full (B, Hq, L, S) score / probability tensor so the
  // optional ``score_mod`` and ``prob_mod`` callbacks — which operate on
  // the whole tensor in ONNX — can rewrite it in place before the
  // softmax and the final ``probs @ V`` matmul, respectively.
  const onnx_kernels::Shape probs_shape = {batch_size, q_num_heads, q_seq_len, kv_seq_len};
  const int64_t probs_count = batch_size * q_num_heads * q_seq_len * kv_seq_len;
  const size_t probs_n_bytes = static_cast<size_t>(probs_count) * sizeof(T);
  Tensor probs = MakeOutputTensor(kElementType, probs_shape, probs_n_bytes, allocator);
  T *pProbs = probs.As<T>();

  const int64_t probs_head_stride = q_seq_len * kv_seq_len;
  const int64_t probs_batch_stride = q_num_heads * probs_head_stride;

  // Phase 1: scores = (Q @ K^T) * scale, written into ``probs`` so the
  // optional ``score_mod`` callback can rewrite it in place.
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < q_num_heads; ++h) {
      const int64_t kv_h = h / group_size;
      const T *Qbh = pQ + b * q_batch_stride + h * q_head_stride;
      const T *Kbh = pK + b * k_batch_stride + kv_h * k_head_stride;
      T *Sbh = pProbs + b * probs_batch_stride + h * probs_head_stride;

      for (int64_t i = 0; i < q_seq_len; ++i) {
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          double s = 0.0;
          for (int64_t d = 0; d < head_size; ++d) {
            s += static_cast<double>(Qbh[i * head_size + d]) *
                 static_cast<double>(Kbh[j * head_size + d]);
          }
          s *= static_cast<double>(scale);
          Sbh[i * kv_seq_len + j] = static_cast<T>(s);
        }
      }
    }
  }

  // Apply the optional ``score_mod`` modifier subgraph callback. The
  // callback may freely rewrite the score values but must preserve the
  // element type and the (B, Hq, L, S) shape.
  if (score_mod) {
    score_mod(probs);
    EXT_ENFORCE_INVALID(probs.data_type == kElementType,
                        "kernel::FlexAttention: 'score_mod' callback must preserve the "
                        "element type of the score tensor.");
    EXT_ENFORCE_INVALID(probs.shape == probs_shape,
                        "kernel::FlexAttention: 'score_mod' callback must preserve the "
                        "(batch_size, q_num_heads, q_seq_len, kv_seq_len) shape of the "
                        "score tensor.");
    EXT_ENFORCE_INVALID(
        probs.size_bytes() == static_cast<size_t>(probs_count) * sizeof(T),
        "kernel::FlexAttention: 'score_mod' callback must preserve the byte size of the "
        "score tensor buffer.");
    pProbs = probs.As<T>();
  }

  // Phase 2: softmax over the last (kv_seq_len) axis, per (b, h, i) row.
  // Handle the all-``-inf`` row produced by an exhaustively masked
  // position by leaving the probabilities at zero, matching ONNX's
  // reference semantics.
  detail::TemporaryTypedBuffer<double> row_buf(static_cast<size_t>(kv_seq_len), allocator,
                                               "kernel::FlexAttention softmax row");
  double *row = row_buf.data();
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < q_num_heads; ++h) {
      T *Pbh = pProbs + b * probs_batch_stride + h * probs_head_stride;
      for (int64_t i = 0; i < q_seq_len; ++i) {
        double max_score = static_cast<double>(Pbh[i * kv_seq_len + 0]);
        for (int64_t j = 1; j < kv_seq_len; ++j) {
          const double s = static_cast<double>(Pbh[i * kv_seq_len + j]);
          if (s > max_score) {
            max_score = s;
          }
        }
        double denom = 0.0;
        if (std::isfinite(max_score)) {
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            const double e = std::exp(static_cast<double>(Pbh[i * kv_seq_len + j]) - max_score);
            row[static_cast<size_t>(j)] = e;
            denom += e;
          }
        } else {
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            row[static_cast<size_t>(j)] = 0.0;
          }
        }
        const double inv_denom = denom != 0.0 ? 1.0 / denom : 0.0;
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          Pbh[i * kv_seq_len + j] = static_cast<T>(row[static_cast<size_t>(j)] * inv_denom);
        }
      }
    }
  }

  // Apply the optional ``prob_mod`` modifier subgraph callback. The
  // callback may freely rewrite the probability values but must preserve
  // the element type and the (B, Hq, L, S) shape.
  if (prob_mod) {
    prob_mod(probs);
    EXT_ENFORCE_INVALID(probs.data_type == kElementType,
                        "kernel::FlexAttention: 'prob_mod' callback must preserve the "
                        "element type of the probability tensor.");
    EXT_ENFORCE_INVALID(probs.shape == probs_shape,
                        "kernel::FlexAttention: 'prob_mod' callback must preserve the "
                        "(batch_size, q_num_heads, q_seq_len, kv_seq_len) shape of the "
                        "probability tensor.");
    EXT_ENFORCE_INVALID(
        probs.size_bytes() == static_cast<size_t>(probs_count) * sizeof(T),
        "kernel::FlexAttention: 'prob_mod' callback must preserve the byte size of the "
        "probability tensor buffer.");
    pProbs = probs.As<T>();
  }

  // Phase 3: Y = probs @ V, per (batch, query-head, query-pos, value-dim).
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < q_num_heads; ++h) {
      const int64_t kv_h = h / group_size;
      const T *Vbh = pV + b * v_batch_stride + kv_h * v_head_stride;
      const T *Pbh = pProbs + b * probs_batch_stride + h * probs_head_stride;
      T *Ybh = pY + b * y_batch_stride + h * y_head_stride;
      for (int64_t i = 0; i < q_seq_len; ++i) {
        for (int64_t dv = 0; dv < v_head_size; ++dv) {
          double y = 0.0;
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            y += static_cast<double>(Pbh[i * kv_seq_len + j]) *
                 static_cast<double>(Vbh[j * v_head_size + dv]);
          }
          Ybh[i * v_head_size + dv] = static_cast<T>(y);
        }
      }
    }
  }
}

} // namespace

Tensor FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                                 RuntimeContext *rt) const {
  CheckRank4Float(Q, "Q");
  const int64_t head_size = Q.shape[3];
  EXT_ENFORCE_INVALID(head_size > 0, "kernel::FlexAttention: 'head_size' must be positive.");
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  return (*this)(Q, K, V, scale, rt);
}

Tensor FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                                 RuntimeContext *rt) const {
  return (*this)(Q, K, V, scale, ScoreModFn{}, ProbModFn{}, rt);
}

Tensor FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                                 const ProbModFn &prob_mod, RuntimeContext *rt) const {
  return (*this)(Q, K, V, scale, ScoreModFn{}, prob_mod, rt);
}

Tensor FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                                 const ScoreModFn &score_mod, const ProbModFn &prob_mod,
                                 RuntimeContext *rt) const {
  CheckRank4Float(Q, "Q");
  CheckRank4Float(V, "V");
  const int64_t batch_size = Q.shape[0];
  const int64_t q_num_heads = Q.shape[1];
  const int64_t q_seq_len = Q.shape[2];
  const int64_t v_head_size = V.shape[3];
  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  const size_t out_n_bytes = static_cast<size_t>(out_count) * ElementBytes(Q.data_type);
  Tensor out = MakeOutputTensor(Q.data_type, {batch_size, q_num_heads, q_seq_len, v_head_size},
                                out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(Q, K, V, scale, score_mod, prob_mod, out);
  return out;
}

void FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                               Tensor &output, RuntimeContext *rt) const {
  (*this)(Q, K, V, scale, ScoreModFn{}, ProbModFn{}, output, rt);
}

void FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                               const ProbModFn &prob_mod, Tensor &output,
                               RuntimeContext *rt) const {
  (*this)(Q, K, V, scale, ScoreModFn{}, prob_mod, output, rt);
}

void FlexAttention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                               const ScoreModFn &score_mod, const ProbModFn &prob_mod,
                               Tensor &output, RuntimeContext *rt) const {
  // Half-precision fast path: promote Q/K/V to FLOAT32, run the reference
  // implementation, then demote the output back. ``score_mod`` and
  // ``prob_mod`` callbacks (if any) are wrapped so they see tensors in the
  // original element type while the inner kernel operates on FLOAT32.
  if (IsHalfPrecision(Q.data_type)) {
    EXT_ENFORCE_INVALID(K.data_type == Q.data_type && V.data_type == Q.data_type,
                        "kernel::FlexAttention: 'Q', 'K', 'V' must share the same dtype.");
    EXT_ENFORCE_INVALID(output.data_type == Q.data_type,
                        "kernel::FlexAttention preallocated output must share Q's element type.");
    const int32_t target_dtype = Q.data_type;
    const Tensor Q_f = PromoteToFloat32(Q);
    const Tensor K_f = PromoteToFloat32(K);
    const Tensor V_f = PromoteToFloat32(V);

    auto wrap_callback = [target_dtype](const ScoreModFn &cb) -> ScoreModFn {
      if (!cb) {
        return ScoreModFn{};
      }
      return [cb, target_dtype](Tensor &scores_f) {
        Tensor scores_half = DemoteFromFloat32(scores_f, target_dtype);
        cb(scores_half);
        scores_f = PromoteToFloat32(scores_half);
      };
    };
    ScoreModFn score_mod_wrapped = wrap_callback(score_mod);
    ProbModFn prob_mod_wrapped = wrap_callback(prob_mod);

    const onnx_kernels::Shape out_shape = {Q.shape[0], Q.shape[1], Q.shape[2], V.shape[3]};
    const int64_t out_count = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    const size_t out_f_n_bytes = static_cast<size_t>(out_count) * sizeof(float);
    RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
    Tensor out_f = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_f_n_bytes,
                                    allocator);
    (*this)(Q_f, K_f, V_f, scale, score_mod_wrapped, prob_mod_wrapped, out_f, rt);
    Tensor demoted = DemoteFromFloat32(out_f, target_dtype);
    EXT_ENFORCE_INVALID(output.shape == demoted.shape,
                        "kernel::FlexAttention preallocated output shape must be (batch_size, "
                        "q_num_heads, q_seq_len, v_head_size).");
    EXT_ENFORCE_INVALID(output.size_bytes() == demoted.size_bytes(),
                        "kernel::FlexAttention preallocated output buffer has unexpected size "
                        "in bytes.");
    std::memcpy(output.mutable_bytes(), demoted.bytes(), demoted.size_bytes());
    return;
  }

  CheckRank4Float(Q, "Q");
  CheckRank4Float(K, "K");
  CheckRank4Float(V, "V");
  EXT_ENFORCE_INVALID(K.data_type == Q.data_type && V.data_type == Q.data_type,
                      "kernel::FlexAttention: 'Q', 'K', 'V' must share the same dtype.");

  // FLOAT and DOUBLE share the reference implementation, parameterized by the
  // floating-point storage type. (FLOAT16/BFLOAT16 are handled by the
  // promote/compute/demote fast path above.)
  if (Q.data_type == DataType::DOUBLE) {
    ComputeFlexAttentionTyped<double>(Q, K, V, scale, score_mod, prob_mod, output,
                                      rt ? rt->allocator() : nullptr);
  } else {
    ComputeFlexAttentionTyped<float>(Q, K, V, scale, score_mod, prob_mod, output,
                                     rt ? rt->allocator() : nullptr);
  }
}

void FlexAttention::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &Q = GetInput(node, 0, rt.tensors());
  const Tensor &K = GetInput(node, 1, rt.tensors());
  const Tensor &V = GetInput(node, 2, rt.tensors());

  // Resolve the scale once: use the explicit attribute if present, otherwise
  // fall back to 1/sqrt(head_size) — matching the kernel's own default.
  const float scale = FindAttribute(node, "scale") != nullptr
                          ? GetAttributeFloatOrDefault(node, "scale", 1.0f)
                          : 1.0f / std::sqrt(static_cast<float>(Q.shape[3]));

  onnx_kernels::kernel::FlexAttention flex(rt.kernel_ctx());
  Tensor Y;
  onnx_kernels::kernel::FlexAttention::ScoreModFn score_mod_fn;
  onnx_kernels::kernel::FlexAttention::ProbModFn prob_mod_fn;
  // Sessions for the score_mod / prob_mod subgraphs are built once and
  // reused for every row the FlexAttention kernel evaluates them on,
  // instead of re-resolving the subgraph's kernels on every call.
  std::unique_ptr<SubgraphSession> score_mod_session;
  std::unique_ptr<SubgraphSession> prob_mod_session;
  const AttributeProto *score_mod_attr = FindAttribute(node, "score_mod");
  if (score_mod_attr != nullptr) {
    const GraphProto &score_mod_graph = score_mod_attr->ref_g();
    EXT_ENFORCE_INVALID(!(score_mod_graph.input().empty()),
                        "RunNode: 'score_mod' subgraph must declare at least one input.");
    const std::string in_name = score_mod_graph.input()[0].name();
    score_mod_session = std::make_unique<SubgraphSession>(rt, score_mod_graph);
    score_mod_fn = [in_name, &rt, &session = *score_mod_session](Tensor &scores) {
      auto outputs = session.Run({{in_name, scores}}, rt, "score_mod");
      if (!outputs.empty()) {
        scores = std::move(outputs[0]);
      }
    };
  }
  const AttributeProto *prob_mod_attr = FindAttribute(node, "prob_mod");
  if (prob_mod_attr != nullptr) {
    const GraphProto &prob_mod_graph = prob_mod_attr->ref_g();
    EXT_ENFORCE_INVALID(!(prob_mod_graph.input().empty()),
                        "RunNode: 'prob_mod' subgraph must declare at least one input.");
    const std::string in_name = prob_mod_graph.input()[0].name();
    prob_mod_session = std::make_unique<SubgraphSession>(rt, prob_mod_graph);
    prob_mod_fn = [in_name, &rt, &session = *prob_mod_session](Tensor &probs) {
      auto outputs = session.Run({{in_name, probs}}, rt, "prob_mod");
      if (!outputs.empty()) {
        probs = std::move(outputs[0]);
      }
    };
  }
  if (score_mod_fn || prob_mod_fn) {
    Y = flex(Q, K, V, scale, score_mod_fn, prob_mod_fn);
  } else {
    Y = flex(Q, K, V, scale);
  }
  SetOutput(node, 0, std::move(Y), rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
