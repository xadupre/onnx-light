// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernel_peak_memory_helpers.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/parallel_for.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 3> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16)};

// Retains FLOAT32 precision with a separate exponent for overflowing finite-input rows.
struct ExtendedFloat {
  float mantissa = 0.0f;
  int exponent = 0;

  static ExtendedFloat Normalize(float value, int exponent = 0) {
    if (value == 0.0f)
      return {};
    int adjustment = 0;
    const float mantissa = std::frexp(value, &adjustment);
    return {mantissa, exponent + adjustment};
  }

  ExtendedFloat Add(ExtendedFloat other) const {
    if (mantissa == 0.0f)
      return other;
    if (other.mantissa == 0.0f)
      return *this;
    if (exponent < other.exponent)
      return other.Add(*this);
    return Normalize(mantissa + std::scalbn(other.mantissa, other.exponent - exponent), exponent);
  }

  ExtendedFloat Subtract(ExtendedFloat other) const {
    return Add({-other.mantissa, other.exponent});
  }

  ExtendedFloat Multiply(ExtendedFloat other) const {
    return Normalize(mantissa * other.mantissa, exponent + other.exponent);
  }

  ExtendedFloat Divide(ExtendedFloat other) const {
    return Normalize(mantissa / other.mantissa, exponent - other.exponent);
  }

  float ToFloat() const { return std::scalbn(mantissa, exponent); }
};

ExtendedFloat ExtendedDotProduct(const float *q, const float *k, int64_t width) {
  ExtendedFloat sum;
  for (int64_t d = 0; d < width; ++d)
    sum = sum.Add(ExtendedFloat::Normalize(q[d]).Multiply(ExtendedFloat::Normalize(k[d])));
  return sum;
}

float RecomputeAttentionValue(const float *probabilities, const float *values, int64_t length,
                              int64_t stride, float ordinary_result) {
  float magnitude = 0.0f;
  float weight = 0.0f;
  for (int64_t j = 0; j < length; ++j) {
    const float value = values[j * stride];
    if (!std::isfinite(value) || !std::isfinite(probabilities[j]))
      return ordinary_result;
    magnitude = std::max(magnitude, std::abs(value));
    weight += probabilities[j];
  }
  if (magnitude == 0.0f || weight == 0.0f)
    return ordinary_result;
  float scaled = 0.0f;
  for (int64_t j = 0; j < length; ++j)
    scaled += probabilities[j] * (values[j * stride] / magnitude);
  // Rounded probabilities can sum above one. Renormalization and the convex
  // bound keep a finite weighted mean finite when restoring its magnitude.
  return magnitude * std::clamp(scaled / weight, -1.0f, 1.0f);
}

template <typename Mask>
bool RecomputeCenteredAttentionScores(const float *q, const float *k, int64_t width, int64_t length,
                                      float scale, float softcap, float *scores, float *qk,
                                      int output_mode, const Mask &mask) {
  // Genuine non-finite inputs keep the ordinary IEEE propagation path.
  if (!std::isfinite(scale) || !std::isfinite(softcap))
    return false;
  for (int64_t d = 0; d < width; ++d)
    if (!std::isfinite(q[d]))
      return false;
  for (int64_t j = 0; j < length; ++j) {
    const float bias = mask(j);
    if (!std::isfinite(bias) && bias != -std::numeric_limits<float>::infinity())
      return false;
    for (int64_t d = 0; d < width; ++d)
      if (!std::isfinite(k[j * width + d]))
        return false;
  }

  const auto scaled_score = [&](int64_t j, float *output) {
    auto score =
        ExtendedDotProduct(q, k + j * width, width).Multiply(ExtendedFloat::Normalize(scale));
    if (output != nullptr && output_mode == 0)
      output[j] = score.ToFloat();
    if (softcap > 0.0f)
      score = ExtendedFloat::Normalize(
          softcap * std::tanh(score.Divide(ExtendedFloat::Normalize(softcap)).ToFloat()));
    if (output != nullptr && output_mode == 1)
      output[j] = score.ToFloat();
    return score;
  };

  ExtendedFloat maximum;
  bool has_maximum = false;
  for (int64_t j = 0; j < length; ++j) {
    auto score = scaled_score(j, qk);
    const float bias = mask(j);
    if (std::isfinite(bias)) {
      score = score.Add(ExtendedFloat::Normalize(bias));
      scores[j] = score.ToFloat();
      if (!has_maximum || score.Subtract(maximum).mantissa > 0.0f) {
        maximum = score;
        has_maximum = true;
      }
    } else {
      scores[j] = -std::numeric_limits<float>::infinity();
    }
    if (qk != nullptr && output_mode == 2)
      qk[j] = scores[j];
  }

  // Recomputes instead of allocating extended scores, and centers before narrowing.
  for (int64_t j = 0; j < length; ++j) {
    const float bias = mask(j);
    if (std::isfinite(bias))
      scores[j] =
          scaled_score(j, nullptr).Add(ExtendedFloat::Normalize(bias)).Subtract(maximum).ToFloat();
  }
  return true;
}

// Validates that ``t`` is a rank-4 tensor whose element type is supported
// by the Attention kernel (FLOAT, FLOAT16, or BFLOAT16). The caller is
// identified by ``label`` for clearer error messages.
void CheckRank4Float(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT || t.data_type == DataType::FLOAT16 ||
                          t.data_type == DataType::BFLOAT16,
                      "kernel::Attention: '", label,
                      "' must be a FLOAT, FLOAT16 or BFLOAT16 tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4, "kernel::Attention: '", label,
                      "' must be a rank-4 tensor.");
  for (int64_t d : t.shape) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::Attention: '", label, "' has a negative dimension.");
  }
}

// Promotes a rank-3 fused ``(batch, seq, num_heads * head_size)`` FLOAT tensor
// into the rank-4 layout ``(batch, num_heads, seq, head_size)`` used by the
// internal kernel. Returns the promoted tensor; ``num_heads`` must divide
// ``hidden_size``.
Tensor PromoteRank3(const Tensor &t, int64_t num_heads, const char *label,
                    RuntimeContext *rt = nullptr) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, "kernel::Attention: '", label,
                      "' must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(t.shape.size() == 3, "kernel::Attention: '", label, "' must be rank-3.");
  const int64_t batch = t.shape[0];
  const int64_t seq = t.shape[1];
  const int64_t hidden = t.shape[2];
  EXT_ENFORCE_INVALID(num_heads > 0, "kernel::Attention: '", label,
                      "' needs a positive ``num_heads`` to promote rank-3.");
  EXT_ENFORCE_INVALID(hidden % num_heads == 0, "kernel::Attention: '", label,
                      "' hidden_size must be a multiple of ``num_heads``.");
  const int64_t head_size = hidden / num_heads;
  const size_t out_n_bytes = t.size_bytes();
  Tensor out =
      rt ? rt->MakeTemporaryTensor(DataType::FLOAT, {batch, num_heads, seq, head_size}, out_n_bytes)
         : MakeOutputTensor(DataType::FLOAT, {batch, num_heads, seq, head_size}, out_n_bytes,
                            nullptr);
  const float *src = t.AsFloat();
  float *dst = out.AsFloat();
  // src strides: (seq*hidden, hidden, 1) over (batch, seq, hidden).
  // hidden indexed as (h * head_size + d).
  // dst strides: (num_heads*seq*head_size, seq*head_size, head_size, 1)
  // over (batch, h, seq, d).
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t d = 0; d < head_size; ++d) {
          dst[((b * num_heads + h) * seq + s) * head_size + d] =
              src[(b * seq + s) * hidden + h * head_size + d];
        }
      }
    }
  }
  return out;
}

// Inverse of ``PromoteRank3``: collapses a rank-4 ``(batch, num_heads, seq,
// head_size)`` tensor into the rank-3 fused layout
// ``(batch, seq, num_heads * head_size)``.
Tensor CollapseToRank3(const Tensor &t, RuntimeContext *rt = nullptr) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, "kernel::Attention: must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 4, "kernel::Attention: must be rank-4.");
  const int64_t batch = t.shape[0];
  const int64_t num_heads = t.shape[1];
  const int64_t seq = t.shape[2];
  const int64_t head_size = t.shape[3];
  const int64_t hidden = num_heads * head_size;
  const size_t out_n_bytes = t.size_bytes();
  Tensor out = rt ? rt->MakeOutputTensor(0, DataType::FLOAT, {batch, seq, hidden}, out_n_bytes)
                  : MakeOutputTensor(DataType::FLOAT, {batch, seq, hidden}, out_n_bytes, nullptr);
  const float *src = t.AsFloat();
  float *dst = out.AsFloat();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < num_heads; ++h) {
      for (int64_t s = 0; s < seq; ++s) {
        for (int64_t d = 0; d < head_size; ++d) {
          dst[(b * seq + s) * hidden + h * head_size + d] =
              src[((b * num_heads + h) * seq + s) * head_size + d];
        }
      }
    }
  }
  return out;
}

// Concatenates two rank-4 FLOAT tensors along axis 2 (the sequence axis).
// Returns a new tensor whose shape equals ``a.shape`` with axis 2 = ``a[2] +
// b[2]``; axes 0, 1, 3 must match between ``a`` and ``b``.
Tensor AllocateResult(RuntimeContext *rt, int output_slot, int32_t data_type, const Shape &shape,
                      size_t n_bytes) {
  if (rt == nullptr) {
    return MakeOutputTensor(data_type, shape, n_bytes, nullptr);
  }
  const auto &roles = rt->output_slot_io_roles();
  if (!roles.empty() && static_cast<size_t>(output_slot) >= roles.size()) {
    return rt->MakeTemporaryTensor(data_type, shape, n_bytes);
  }
  return rt->MakeOutputTensor(output_slot, data_type, shape, n_bytes);
}

bool HasRecordedOutputSlot(const RuntimeContext *rt, int output_slot) {
  return rt != nullptr && !rt->output_slot_io_roles().empty() && output_slot >= 0 &&
         static_cast<size_t>(output_slot) < rt->output_slot_io_roles().size();
}

size_t FloatCacheBytes(const Shape &shape) {
  const int64_t count = shape.product(0, shape.size(), "Attention cache");
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(count) <=
                          std::numeric_limits<size_t>::max() / sizeof(float),
                      "kernel::Attention: cache byte size overflow.");
  return static_cast<size_t>(count) * sizeof(float);
}

Tensor ConcatAxis2(const Tensor &a, const Tensor &b, int output_slot, RuntimeContext *rt = nullptr,
                   bool allow_capacity = true, bool declared_output = true) {
  EXT_ENFORCE_INVALID(a.shape.size() == 4 && b.shape.size() == 4,
                      "kernel::Attention: concat inputs must be rank-4.");
  EXT_ENFORCE_INVALID(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] &&
                          a.shape[3] == b.shape[3],
                      "kernel::Attention: past KV shape must match current K/V on axes 0, 1, 3.");
  const int64_t batch = a.shape[0];
  const int64_t heads = a.shape[1];
  const int64_t la = a.shape[2];
  const int64_t lb = b.shape[2];
  const int64_t d = a.shape[3];
  EXT_ENFORCE_INVALID(la >= 0 && lb >= 0 && la <= std::numeric_limits<int64_t>::max() - lb,
                      "kernel::Attention: cache sequence length overflow.");
  const int64_t lc = la + lb;
  const Shape shape{batch, heads, lc, d};
  const size_t out_n_bytes = FloatCacheBytes(shape);
  const size_t prefix = FloatCacheBytes(a.shape);
  const size_t appended = FloatCacheBytes(b.shape);
  EXT_ENFORCE_INVALID(a.size_bytes() == prefix && b.size_bytes() == appended,
                      "kernel::Attention: cache tensor byte extent mismatch.");
  EXT_ENFORCE_INVALID((prefix == 0 || a.bytes() != nullptr) &&
                          (appended == 0 || b.bytes() != nullptr),
                      "kernel::Attention: cache tensor has a null data pointer.");
  if (rt != nullptr && allow_capacity) {
    auto reservation = rt->ReservePersistentAppend(a, shape, 2, output_slot + 3, output_slot);
    if (reservation) {
      const auto destination = reservation->writable_bytes();
      EXT_ENFORCE_INVALID(destination.size() == appended,
                          "kernel::Attention: reserved append byte extent mismatch.");
      if (appended != 0)
        std::memmove(destination.data(), b.bytes(), appended);
      if (rt->events_enabled())
        rt->RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                         .storage_append_copied_bytes = appended});
      return rt->CommitPersistentAppend(output_slot, std::move(*reservation), appended);
    }
  }
  // Multiple dense batch/head slices change stride when the sequence grows.
  // They must be repacked rather than treated as a contiguous append buffer.
  Tensor out =
      rt != nullptr && !declared_output
          ? rt->MakeTemporaryTensor(DataType::FLOAT, {batch, heads, lc, d}, out_n_bytes)
          : AllocateResult(rt, output_slot, DataType::FLOAT, {batch, heads, lc, d}, out_n_bytes);
  if (rt != nullptr && rt->events_enabled())
    rt->RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                     .storage_allocations = 1,
                     .storage_allocated_bytes = out_n_bytes,
                     .storage_prefix_copied_bytes = prefix,
                     .storage_append_copied_bytes = appended});
  if (out_n_bytes == 0)
    return out;
  const float *pa = a.AsFloat();
  const float *pb = b.AsFloat();
  float *po = out.AsFloat();
  for (int64_t bi = 0; bi < batch; ++bi) {
    for (int64_t h = 0; h < heads; ++h) {
      // copy ``a`` slice
      const int64_t off_a = (bi * heads + h) * la * d;
      const int64_t off_o = (bi * heads + h) * lc * d;
      if (la != 0)
        std::copy(pa + off_a, pa + off_a + la * d, po + off_o);
      const int64_t off_b = (bi * heads + h) * lb * d;
      if (lb != 0)
        std::copy(pb + off_b, pb + off_b + lb * d, po + off_o + la * d);
    }
  }
  return out;
}

Tensor CopyOutput(const Tensor &src, int output_slot, RuntimeContext *rt) {
  Tensor out = AllocateResult(rt, output_slot, src.data_type, src.shape, src.size_bytes());
  if (rt != nullptr && rt->events_enabled())
    rt->RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                     .storage_allocations = 1,
                     .storage_allocated_bytes = src.size_bytes(),
                     .storage_append_copied_bytes = src.size_bytes()});
  if (src.size_bytes() != 0) {
    std::memcpy(out.mutable_bytes(), src.bytes(), src.size_bytes());
  }
  return out;
}

void ValidateAttentionMask(const Tensor *mask, int64_t batch_size, int64_t q_num_heads,
                           int64_t q_seq_len) {
  if (mask == nullptr) {
    return;
  }
  EXT_ENFORCE_INVALID(mask->data_type == DataType::FLOAT || mask->data_type == DataType::BOOL,
                      "kernel::Attention: 'attn_mask' must be FLOAT or BOOL.");
  const int rank = static_cast<int>(mask->shape.size());
  EXT_ENFORCE_INVALID(rank >= 1 && rank <= 4,
                      "kernel::Attention: 'attn_mask' must have rank between 1 and 4.");
  int64_t shape4[4] = {1, 1, 1, 1};
  for (int k = 0; k < rank; ++k) {
    const int64_t dim = mask->shape[static_cast<size_t>(k)];
    EXT_ENFORCE_INVALID(dim >= 0, "kernel::Attention: 'attn_mask' has a negative dimension.");
    shape4[4 - rank + k] = dim;
  }
  const int64_t out_dims[3] = {batch_size, q_num_heads, q_seq_len};
  for (int k = 0; k < 3; ++k) {
    EXT_ENFORCE_INVALID(shape4[k] == 1 || shape4[k] == out_dims[k],
                        "kernel::Attention: 'attn_mask' is not broadcastable to (batch_size, "
                        "q_num_heads, q_seq_len); its last axis defines the mask KV length.");
  }
}

// Returns the value of a validated FLOAT or BOOL ``attn_mask`` at the
// broadcasted index ``(b, h, i, j)``.
float BroadcastedMaskValue(const Tensor &mask, int64_t b, int64_t h, int64_t i, int64_t j) {
  const int rank = static_cast<int>(mask.shape.size());
  int64_t shape4[4] = {1, 1, 1, 1};
  for (int k = 0; k < rank; ++k) {
    shape4[4 - rank + k] = mask.shape[static_cast<size_t>(k)];
  }
  const int64_t idx[4] = {b, h, i, j};
  int64_t linear = 0;
  int64_t stride = 1;
  for (int k = 3; k >= 0; --k) {
    const int64_t coord = shape4[k] == 1 ? 0 : idx[k];
    linear += coord * stride;
    stride *= shape4[k];
  }
  if (mask.data_type == DataType::FLOAT) {
    return mask.AsFloat()[linear];
  }
  if (mask.data_type == DataType::BOOL) {
    return mask.AsBool()[linear] != 0 ? 0.0f : -std::numeric_limits<float>::infinity();
  }
  return 0.0f;
}

// Returns the mask value at ``(b, h, i, j)`` allowing the mask to be
// shorter than ``total_kv_seq_len`` along its last axis: missing positions
// are treated as ``-inf`` (FLOAT) / ``false`` (BOOL), matching the upstream
// ``np.pad(..., constant_values=-inf)`` behaviour. Returns ``0.0`` when
// ``mask == nullptr``.
float MaskValuePadded(const Tensor *mask, int64_t b, int64_t h, int64_t i, int64_t j) {
  if (mask == nullptr) {
    return 0.0f;
  }
  const int rank = static_cast<int>(mask->shape.size());
  const int64_t mask_kv = mask->shape[static_cast<size_t>(rank - 1)];
  if (j >= mask_kv) {
    return -std::numeric_limits<float>::infinity();
  }
  return BroadcastedMaskValue(*mask, b, h, i, j);
}

// Core rank-4 Attention. ``Q4``/``K4``/``V4`` are already in the internal
// ``(batch, num_heads, seq, head_size)`` layout. Builds ``present_key`` /
// ``present_value`` (concatenating the optional past tensors), runs the
// scaled dot-product attention and returns a rank-4 ``Y`` together with the
// present tensors and the mode-dependent ``qk_matmul_output``.
Attention::Result ComputeAttentionRank4(const Tensor &Q4, const Tensor &K4, const Tensor &V4,
                                        const Attention::Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value,
                                        const Tensor *nonpad_kv_seqlen, RuntimeContext *rt,
                                        bool temporary_y, bool allow_capacity,
                                        int64_t parallel_minimum_elements,
                                        const std::array<bool, 4> &outputs) {
  CheckRank4Float(Q4, "Q");
  CheckRank4Float(K4, "K");
  CheckRank4Float(V4, "V");
  EXT_ENFORCE_INVALID(attrs.left_window_size >= -1,
                      "kernel::Attention: 'left_window_size' must be -1 or nonnegative.");
  EXT_ENFORCE_INVALID(attrs.right_window_size >= -1,
                      "kernel::Attention: 'right_window_size' must be -1 or nonnegative.");
  EXT_ENFORCE_INVALID((past_key == nullptr) == (past_value == nullptr),
                      "kernel::Attention: past_key and past_value must be provided together.");
  EXT_ENFORCE_INVALID(nonpad_kv_seqlen == nullptr || past_key == nullptr,
                      "kernel::Attention: nonpad_kv_seqlen cannot be combined with past cache.");

  const int64_t batch_size = Q4.shape[0];
  const int64_t q_num_heads = Q4.shape[1];
  const int64_t q_seq_len = Q4.shape[2];
  const int64_t head_size = Q4.shape[3];
  const int64_t kv_num_heads = K4.shape[1];

  EXT_ENFORCE_INVALID(K4.shape[0] == batch_size && V4.shape[0] == batch_size,
                      "kernel::Attention: Q, K, V must share the same batch size.");
  EXT_ENFORCE_INVALID(K4.shape[3] == head_size,
                      "kernel::Attention: K head_size must match Q head_size.");
  EXT_ENFORCE_INVALID(V4.shape[1] == kv_num_heads,
                      "kernel::Attention: V num_heads must match K num_heads.");
  EXT_ENFORCE_INVALID(V4.shape[2] == K4.shape[2],
                      "kernel::Attention: V kv_seq_len must match K kv_seq_len.");
  EXT_ENFORCE_INVALID(
      kv_num_heads > 0 && q_num_heads % kv_num_heads == 0,
      "kernel::Attention: q_num_heads must be a positive multiple of kv_num_heads.");

  // ----- Build present_key / present_value -------------------------------
  // The upstream operator concatenates past_key/past_value with K/V along
  // the sequence axis. When neither is supplied, present == K/V.
  Tensor present_key;
  Tensor present_value;
  if (past_key != nullptr) {
    present_key = ConcatAxis2(*past_key, K4, 1, rt, allow_capacity && outputs[1], outputs[1]);
    present_value = ConcatAxis2(*past_value, V4, 2, rt, allow_capacity && outputs[2], outputs[2]);
  } else {
    if (outputs[1])
      present_key = HasRecordedOutputSlot(rt, 1) ? CopyOutput(K4, 1, rt) : K4;
    if (outputs[2])
      present_value = HasRecordedOutputSlot(rt, 2) ? CopyOutput(V4, 2, rt) : V4;
  }
  const Tensor &effective_key = past_key != nullptr || outputs[1] ? present_key : K4;
  const Tensor &effective_value = past_value != nullptr || outputs[2] ? present_value : V4;
  const int64_t total_kv_seq_len = effective_key.shape[2];
  const int64_t past_kv_seq_len = past_key != nullptr ? past_key->shape[2] : 0;
  const int64_t v_head_size = effective_value.shape[3];

  // ----- Resolve nonpad_kv_seqlen ----------------------------------------
  // Optional 1-D INT64 tensor of length ``batch_size``. Positions
  // ``j >= nonpad_kv_seqlen[b]`` along the key/value sequence axis are
  // padding and masked out with ``-inf`` (mirroring the upstream
  // ``padding_mask`` derived from ``nonpad_kv_seqlen``).
  const int64_t *nonpad_lengths = nullptr;
  if (nonpad_kv_seqlen != nullptr && nonpad_kv_seqlen->size_bytes() != 0) {
    EXT_ENFORCE_INVALID(nonpad_kv_seqlen->data_type == DataType::INT64,
                        "kernel::Attention: 'nonpad_kv_seqlen' must be INT64.");
    EXT_ENFORCE_INVALID(
        nonpad_kv_seqlen->shape.size() == 1 && nonpad_kv_seqlen->shape[0] == batch_size,
        "kernel::Attention: 'nonpad_kv_seqlen' must be a 1-D tensor of length batch_size.");
    nonpad_lengths = nonpad_kv_seqlen->AsInt64();
  }

  // ----- Resolve scale ---------------------------------------------------
  const float scale =
      attrs.has_scale ? attrs.scale : 1.0f / std::sqrt(static_cast<float>(head_size));

  // ----- Allocate outputs ------------------------------------------------
  const int64_t out_count_y = batch_size * q_num_heads * q_seq_len * v_head_size;
  const size_t Y_n_bytes = static_cast<size_t>(out_count_y) * sizeof(float);
  const Shape y_shape{batch_size, q_num_heads, q_seq_len, v_head_size};
  Tensor Y = temporary_y && rt ? rt->MakeTemporaryTensor(DataType::FLOAT, y_shape, Y_n_bytes)
                               : AllocateResult(rt, 0, DataType::FLOAT, y_shape, Y_n_bytes);
  Tensor qk_out;
  if (outputs[3]) {
    const Shape qk_shape{batch_size, q_num_heads, q_seq_len, total_kv_seq_len};
    qk_out = AllocateResult(rt, 3, DataType::FLOAT, qk_shape, FloatCacheBytes(qk_shape));
  }

  // ----- Compute ---------------------------------------------------------
  const int64_t group_size = q_num_heads / kv_num_heads;
  const float *pQ = Q4.AsFloat();
  const float *pK = effective_key.AsFloat();
  const float *pV = effective_value.AsFloat();
  float *pY = Y.AsFloat();
  float *pQK = outputs[3] ? qk_out.AsFloat() : nullptr;

  const int64_t q_head_stride = q_seq_len * head_size;
  const int64_t q_batch_stride = q_num_heads * q_head_stride;
  const int64_t k_head_stride = total_kv_seq_len * head_size;
  const int64_t k_batch_stride = kv_num_heads * k_head_stride;
  const int64_t v_head_stride = total_kv_seq_len * v_head_size;
  const int64_t v_batch_stride = kv_num_heads * v_head_stride;
  const int64_t y_head_stride = q_seq_len * v_head_size;
  const int64_t y_batch_stride = q_num_heads * y_head_stride;
  const int64_t qk_head_stride = q_seq_len * total_kv_seq_len;
  const int64_t qk_batch_stride = q_num_heads * qk_head_stride;

  // Upstream's _compute_attention multiplies Q and K by sqrt(scale)
  // separately and then matmuls, which is numerically equivalent to a
  // single ``* scale`` after the matmul.
  // The per-row scratch buffers are acquired from the runtime allocator (when
  // one is provided) so no working memory is allocated outside it; they fall
  // back to inline storage when ``allocator`` is null.
  const int64_t task_count = batch_size * q_num_heads;
  const int64_t task_work =
      std::max<int64_t>(1, q_seq_len * total_kv_seq_len * (head_size + v_head_size));
  const int64_t task_grain =
      std::max<int64_t>(1, parallel_minimum_elements / task_work +
                               static_cast<int64_t>(parallel_minimum_elements % task_work != 0));
  ValidateAttentionMask(attn_mask, batch_size, q_num_heads, q_seq_len);
  const int64_t participants = std::max<int64_t>(1, ParallelForThreadCount());
  const bool parallel_tasks = participants > 1 && task_count > 1 && task_count >= task_grain;
  const int64_t scratch_slots =
      task_count == 0 || q_seq_len == 0
          ? 0
          : (parallel_tasks ? std::min<int64_t>(task_count, participants) : 1);
  const Shape scratch_shape{scratch_slots, total_kv_seq_len};
  const size_t scratch_n_bytes = FloatCacheBytes(scratch_shape);
  Tensor scores_buf =
      rt ? rt->MakeTemporaryTensor(DataType::FLOAT, scratch_shape, scratch_n_bytes)
         : MakeOutputTensor(DataType::FLOAT, scratch_shape, scratch_n_bytes, nullptr);
  const int64_t wave_size =
      parallel_tasks ? std::min<int64_t>(task_count, participants) : task_count;
  for (int64_t wave_begin = 0; wave_begin < task_count; wave_begin += wave_size) {
    const int64_t wave_count = std::min<int64_t>(wave_size, task_count - wave_begin);
    ParallelFor(
        wave_count, parallel_tasks ? 1 : task_grain,
        [&](int64_t begin, int64_t end) {
          for (int64_t slot = begin; slot < end; ++slot) {
            const int64_t task = wave_begin + slot;
            const int64_t scratch_slot = parallel_tasks ? slot : 0;
            float *scores = scratch_n_bytes != 0
                                ? scores_buf.AsFloat() + scratch_slot * total_kv_seq_len
                                : scores_buf.AsFloat();
            const int64_t b = task / q_num_heads;
            const int64_t h = task % q_num_heads;
            // Bottom-right / offset-aware causal frontier (mirrors onnx/onnx#8068):
            // a query at in-block index ``i`` attends key ``j`` iff ``j <= i + offset``,
            // where ``offset`` is the number of valid keys that precede this query block:
            //   * past_key present (internal cache):    offset = past_kv_seq_len
            //   * nonpad_kv_seqlen present, no past_key (external/static cache):
            //                                           offset = nonpad_kv_seqlen[b] - q_seq_len
            //   * neither:                              offset = 0 (ordinary top-left causal)
            // ``offset`` is intentionally not clamped to ``>= 0``: a negative offset
            // (out-of-contract over-long query block) fully masks the affected rows,
            // which the fully-masked-row guard below then zeroes.
            int64_t causal_offset = past_kv_seq_len;
            if (past_key == nullptr && nonpad_lengths != nullptr) {
              causal_offset = nonpad_lengths[b] - q_seq_len;
            }
            const int64_t kv_h = h / group_size;
            const float *Qbh =
                q_head_stride != 0 ? pQ + b * q_batch_stride + h * q_head_stride : pQ;
            const float *Kbh =
                k_head_stride != 0 ? pK + b * k_batch_stride + kv_h * k_head_stride : pK;
            const float *Vbh =
                v_head_stride != 0 ? pV + b * v_batch_stride + kv_h * v_head_stride : pV;
            float *Ybh = y_head_stride != 0 ? pY + b * y_batch_stride + h * y_head_stride : pY;
            float *QKbh = pQK != nullptr && qk_head_stride != 0
                              ? pQK + b * qk_batch_stride + h * qk_head_stride
                              : pQK;

            for (int64_t i = 0; i < q_seq_len; ++i) {
              bool row_fully_masked = true;
              bool needs_stable_row = false;
              float max_score = -std::numeric_limits<float>::infinity();
              const auto mask_value = [&](int64_t j) {
                float bias = MaskValuePadded(attn_mask, b, h, i, j);
                if (attrs.is_causal && j > i + causal_offset)
                  bias = -std::numeric_limits<float>::infinity();
                const int64_t window_diff = i + causal_offset - j;
                if ((attrs.left_window_size >= 0 && window_diff > attrs.left_window_size) ||
                    (attrs.right_window_size >= 0 && -window_diff > attrs.right_window_size))
                  bias = -std::numeric_limits<float>::infinity();
                if (nonpad_lengths != nullptr && j >= nonpad_lengths[b])
                  bias = -std::numeric_limits<float>::infinity();
                return bias;
              };
              for (int64_t j = 0; j < total_kv_seq_len; ++j) {
                float s = 0.0f;
                for (int64_t d = 0; d < head_size; ++d) {
                  s += Qbh[i * head_size + d] * Kbh[j * head_size + d];
                }
                s *= scale;
                needs_stable_row |= !std::isfinite(s);
                if (QKbh != nullptr && attrs.qk_matmul_output_mode == 0)
                  QKbh[i * total_kv_seq_len + j] = s;
                // Softcap precedes the additive mask so masked positions remain -inf.
                if (attrs.softcap > 0.0f)
                  s = attrs.softcap * std::tanh(s / attrs.softcap);
                if (QKbh != nullptr && attrs.qk_matmul_output_mode == 1)
                  QKbh[i * total_kv_seq_len + j] = s;

                const float b_val = mask_value(j);
                row_fully_masked &= std::isinf(b_val) && b_val < 0.0f;
                s += b_val;
                needs_stable_row |= std::isfinite(b_val) && !std::isfinite(s);
                scores[static_cast<size_t>(j)] = s;
                if (QKbh != nullptr && attrs.qk_matmul_output_mode == 2)
                  QKbh[i * total_kv_seq_len + j] = s;
                if (s > max_score) {
                  max_score = s;
                }
              }
              const bool centered =
                  needs_stable_row && RecomputeCenteredAttentionScores(
                                          head_size != 0 ? Qbh + i * head_size : Qbh, Kbh,
                                          head_size, total_kv_seq_len, scale, attrs.softcap, scores,
                                          QKbh != nullptr ? QKbh + i * total_kv_seq_len : nullptr,
                                          attrs.qk_matmul_output_mode, mask_value);
              // Softmax over last axis.
              float denom = 0.0f;
              for (int64_t j = 0; j < total_kv_seq_len; ++j) {
                const float e =
                    row_fully_masked
                        ? 0.0f
                        : std::exp(centered ? scores[static_cast<size_t>(j)]
                                            : scores[static_cast<size_t>(j)] - max_score);
                scores[static_cast<size_t>(j)] = e;
                denom += e;
              }
              const float inv_denom = denom != 0.0f ? 1.0f / denom : 0.0f;
              for (int64_t j = 0; j < total_kv_seq_len; ++j) {
                scores[static_cast<size_t>(j)] *= inv_denom;
              }
              // qk_matmul_output_mode 3: after softmax (with the fully-masked-row
              // guard applied above).
              if (QKbh != nullptr && attrs.qk_matmul_output_mode == 3) {
                for (int64_t j = 0; j < total_kv_seq_len; ++j) {
                  QKbh[i * total_kv_seq_len + j] = scores[static_cast<size_t>(j)];
                }
              }
              // Y[i, dv] = sum_j probs[j] * V[j, dv]
              for (int64_t dv = 0; dv < v_head_size; ++dv) {
                float y = 0.0f;
                for (int64_t j = 0; j < total_kv_seq_len; ++j) {
                  y += scores[static_cast<size_t>(j)] * Vbh[j * v_head_size + dv];
                }
                if (!std::isfinite(y))
                  y = RecomputeAttentionValue(scores, Vbh + dv, total_kv_seq_len, v_head_size, y);
                Ybh[i * v_head_size + dv] = y;
              }
            }
          }
        },
        "Attention");
  }

  Attention::Result r;
  r.Y = std::move(Y);
  r.present_key = std::move(present_key);
  r.present_value = std::move(present_value);
  r.qk_matmul_output = std::move(qk_out);
  return r;
}

// Rank-3 Attention. ``Q``/``K``/``V`` use the fused
// ``(batch, seq, num_heads * head_size)`` layout. The inputs are promoted to
// the rank-4 layout, delegated to ``ComputeAttentionRank4`` and the primary
// output ``Y`` is collapsed back to rank-3. The present/qk auxiliary outputs
// remain rank-4, matching the upstream operator.
Attention::Result ComputeAttentionRank3(const Tensor &Q, const Tensor &K, const Tensor &V,
                                        const Attention::Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value,
                                        const Tensor *nonpad_kv_seqlen, RuntimeContext *rt,
                                        int64_t parallel_minimum_elements,
                                        const std::array<bool, 4> &outputs) {
  Tensor Q4 = PromoteRank3(Q, attrs.q_num_heads, "Q", rt);
  Tensor K4 = PromoteRank3(K, attrs.kv_num_heads, "K", rt);
  Tensor V4 = PromoteRank3(V, attrs.kv_num_heads, "V", rt);
  Attention::Result r =
      ComputeAttentionRank4(Q4, K4, V4, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                            rt, /*temporary_y=*/true,
                            /*allow_capacity=*/false, parallel_minimum_elements, outputs);
  r.Y = CollapseToRank3(r.Y, rt);
  return r;
}

} // namespace

Attention::Attention(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Attention", kSupportedElementTypes, kPortableParallelMinimum,
                            kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(Attention)

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                             RuntimeContext *rt) const {
  CheckRank4Float(Q, "Q");
  const int64_t head_size = Q.shape[3];
  EXT_ENFORCE_INVALID(head_size > 0, "kernel::Attention: 'head_size' must be positive.");
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  return (*this)(Q, K, V, scale, rt);
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                             RuntimeContext *rt) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  return (*this)(Q, K, V, attrs, nullptr, nullptr, nullptr, nullptr, rt).Y;
}

Tensor Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                             const Tensor &attn_mask, RuntimeContext *rt) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  const Tensor *const mask_ptr =
      attn_mask.shape.empty() && attn_mask.size_bytes() == 0 ? nullptr : &attn_mask;
  return (*this)(Q, K, V, attrs, mask_ptr, nullptr, nullptr, nullptr, rt).Y;
}

void Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                           const Tensor *attn_mask, Tensor &output) const {
  Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = scale;
  Result r = (*this)(Q, K, V, attrs, attn_mask);
  EXT_ENFORCE_INVALID(output.data_type == Q.data_type,
                      "kernel::Attention preallocated output must share Q's element type.");
  EXT_ENFORCE_INVALID(output.shape == r.Y.shape,
                      "kernel::Attention preallocated output shape must be (batch_size, "
                      "q_num_heads, q_seq_len, v_head_size).");
  EXT_ENFORCE_INVALID(output.size_bytes() == r.Y.size_bytes(),
                      "kernel::Attention preallocated output buffer has unexpected size in bytes.");
  if (r.Y.size_bytes() != 0)
    std::memcpy(output.mutable_bytes(), r.Y.bytes(), r.Y.size_bytes());
}

Attention::Result Attention::operator()(const Tensor &Q, const Tensor &K, const Tensor &V,
                                        const Attributes &attrs, const Tensor *attn_mask,
                                        const Tensor *past_key, const Tensor *past_value,
                                        const Tensor *nonpad_kv_seqlen, RuntimeContext *rt) const {
  std::array<bool, 4> outputs{true, true, true, true};
  for (int slot = 1; slot < 4; ++slot) {
    if (node_ != nullptr)
      outputs[slot] = slot < node_->output_size() && !node_->output(slot).empty();
    else if (rt != nullptr && !rt->output_slot_io_roles().empty())
      outputs[slot] = HasRecordedOutputSlot(rt, slot);
  }
  // ----- Half-precision fast path ----------------------------------------
  // FLOAT16 / BFLOAT16 inputs are promoted to FLOAT32 here, the reference
  // implementation runs in float32, and the result tensors are demoted
  // back to the original element type. ``attn_mask`` may be FLOAT or BOOL
  // (or a matching half-precision dtype); BOOL masks are forwarded as-is.
  if (IsHalfPrecision(Q.data_type)) {
    EXT_ENFORCE_INVALID(K.data_type == Q.data_type && V.data_type == Q.data_type,
                        "kernel::Attention: Q, K, V must share the same dtype.");
    const int32_t target_dtype = Q.data_type;
    RuntimeContext scratch_rt = rt ? rt->MakeFunctionContext() : RuntimeContext(ctx_);
    if (rt != nullptr)
      scratch_rt.set_output_slot_io_roles(
          std::vector<bool>(rt->output_slot_io_roles().size(), false));
    if (rt && rt->events_enabled())
      scratch_rt.set_current_node_index(rt->current_node_index());
    RuntimeContext *compute_rt = rt ? &scratch_rt : nullptr;
    const Tensor Q_f = PromoteToFloat32(Q, compute_rt);
    const Tensor K_f = PromoteToFloat32(K, compute_rt);
    const Tensor V_f = PromoteToFloat32(V, compute_rt);
    Tensor attn_mask_f;
    const Tensor *attn_mask_ptr = attn_mask;
    if (attn_mask != nullptr && IsHalfPrecision(attn_mask->data_type)) {
      attn_mask_f = PromoteToFloat32(*attn_mask, compute_rt);
      attn_mask_ptr = &attn_mask_f;
    }
    Tensor past_key_f;
    const Tensor *past_key_ptr = past_key;
    if (past_key != nullptr && IsHalfPrecision(past_key->data_type)) {
      past_key_f = PromoteToFloat32(*past_key, compute_rt);
      past_key_ptr = &past_key_f;
    }
    Tensor past_value_f;
    const Tensor *past_value_ptr = past_value;
    if (past_value != nullptr && IsHalfPrecision(past_value->data_type)) {
      past_value_f = PromoteToFloat32(*past_value, compute_rt);
      past_value_ptr = &past_value_f;
    }
    Result r_f = (*this)(Q_f, K_f, V_f, attrs, attn_mask_ptr, past_key_ptr, past_value_ptr,
                         nonpad_kv_seqlen, compute_rt);
    Result r;
    constexpr int64_t kSerialDemotion = std::numeric_limits<int64_t>::max();
    auto demote_result = [&](const Tensor &value, int output_slot) {
      const bool has_recorded_roles = rt != nullptr && !rt->output_slot_io_roles().empty();
      const bool is_node_output = HasRecordedOutputSlot(rt, output_slot);
      RuntimeContext *destination_rt = !has_recorded_roles || is_node_output ? rt : compute_rt;
      const int destination_slot = is_node_output ? output_slot : 0;
      return DemoteFromFloat32(value, target_dtype, destination_rt, kSerialDemotion,
                               destination_slot);
    };
    r.Y = demote_result(r_f.Y, 0);
    if (outputs[1])
      r.present_key = demote_result(r_f.present_key, 1);
    if (outputs[2])
      r.present_value = demote_result(r_f.present_value, 2);
    if (outputs[3])
      r.qk_matmul_output = demote_result(r_f.qk_matmul_output, 3);
    return r;
  }

  // ----- Dispatch on rank -----------------------------------------------
  // Q/K/V may be provided either as rank-4 ``(batch, num_heads, seq,
  // head_size)`` tensors or as rank-3 ``(batch, seq, num_heads * head_size)``
  // packed tensors. They must all share the same rank; each case is handled
  // by its own dedicated implementation.
  EXT_ENFORCE_INVALID(Q.shape.size() == K.shape.size() && Q.shape.size() == V.shape.size(),
                      "kernel::Attention: Q, K, V must all share the same rank.");
  EXT_ENFORCE_INVALID(Q.shape.size() == 3 || Q.shape.size() == 4,
                      "kernel::Attention: Q, K, V must be rank-3 or rank-4.");
  if (Q.shape.size() == 3) {
    return ComputeAttentionRank3(Q, K, V, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                                 rt, tuning().parallel_minimum_elements, outputs);
  }
  return ComputeAttentionRank4(Q, K, V, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                               rt, false, true, tuning().parallel_minimum_elements, outputs);
}

void Attention::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 7), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 7 input(s), got ", node.input_size(),
                      ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 4), "RunNode: op '",
                      node.op_type(), "' expects between 1 and 4 output(s), got ",
                      node.output_size(), ".");
  const Tensor &q = GetInput(node, 0, rt);
  const Tensor &k = GetInput(node, 1, rt);
  const Tensor &v = GetInput(node, 2, rt);
  const Tensor *attn_mask = GetOptionalInput(node, 3, rt);
  const Tensor *past_key = GetOptionalInput(node, 4, rt);
  const Tensor *past_value = GetOptionalInput(node, 5, rt);
  const Tensor *nonpad_kv_seqlen = GetOptionalInput(node, 6, rt);

  onnx_kernels::kernel::Attention::Attributes attrs;
  if (FindAttribute(node, "scale") != nullptr) {
    attrs.has_scale = true;
    attrs.scale = GetAttributeFloatOrDefault(node, "scale", 0.0f);
  }
  attrs.is_causal = GetAttributeIntOrDefault(node, "is_causal", 0) != 0;
  attrs.softcap = GetAttributeFloatOrDefault(node, "softcap", 0.0f);
  attrs.qk_matmul_output_mode =
      static_cast<int>(GetAttributeIntOrDefault(node, "qk_matmul_output_mode", 0));
  attrs.q_num_heads = GetAttributeIntOrDefault(node, "q_num_heads", 0);
  attrs.kv_num_heads = GetAttributeIntOrDefault(node, "kv_num_heads", 0);
  attrs.left_window_size = GetAttributeIntOrDefault(node, "left_window_size", -1);
  attrs.right_window_size = GetAttributeIntOrDefault(node, "right_window_size", -1);

  onnx_kernels::kernel::Attention::Result result =
      (*this)(q, k, v, attrs, attn_mask, past_key, past_value, nonpad_kv_seqlen, &rt);
  SetOutput(node, 0, std::move(result.Y), rt);

  auto set_optional_output = [&node, &rt](int index, Tensor output) {
    if (index >= node.output_size()) {
      return;
    }
    const std::string &name = node.output(index);
    if (name.empty()) {
      return;
    }
    output.name = name;
    rt.Put(name, std::move(output), RuntimeEventKind::kIntermediate);
  };
  set_optional_output(1, std::move(result.present_key));
  set_optional_output(2, std::move(result.present_value));
  set_optional_output(3, std::move(result.qk_matmul_output));
}

core::shapes::PeakMemoryEstimate
ComputePeakMemoryAttention(Device device, const std::vector<SymShape> &input_shapes) {
  return ComputePeakMemoryAttention(device, NodeProto{}, input_shapes);
}

core::shapes::PeakMemoryEstimate
ComputePeakMemoryAttention(Device, const NodeProto &node,
                           const std::vector<SymShape> &input_shapes) {
  if (input_shapes.size() < 3) {
    return core::shapes::UnknownPeakMemory("", "Attention");
  }
  for (int slot = 3; slot < std::min(node.input_size(), 6); ++slot) {
    if (!node.input(slot).empty() &&
        (static_cast<size_t>(slot) >= input_shapes.size() || input_shapes[slot].Empty()))
      return core::shapes::UnknownPeakMemory("", "Attention");
  }
  const SymShape &q_shape = input_shapes[0];
  const SymShape &k_shape = input_shapes[1];
  const bool packed = q_shape.Rank() == 3;
  if ((!packed && q_shape.Rank() != 4) || k_shape.Rank() != q_shape.Rank() ||
      input_shapes[2].Rank() != q_shape.Rank()) {
    return core::shapes::UnknownPeakMemory("", "Attention");
  }

  SymDim q_num_heads = packed ? SymDim(int64_t{0}) : q_shape[1];
  int64_t packed_kv_heads = 1;
  if (packed) {
    int64_t heads = 0, kv_heads = 0;
    bool has_q = false, has_kv = false;
    for (const auto &attr : node.attribute()) {
      if (attr.name() == "q_num_heads") {
        heads = attr.i();
        has_q = true;
      } else if (attr.name() == "kv_num_heads") {
        kv_heads = attr.i();
        has_kv = true;
      }
    }
    if (!has_q || !has_kv)
      return core::shapes::UnknownPeakMemory("", "Attention");
    EXT_ENFORCE_INVALID(heads > 0 && kv_heads > 0,
                        "Attention scratch: q_num_heads and kv_num_heads must be positive.");
    EXT_ENFORCE_INVALID(heads % kv_heads == 0,
                        "Attention scratch: q_num_heads must be a multiple of kv_num_heads.");
    q_num_heads = SymDim(heads);
    packed_kv_heads = kv_heads;
  }

  const SymDim &batch = q_shape[0];
  const SymDim &q_seq_len = q_shape[packed ? 1 : 2];
  const SymDim &kv_seq_len = k_shape[packed ? 1 : 2];
  const auto unresolved_dimension = [](const SymDim &dim) {
    return dim.IsExpr() && (dim.AsExpr().empty() || dim.AsExpr() == "?");
  };
  const auto validate_dimension = [](const char *name, const SymDim &dim) {
    if (dim.IsInt()) {
      const int64_t value = dim.AsInt();
      EXT_ENFORCE_INVALID(value >= 0, "Negative Attention scratch dimension '", name, "': ", value,
                          ".");
    }
  };
  validate_dimension("batch", batch);
  validate_dimension("q_num_heads", q_num_heads);
  validate_dimension("q_seq_len", q_seq_len);
  validate_dimension("kv_seq_len", kv_seq_len);
  SymDim past_seq_len(int64_t{0});
  bool has_past = false;
  for (std::size_t i : {std::size_t{4}, std::size_t{5}}) {
    // Shape-only callers have no input names; empty shapes denote omitted inputs.
    const bool present =
        node.input_size() == 0
            ? i < input_shapes.size() && input_shapes[i].Rank() != 0
            : i < static_cast<std::size_t>(node.input_size()) && !node.input(i).empty();
    if (!present)
      continue;
    if (i >= input_shapes.size() || input_shapes[i].Rank() != 4)
      return core::shapes::UnknownPeakMemory("", "Attention");
    const SymDim &length = input_shapes[i][2];
    validate_dimension("past_seq_len", length);
    if (has_past && past_seq_len.IsInt() && length.IsInt())
      EXT_ENFORCE_INVALID(past_seq_len.AsInt() == length.AsInt(),
                          "Attention scratch: past key/value sequence lengths must match.");
    if (!has_past) {
      past_seq_len = length;
    } else if (unresolved_dimension(past_seq_len) || unresolved_dimension(length)) {
      past_seq_len = SymDim("?");
    } else if (past_seq_len != length) {
      // Keeps both symbolic bounds until their equality is known.
      past_seq_len = core::symbolic::FromDimType(core::expressions::dim_max(
          core::symbolic::ToDimType(past_seq_len), core::symbolic::ToDimType(length)));
    }
    has_past = true;
  }
  SymDim total_seq_len("?");
  if (!unresolved_dimension(kv_seq_len) && !unresolved_dimension(past_seq_len)) {
    if (kv_seq_len.IsInt() && past_seq_len.IsInt()) {
      total_seq_len =
          SymDim(core::shapes::CheckedDimensionAdd(kv_seq_len.AsInt(), past_seq_len.AsInt()));
    } else {
      total_seq_len = core::symbolic::FromDimType(core::expressions::dim_add(
          core::symbolic::ToDimType(kv_seq_len), core::symbolic::ToDimType(past_seq_len)));
    }
  }
  const SymDim *dimensions[] = {&batch, &q_num_heads, &q_seq_len, &total_seq_len};
  bool unresolved = false;
  for (const SymDim *dimension : dimensions) {
    const SymDim &dim = *dimension;
    unresolved |= unresolved_dimension(dim);
  }
  const auto present_input = [&](size_t slot) {
    return slot < input_shapes.size() &&
           (slot < 3 || (node.input_size() == 0 ? !input_shapes[slot].Empty()
                                                : slot < static_cast<size_t>(node.input_size()) &&
                                                      !node.input(static_cast<int>(slot)).empty()));
  };
  for (size_t slot = 0; slot < input_shapes.size(); ++slot) {
    if (!present_input(slot))
      continue;
    for (const auto &dim : input_shapes[slot].Dims()) {
      validate_dimension("input", dim);
      unresolved |= unresolved_dimension(dim);
    }
  }
  if (unresolved)
    return core::shapes::UnknownPeakMemory("", "Attention");

  using namespace core::expressions;
  using core::symbolic::ToDimType;
  const auto elements = [&](size_t slot) -> DimType {
    if (!present_input(slot))
      return int64_t{0};
    std::vector<DimType> factors;
    for (const auto &dim : input_shapes[slot].Dims())
      factors.push_back(ToDimType(dim));
    return dim_multi_mul(factors);
  };
  const auto has_output = [&](int slot) {
    return node.output_size() == 0 || (slot < node.output_size() && !node.output(slot).empty());
  };
  const DimType b = ToDimType(batch);
  const DimType h = ToDimType(q_num_heads);
  const DimType tq = ToDimType(q_seq_len);
  const DimType length = ToDimType(total_seq_len);
  const auto &v_shape = input_shapes[2];
  const DimType kv_heads = packed ? DimType{int64_t{1}} : ToDimType(k_shape[1]);
  const DimType value_width = packed
                                  ? dim_exact_div(ToDimType(v_shape[2]), DimType{packed_kv_heads})
                                  : ToDimType(v_shape[3]);
  DimType input_elements = int64_t{0};
  for (size_t slot = 0; slot < std::min(size_t{6}, input_shapes.size()); ++slot)
    input_elements = dim_add(input_elements, elements(slot));
  const DimType qkv = dim_add(dim_add(elements(0), elements(1)), elements(2));
  const DimType y = dim_multi_mul({b, h, tq, value_width});

  // Shapes alone cannot select a dtype. Bound FLOAT16/BFLOAT16 promotions
  // and FLOAT32 intermediate results; declared final outputs are excluded.
  DimType bytes = dim_mul(int64_t{4}, dim_add(input_elements, y));
  if (packed)
    bytes = dim_add(bytes, dim_mul(int64_t{4}, dim_add(qkv, y)));
  for (int slot : {1, 2}) {
    if (has_past || has_output(slot)) {
      const DimType width = packed ? ToDimType(input_shapes[slot][2])
                                   : dim_mul(kv_heads, ToDimType(input_shapes[slot][3]));
      bytes = dim_add(bytes, dim_multi_mul({int64_t{4}, b, length, width}));
    }
  }
  if (has_output(3))
    bytes = dim_add(bytes, dim_multi_mul({int64_t{4}, b, h, tq, length}));
  if ((!batch.IsInt() || batch.AsInt() != 0) &&
      (!q_num_heads.IsInt() || q_num_heads.AsInt() != 0) &&
      (!q_seq_len.IsInt() || q_seq_len.AsInt() != 0))
    bytes = dim_add(bytes, dim_multi_mul({int64_t{4}, b, h, length}));
  const SymDim maximum = core::symbolic::FromDimType(bytes);
  if (maximum.IsInt() && maximum.AsInt() == 0)
    return core::shapes::PeakMemoryEstimate::Exact(maximum);
  return {maximum, std::nullopt};
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
