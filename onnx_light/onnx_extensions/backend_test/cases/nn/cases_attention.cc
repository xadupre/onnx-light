// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Local helpers used to keep the test-case declarations terse. Every helper
// only manipulates a ``NodeProto``; the actual computations are delegated to
// the ``kernel::Attention`` reference implementation so the expected outputs
// remain self-consistent with this library's kernel.

void AddInt(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *a = node.add_attribute();
  a->set_name(name);
  a->set_type(AttributeProto::AttributeType::INT);
  a->set_i(value);
}

void AddFloat(NodeProto &node, const char *name, float value) {
  AttributeProto *a = node.add_attribute();
  a->set_name(name);
  a->set_type(AttributeProto::AttributeType::FLOAT);
  a->set_f(value);
}

// Builds a base ``Attention`` node template with the requested IO names. The
// ``inputs`` argument lists every position (entries equal to the empty
// string are still added as placeholders, matching upstream's convention
// for skipping an optional input). Same for ``outputs``.
NodeProto MakeAttentionNode(const std::vector<std::string> &inputs,
                            const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("Attention");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

// IEEE-754 binary16 ↔ binary32 conversions and the ``FloatToFloat16Tensor``
// / ``RoundToFloat16`` helpers are provided by
// ``onnx_core/runtime/kernels/cast_helper.h`` as
// ``FloatToFloat16Tensor`` / ``RoundToFloat16``.

// Builds a small deterministic FLOAT tensor of the requested shape. Values
// are derived from a simple LCG seeded by ``seed`` and then mapped into
// ``[lo, hi]`` so the generated data covers a range where FP16 rounding is
// well-behaved.
Tensor MakeDeterministicFloatTensor(const std::vector<int64_t> &shape, uint32_t seed, float lo,
                                    float hi) {
  int64_t n = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, "MakeDeterministicFloatTensor: negative dimension.");
    n *= d;
  }
  std::vector<float> values(static_cast<size_t>(n));
  uint32_t s = seed;
  for (int64_t i = 0; i < n; ++i) {
    // Numerical Recipes LCG; produces a deterministic pseudo-uniform stream.
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s & 0x00ffffffu) / static_cast<float>(0x01000000u);
    values[static_cast<size_t>(i)] = lo + (hi - lo) * u;
  }
  return Tensor::FromFloat("", shape, values);
}

Tensor MakeConstantFloatTensor(const std::vector<int64_t> &shape, float value) {
  int64_t count = 1;
  for (int64_t dimension : shape) {
    count *= dimension;
  }
  return Tensor::FromFloat("", shape, std::vector<float>(static_cast<size_t>(count), value));
}

Tensor MakePositionValueTensor4(int64_t batch, int64_t heads, int64_t sequence_length,
                                int64_t head_size, int64_t position_offset = 0) {
  std::vector<float> values(static_cast<size_t>(batch * heads * sequence_length * head_size));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t s = 0; s < sequence_length; ++s) {
        for (int64_t d = 0; d < head_size; ++d) {
          values[static_cast<size_t>(((b * heads + h) * sequence_length + s) * head_size + d)] =
              static_cast<float>(100 * h + 10 * d + position_offset + s);
        }
      }
    }
  }
  return Tensor::FromFloat("", {batch, heads, sequence_length, head_size}, values);
}

Tensor MakePositionValueTensor3(int64_t batch, int64_t heads, int64_t sequence_length,
                                int64_t head_size) {
  std::vector<float> values(static_cast<size_t>(batch * sequence_length * heads * head_size));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < sequence_length; ++s) {
      for (int64_t h = 0; h < heads; ++h) {
        for (int64_t d = 0; d < head_size; ++d) {
          values[static_cast<size_t>((b * sequence_length + s) * heads * head_size + h * head_size +
                                     d)] = static_cast<float>(100 * h + 10 * d + s);
        }
      }
    }
  }
  return Tensor::FromFloat("", {batch, sequence_length, heads * head_size}, values);
}

using WindowMaskPredicate = std::function<bool(int64_t, int64_t, int64_t, int64_t)>;

struct UniformWindowReference {
  Tensor Y;
  Tensor probabilities;
};

UniformWindowReference MakeUniformWindowReference4(
    int64_t batch, int64_t query_heads, int64_t kv_heads, int64_t query_length, int64_t kv_length,
    int64_t value_head_size, int64_t left_window_size, int64_t right_window_size, bool is_causal,
    const std::vector<int64_t> &offsets, const std::vector<int64_t> &valid_lengths,
    const WindowMaskPredicate &mask_allows) {
  std::vector<float> y_values(
      static_cast<size_t>(batch * query_heads * query_length * value_head_size), 0.0f);
  std::vector<float> probability_values(
      static_cast<size_t>(batch * query_heads * query_length * kv_length), 0.0f);
  const int64_t group_size = query_heads / kv_heads;
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t offset = offsets.empty() ? 0 : offsets[static_cast<size_t>(b)];
    const int64_t valid_length =
        valid_lengths.empty() ? kv_length : valid_lengths[static_cast<size_t>(b)];
    for (int64_t h = 0; h < query_heads; ++h) {
      const int64_t kv_head = h / group_size;
      for (int64_t i = 0; i < query_length; ++i) {
        std::vector<int64_t> allowed;
        for (int64_t j = 0; j < kv_length; ++j) {
          const int64_t difference = i + offset - j;
          if (j >= valid_length || (is_causal && difference < 0) ||
              (left_window_size >= 0 && difference > left_window_size) ||
              (right_window_size >= 0 && -difference > right_window_size) ||
              (mask_allows && !mask_allows(b, h, i, j))) {
            continue;
          }
          allowed.push_back(j);
        }
        if (allowed.empty()) {
          continue;
        }
        const float probability = 1.0f / static_cast<float>(allowed.size());
        for (int64_t j : allowed) {
          probability_values[static_cast<size_t>(
              ((b * query_heads + h) * query_length + i) * kv_length + j)] = probability;
        }
        for (int64_t d = 0; d < value_head_size; ++d) {
          float sum = 0.0f;
          for (int64_t j : allowed) {
            sum += static_cast<float>(100 * kv_head + 10 * d + j);
          }
          y_values[static_cast<size_t>(
              ((b * query_heads + h) * query_length + i) * value_head_size + d)] =
              sum / static_cast<float>(allowed.size());
        }
      }
    }
  }
  return {Tensor::FromFloat("", {batch, query_heads, query_length, value_head_size}, y_values),
          Tensor::FromFloat("", {batch, query_heads, query_length, kv_length}, probability_values)};
}

Tensor CollapseReferenceToRank3(const Tensor &rank4) {
  const int64_t batch = rank4.shape[0];
  const int64_t heads = rank4.shape[1];
  const int64_t sequence_length = rank4.shape[2];
  const int64_t head_size = rank4.shape[3];
  std::vector<float> values(static_cast<size_t>(rank4.element_count()));
  const float *source = rank4.AsFloat();
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t s = 0; s < sequence_length; ++s) {
        for (int64_t d = 0; d < head_size; ++d) {
          values[static_cast<size_t>((b * sequence_length + s) * heads * head_size + h * head_size +
                                     d)] =
              source[((b * heads + h) * sequence_length + s) * head_size + d];
        }
      }
    }
  }
  return Tensor::FromFloat("", {batch, sequence_length, heads * head_size}, values);
}

// ---- Deterministic small input tensors --------------------------------
// All test cases reuse the same handful of tiny tensors. Inputs were
// chosen so every kernel feature (GQA, causal, masks, softcap, past KV,
// rank-3 fused layout) can be exercised with shapes large enough to
// distinguish the variants but small enough to keep this file compact.

Tensor MakeQ_1_2_2_2() {
  // (batch=1, q_heads=2, q_seq=2, head_size=2)
  return Tensor::FromFloat("", {1, 2, 2, 2},
                           {
                               1.0f, 0.0f, // head 0, q0
                               0.0f, 1.0f, // head 0, q1
                               0.5f, 0.5f, // head 1, q0
                               1.0f, -1.0f // head 1, q1
                           });
}

Tensor MakeK_1_2_3_2() {
  // (batch=1, kv_heads=2, kv_seq=3, head_size=2)
  return Tensor::FromFloat("", {1, 2, 3, 2},
                           {
                               1.0f,
                               0.0f,
                               0.5f,
                               0.5f,
                               0.0f,
                               1.0f, // head 0
                               -1.0f,
                               1.0f,
                               1.0f,
                               1.0f,
                               0.25f,
                               -0.5f, // head 1
                           });
}

Tensor MakeV_1_2_3_2() {
  // (batch=1, kv_heads=2, kv_seq=3, v_head_size=2)
  return Tensor::FromFloat("", {1, 2, 3, 2},
                           {
                               1.0f,
                               0.0f,
                               0.0f,
                               1.0f,
                               -1.0f,
                               1.0f, // head 0
                               2.0f,
                               -2.0f,
                               0.5f,
                               0.25f,
                               -0.5f,
                               0.0f, // head 1
                           });
}

Tensor MakeV_1_2_3_3() {
  // (batch=1, kv_heads=2, kv_seq=3, v_head_size=3) — for diff_head_sizes.
  return Tensor::FromFloat("", {1, 2, 3, 3},
                           {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 2.0f, -1.0f, 1.0f, 0.5f, 2.0f, -2.0f,
                            1.0f, 0.5f, 0.25f, -0.25f, -0.5f, 0.0f, 1.0f});
}

Tensor MakeQ_1_2_2_2_basic() {
  return Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
}

Tensor MakeKV_basic_K() {
  return Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
}

Tensor MakeKV_basic_V() {
  return Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, 1.0f});
}

Tensor MakeQ_1_4_2_2_gqa() {
  return Tensor::FromFloat("", {1, 4, 2, 2},
                           {0.1f, 0.2f, 0.3f, 0.4f, -0.1f, 0.05f, 0.2f, -0.3f, 0.5f, 0.5f, 0.0f,
                            1.0f, 1.0f, 0.0f, 0.5f, -0.5f});
}

void RegisterAttentionBenchmark(std::vector<TestCase> &registry,
                                const onnx_kernels::kernel::Attention &attention,
                                const OpsetId &opset, const std::string &name, int64_t query_heads,
                                int64_t kv_heads, int64_t query_length, int64_t kv_length,
                                bool causal) {
  constexpr int64_t batch = 1;
  constexpr int64_t head_size = 64;
  const std::vector<int64_t> q_shape = {batch, query_heads, query_length, head_size};
  const std::vector<int64_t> kv_shape = {batch, kv_heads, kv_length, head_size};
  const int64_t q_count = batch * query_heads * query_length * head_size;
  const int64_t kv_count = batch * kv_heads * kv_length * head_size;

  NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
  if (causal) {
    AddInt(node, "is_causal", 1);
  }
  onnx_kernels::kernel::Attention::Attributes attributes;
  attributes.is_causal = causal;
  Expect(registry, std::move(node), name, {opset}, {q_count, kv_count, kv_count}, {q_count},
         [attention, attributes, q_shape, kv_shape, q_count, kv_count]() -> IoData {
           Tensor Q = RandnTensor(DataType::FLOAT, q_shape, 2501 + q_count);
           Tensor K = RandnTensor(DataType::FLOAT, kv_shape, 2502 + kv_count);
           Tensor V = RandnTensor(DataType::FLOAT, kv_shape, 2503 + kv_count);
           Tensor Y = attention(Q, K, V, attributes).Y;
           return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
         });
}

} // namespace

void RegisterAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(23);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Attention attention{ctx};

  if (mode == TestMode::BENCHMARK) {
    RegisterAttentionBenchmark(registry, attention, opset,
                               "test_cc_attention_prefill_mha_benchmark", 12, 12, 128, 128, true);
    RegisterAttentionBenchmark(registry, attention, opset,
                               "test_cc_attention_prefill_gqa_benchmark", 16, 4, 128, 128, true);
    RegisterAttentionBenchmark(registry, attention, opset,
                               "test_cc_attention_prefill_mqa_benchmark", 16, 1, 128, 128, true);
    RegisterAttentionBenchmark(registry, attention, opset,
                               "test_cc_attention_decode_mha_128_benchmark", 12, 12, 1, 128, false);
    RegisterAttentionBenchmark(registry, attention, opset,
                               "test_cc_attention_decode_gqa_128_benchmark", 16, 4, 1, 128, false);
    RegisterAttentionBenchmark(registry, attention, opset,
                               "test_cc_attention_decode_mqa_1024_benchmark", 16, 1, 1, 1024,
                               false);
    return;
  }

  // -------------------------------------------------------------------
  // Case 1: basic MHA (matches upstream ``test_attention_4d``).
  {
    Tensor Q = MakeQ_1_2_2_2_basic();
    Tensor K = MakeKV_basic_K();
    Tensor V = MakeKV_basic_V();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_attention_4d", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 2: GQA (matches upstream ``test_attention_4d_gqa``).
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // -------------------------------------------------------------------
  // Case 3: explicit ``scale`` attribute (1e-2, far from the default
  // 1/sqrt(head_size)). Exercises the scale-attribute path.
  {
    Tensor Q = MakeQ_1_2_2_2_basic();
    Tensor K = MakeKV_basic_K();
    Tensor V = MakeKV_basic_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1e-2f);
    Expect(registry, std::move(node), "test_cc_attention_4d_scaled", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 4: ``diff_heads_sizes`` — V has a head_size that differs from
  // Q/K's head_size, exercising the asymmetric ``v_head_size`` path.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_sizes", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // Case 5: ``is_causal`` — upper-triangular ``-inf`` mask. Q/K/V are
  // square in the sequence axis (q_seq = kv_seq = 3) so the causal mask
  // is a strict lower triangle of allowed attention.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 3, 2},
                                 {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f,      // head 0
                                  -1.0f, 1.0f, 1.0f, -1.0f, 0.25f, 0.5f}); // head 1
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_causal", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 6: FLOAT 4D ``attn_mask`` added as a bias.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {1, 2, 2, 3},
                                    {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, -0.1f,  // head 0
                                     -0.2f, 0.3f, 0.0f, 0.0f, -0.1f, 0.4f}); // head 1
    Tensor Y = attention(Q, K, V, /*scale=*/0.5f, mask);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask_4d", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // Case 7: FLOAT 3D ``attn_mask`` broadcast over the head axis.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {1, 2, 3}, {0.0f, -1.0f, 0.5f, 0.2f, 0.0f, -0.4f});
    Tensor Y = attention(Q, K, V, /*scale=*/0.5f, mask);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask_3d", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // Case 8: FLOAT 2D ``attn_mask`` broadcast over batch and heads (matches
  // upstream ``test_attention_4d_attn_mask``).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    Tensor Y = attention(Q, K, V, /*scale=*/0.5f, mask);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)}, {std::move(Y)}};
    });
  }

  // Case 9: BOOL ``attn_mask`` — ``true`` = attend, ``false`` = ``-inf``.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromBool("", {1, 2, 2, 3},
                                   {1, 1, 0, 1, 0, 1,   // head 0
                                    1, 0, 1, 1, 1, 0}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask_bool", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // Case 10: ``softcap > 0``. ``softcap * tanh(s / softcap)`` is applied
  // between the bias and the softmax. Scale is set to a non-default value
  // to make the saturating effect visible.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_softcap", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 11: ``past_key`` / ``past_value`` are concatenated with K/V to
  // produce ``present_key`` / ``present_value`` of length
  // ``past_kv_seq_len + kv_seq_len``. All three outputs are exposed.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor past_key = Tensor::FromFloat("", {1, 2, 2, 2},
                                        {0.5f, -0.5f, 0.0f, 0.5f,   // head 0
                                         1.0f, 0.0f, -0.5f, 1.0f}); // head 1
    Tensor past_value = Tensor::FromFloat("", {1, 2, 2, 2},
                                          {0.5f, 0.5f, -1.0f, 0.0f,   // head 0
                                           0.0f, 0.5f, 0.5f, -0.5f}); // head 1
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(registry, std::move(node), "test_cc_attention_4d_with_past_and_present", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // Case 12: ``qk_matmul_output`` exposed with mode 0 (raw QK^T * scale,
  // no bias / softcap / softmax applied).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    Expect(registry, std::move(node), "test_cc_attention_4d_with_qk_matmul", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  // Case 13: ``qk_matmul_output`` mode 1 — after adding ``attn_mask``.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask);
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y", "", "", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_with_qk_matmul_bias", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  // Case 14: ``qk_matmul_output`` mode 2 — after the softcap.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    attrs.qk_matmul_output_mode = 2;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 2);
    Expect(registry, std::move(node), "test_cc_attention_4d_with_qk_matmul_softcap", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  // Case 15: ``qk_matmul_output`` mode 3 — after softmax (probabilities).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(registry, std::move(node), "test_cc_attention_4d_with_qk_matmul_softmax", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  // -------------------------------------------------------------------
  // Rank-3 fused-layout cases. Inputs are
  // ``(batch, seq, num_heads * head_size)`` and the kernel transparently
  // reshapes/transposes to rank-4, runs attention and reshapes back.
  //
  // The same data values used by the rank-4 cases are reused via the
  // kernel's ``CollapseToRank3`` round-trip — this guarantees the rank-3
  // outputs are bit-exact replicas of the rank-4 ones modulo the layout.

  auto rank3_inputs = []() {
    // Q: (1, 2, 4) ← collapse of (1, 2, 2, 2)
    return Tensor::FromFloat("", {1, 2, 4}, {1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 1.0f, 1.0f, -1.0f});
  };
  auto rank3_K = []() {
    // K: (1, 3, 4) ← collapse of (1, 2, 3, 2)
    return Tensor::FromFloat(
        "", {1, 3, 4}, {1.0f, 0.0f, -1.0f, 1.0f, 0.5f, 0.5f, 1.0f, 1.0f, 0.0f, 1.0f, 0.25f, -0.5f});
  };
  auto rank3_V = []() {
    // V: (1, 3, 4) ← collapse of (1, 2, 3, 2)
    return Tensor::FromFloat(
        "", {1, 3, 4},
        {1.0f, 0.0f, 2.0f, -2.0f, 0.0f, 1.0f, 0.5f, 0.25f, -1.0f, 1.0f, -0.5f, 0.0f});
  };

  // Case 16: basic rank-3 MHA.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 17: rank-3 GQA — Q has 4 heads, K/V have 2 heads.
  {
    // Q: (1, 2, 8) ← collapse of (1, 4, 2, 2). Uses the same 16 values as
    // the rank-4 GQA case, re-arranged for fused layout.
    Tensor Q = Tensor::FromFloat(
        "", {1, 2, 8},
        // seq=0: head0[0], head1[0], head2[0], head3[0] each contributing 2 values
        {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f,
         // seq=1: head0[1], head1[1], head2[1], head3[1]
         0.3f, 0.4f, 0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d_gqa", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 18: rank-3 causal — square q/kv sequence lengths.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 3, 4}, {1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.5f, 0.5f, 0.25f, 0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_3d_causal", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // Case 19: rank-3 with ``past_key``/``past_value`` and full
  // ``present_*`` outputs. Past KV is rank-4 by spec; only Q/K/V are rank-3.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor past_key = Tensor::FromFloat("", {1, 2, 2, 2},
                                        {0.5f, -0.5f, 0.0f, 0.5f,   // head 0
                                         1.0f, 0.0f, -0.5f, 1.0f}); // head 1
    Tensor past_value = Tensor::FromFloat("", {1, 2, 2, 2},
                                          {0.5f, 0.5f, -1.0f, 0.0f,   // head 0
                                           0.0f, 0.5f, 0.5f, -0.5f}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d_with_past_and_present", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // -------------------------------------------------------------------
  // Additional rank-3 (fused layout) variants mirroring the upstream
  // ``test_attention_3d_*`` cases.

  // 3D scaled.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1e-2f);
    Expect(registry, std::move(node), "test_cc_attention_3d_scaled", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 3D softcap.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_softcap", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 3D attn_mask (FLOAT, broadcast over batch/heads).
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_attn_mask", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)}, {std::move(Y)}};
    });
  }

  // 3D ``diff_heads_sizes`` — V has a different head_size than Q/K.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    // V: (1, 3, 6) ← collapse of (1, 2, 3, 3) so v_head_size=3.
    Tensor V = Tensor::FromFloat("", {1, 3, 6},
                                 {1.0f, 0.0f, -1.0f, 2.0f, -2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.5f,
                                  0.25f, -0.25f, -1.0f, 1.0f, 0.5f, -0.5f, 0.0f, 1.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d_diff_heads_sizes", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 3D GQA + scaled.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1e-2f);
    Expect(registry, std::move(node), "test_cc_attention_3d_gqa_scaled", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 3D GQA + softcap.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_gqa_softcap", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 3D GQA + attn_mask.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_gqa_attn_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 3D GQA + causal — square q/kv sequence lengths.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 3, 8}, {0.1f, 0.2f,  -0.1f, 0.05f, 0.5f, 0.5f, 1.0f,  0.0f,
                                          0.3f, 0.4f,  0.2f,  -0.3f, 0.0f, 1.0f, 0.5f,  -0.5f,
                                          0.2f, -0.1f, 0.25f, 0.0f,  0.5f, 0.5f, -1.0f, 1.0f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_3d_gqa_causal", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 3D GQA + past_key / past_value.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d_gqa_with_past_and_present", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // 3D with past_and_present + qk_matmul_output (mode 0).
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d_with_past_and_present_qk_matmul",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 3D with past_and_present + qk_matmul_output (mode 1, post-bias).
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f, 0.0f, -0.2f, -0.1f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node), "test_cc_attention_3d_with_past_and_present_qk_matmul_bias",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 3D with past_and_present + qk_matmul_output (mode 2, post-softcap).
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    attrs.qk_matmul_output_mode = 2;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 2);
    Expect(registry, std::move(node),
           "test_cc_attention_3d_with_past_and_present_qk_matmul_softcap", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 3D with past_and_present + qk_matmul_output (mode 3, post-softmax).
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(registry, std::move(node),
           "test_cc_attention_3d_with_past_and_present_qk_matmul_softmax", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // -------------------------------------------------------------------
  // Additional rank-4 variants mirroring the upstream
  // ``test_attention_4d_*`` cases.

  // 4D GQA + scaled.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1e-2f);
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_scaled", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 4D GQA + softcap.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_softcap", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 4D GQA + attn_mask (FLOAT, broadcast over heads).
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_attn_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 4D GQA + causal — square q/kv sequence lengths.
  {
    Tensor Q = Tensor::FromFloat("", {1, 4, 3, 2},
                                 {
                                     0.1f,  0.2f,  0.3f,   0.4f,  -0.1f, 0.05f, // head 0
                                     0.2f,  -0.3f, 0.5f,   0.5f,  0.0f,  1.0f,  // head 1
                                     1.0f,  0.0f,  0.5f,   -0.5f, 0.25f, 0.1f,  // head 2
                                     -0.5f, 0.5f,  -0.25f, 0.75f, 0.1f,  -0.1f  // head 3
                                 });
    Tensor K = Tensor::FromFloat(
        "", {1, 2, 3, 2},
        {1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.25f, -0.5f});
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_causal", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  // 4D GQA + past_key / past_value.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_with_past_and_present", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // 4D + 4D ``attn_mask`` + causal.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 2, 3, 2},
        {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 1.0f, -1.0f, 0.25f, 0.5f});
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask =
        Tensor::FromFloat("", {1, 2, 3, 3},
                          {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, -0.1f, 0.5f, -0.2f, 0.0f,  // head 0
                           -0.2f, 0.3f, 0.0f, 0.0f, -0.1f, 0.4f, 0.1f, 0.0f, -0.3f}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask_4d_causal", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 4D + 3D ``attn_mask`` + causal.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 2, 3, 2},
        {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 1.0f, -1.0f, 0.25f, 0.5f});
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask =
        Tensor::FromFloat("", {1, 3, 3}, {0.0f, -1.0f, 0.5f, 0.2f, 0.0f, -0.4f, 0.1f, -0.3f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask_3d_causal", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 4D BOOL ``attn_mask`` with 4D shape.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromBool("", {1, 2, 2, 3},
                                   {1, 1, 0, 1, 0, 1,   // head 0
                                    1, 0, 1, 1, 1, 0}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_attn_mask_bool_4d", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 4D with past_and_present + qk_matmul_output (mode 0).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    Expect(registry, std::move(node), "test_cc_attention_4d_with_past_and_present_qk_matmul",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 4D with past_and_present + qk_matmul_output (mode 1, post-bias).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f, 0.0f, -0.2f, -0.1f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_with_past_and_present_qk_matmul_bias",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 4D with past_and_present + qk_matmul_bias + 3D mask.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {1, 2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f, 0.0f, -0.2f, -0.1f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 4D with past_and_present + qk_matmul_bias + 4D mask.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {1, 2, 2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f,  0.0f, -0.2f, -0.1f, 0.0f,
                           0.1f, 0.2f,  -0.3f, 0.0f, 0.4f, -0.4f, 0.0f, 0.3f,  -0.2f, 0.1f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 4D with past_and_present + qk_matmul_bias + 3D mask + causal.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {1, 2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f, 0.0f, -0.2f, -0.1f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 4D with past_and_present + qk_matmul_bias + 4D mask + causal.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {1, 2, 2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f,  0.0f, -0.2f, -0.1f, 0.0f,
                           0.1f, 0.2f,  -0.3f, 0.0f, 0.4f, -0.4f, 0.0f, 0.3f,  -0.2f, 0.1f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value),
                            std::move(r.qk_matmul_output)}};
           });
  }

  // 4D ``diff_heads_sizes`` + scaled.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1e-2f);
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_sizes_scaled", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 4D ``diff_heads_sizes`` + softcap.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_sizes_softcap", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 4D ``diff_heads_sizes`` + attn_mask.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    Tensor Y = attention(Q, K, V, /*scale=*/0.5f, mask);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_sizes_attn_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 4D ``diff_heads_sizes`` + causal — square q/kv lengths.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 2, 3, 2},
        {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 1.0f, -1.0f, 0.25f, 0.5f});
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_sizes_causal", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 4D ``diff_heads`` (Q has more heads than KV) with past_and_present.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_with_past_and_present",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // 4D ``diff_heads`` with past_and_present + 3D mask.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor mask = Tensor::FromFloat(
        "", {1, 2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f, 0.0f, -0.2f, -0.1f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_diff_heads_with_past_and_present_mask3d", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // 4D ``diff_heads`` with past_and_present + 4D mask.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    // Broadcastable over the head axis (1 vs 4) and batch.
    Tensor mask = Tensor::FromFloat(
        "", {1, 1, 2, 5}, {0.0f, -0.5f, -1.0f, 0.2f, 0.0f, 0.5f, 0.0f, -0.2f, -0.1f, 0.0f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_diff_heads_with_past_and_present_mask4d", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // 4D ``diff_heads`` with 4D mask serving as padded-KV mask.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    // Mask is broadcastable over heads; final column is -inf-like (-1e4)
    // to emulate KV-padding suppression on the trailing position.
    Tensor mask = Tensor::FromFloat("", {1, 1, 2, 3}, {0.0f, 0.0f, -1.0e4f, 0.0f, 0.0f, -1.0e4f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_mask4d_padded_kv", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 4D ``diff_heads`` with ``nonpad_kv_seqlen`` masking trailing padding
  // key/value positions (opset 24). Mirrors upstream
  // ``test_attention_4d_diff_heads_mask4d_padded_kv``.
  {
    const OpsetId opset24 = DefaultOpset(24);
    Tensor Q = Tensor::FromFloat("", {2, 2, 2, 2},
                                 {0.1f, 0.2f, 0.3f, 0.4f, -0.1f, 0.05f, 0.2f, -0.3f, 0.5f, 0.5f,
                                  0.0f, 1.0f, 1.0f, 0.0f, 0.5f, -0.5f});
    Tensor K =
        Tensor::FromFloat("", {2, 2, 3, 2}, {0.5f,  -0.5f,  0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f,
                                             0.25f, -0.25f, 0.5f, 0.5f, 1.0f, 1.0f, -1.0f, 0.0f,
                                             0.0f,  -1.0f,  0.5f, 0.5f, 0.1f, 0.2f, 0.3f,  0.4f});
    Tensor V =
        Tensor::FromFloat("", {2, 2, 3, 2}, {0.5f,  0.5f, -1.0f, 0.0f, 0.0f, 0.5f,  0.5f,  -0.5f,
                                             1.0f,  1.0f, -0.5f, 0.5f, 0.2f, -0.2f, 0.4f,  0.6f,
                                             -0.1f, 0.3f, 0.0f,  1.0f, 1.0f, 0.0f,  -1.0f, -1.0f});
    Tensor mask =
        Tensor::FromFloat("", {2, 2, 2, 3}, {0.0f,  -0.5f, -1.0f, 0.2f,  0.0f, -0.2f, 0.5f, 0.0f,
                                             -0.1f, 0.0f,  -0.3f, 0.1f,  0.0f, 0.0f,  0.0f, -0.4f,
                                             -0.2f, 0.0f,  0.1f,  -0.1f, 0.2f, 0.0f,  0.0f, -0.5f});
    Tensor nonpad_kv_seqlen = Tensor::FromInt64("", {2}, {2, 3});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask, /*past_key=*/nullptr, /*past_value=*/nullptr,
                         &nonpad_kv_seqlen)
                   .Y;
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_diff_heads_mask4d_nonpad_kv", {opset24},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(nonpad_kv_seqlen)},
                           {std::move(Y)}};
           });
  }

  // 4D external/static KV-cache causal decode + continued-prefill with
  // ``nonpad_kv_seqlen`` and no ``past_key`` (opset 24). Mirrors upstream
  // ``test_attention_4d_gqa_causal_nonpad_decode`` /
  // ``test_attention_4d_causal_nonpad_continued_prefill`` (onnx/onnx#8068):
  // the ``is_causal`` frontier is bottom-right / offset-aware, so each query
  // attends keys ``j <= i + (nonpad_kv_seqlen[b] - q_seq_len)``.
  {
    const OpsetId opset24 = DefaultOpset(24);
    // q_seq=2, kv_seq=4, nonpad=[4,3] -> offsets {2, 1}.
    Tensor Q = MakeDeterministicFloatTensor({2, 2, 2, 2}, 0x51a1u, 0.0f, 1.0f);
    Tensor K = MakeDeterministicFloatTensor({2, 2, 4, 2}, 0x51a2u, 0.0f, 1.0f);
    Tensor V = MakeDeterministicFloatTensor({2, 2, 4, 2}, 0x51a3u, 0.0f, 1.0f);
    Tensor nonpad_kv_seqlen = Tensor::FromInt64("", {2}, {4, 3});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, /*past_key=*/nullptr,
                         /*past_value=*/nullptr, &nonpad_kv_seqlen)
                   .Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_causal_nonpad_kv_continued_prefill",
           {opset24}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(nonpad_kv_seqlen)},
                           {std::move(Y)}};
           });
  }

  // Additional upstream-name parity coverage for backend-test comparison.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor bool_mask = Tensor::FromBool("", {1, 2, 2, 3},
                                        {0, 0, 0, 1, 1, 1,   // head 0
                                         0, 0, 0, 1, 1, 1}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &bool_mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node),
           "test_cc_attention_23_boolmask_fullymasked_row_nan_robustness", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(bool_mask)},
                           {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor bool_mask = Tensor::FromBool("", {1, 2, 2, 3},
                                        {0, 0, 0, 1, 1, 1,   // head 0
                                         0, 0, 0, 1, 1, 1}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, &bool_mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_causal_boolmask_nan_robustness", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(bool_mask)},
                           {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor bool_mask = Tensor::FromBool("", {1, 2, 2, 3},
                                        {0, 0, 0, 0, 0, 0,   // head 0
                                         0, 0, 0, 0, 0, 0}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs, &bool_mask);
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y", "", "", "qk_matmul_output"});
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(registry, std::move(node),
           "test_cc_attention_23_fullymasked_qk_matmul_output_mode3_zero", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(bool_mask)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  {
    const OpsetId opset24 = DefaultOpset(24);
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor bool_mask = Tensor::FromBool("", {1, 2, 2, 3},
                                        {0, 0, 0, 0, 0, 0,   // head 0
                                         0, 0, 0, 0, 0, 0}); // head 1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs, &bool_mask);
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y", "", "", "qk_matmul_output"});
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(registry, std::move(node),
           "test_cc_attention_24_fullymasked_qk_matmul_output_mode3_zero", {opset24},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(bool_mask)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  {
    const OpsetId opset24 = DefaultOpset(24);
    constexpr uint32_t kSoftmaxPrecisionQSeed = 0x62a1u;
    constexpr uint32_t kSoftmaxPrecisionKSeed = 0x62a2u;
    constexpr uint32_t kSoftmaxPrecisionVSeed = 0x62a3u;
    Tensor Q = MakeDeterministicFloatTensor({1, 2, 2, 4}, kSoftmaxPrecisionQSeed, 0.0f, 1.0f);
    Tensor K = MakeDeterministicFloatTensor({1, 2, 3, 4}, kSoftmaxPrecisionKSeed, 0.0f, 1.0f);
    Tensor V = MakeDeterministicFloatTensor({1, 2, 3, 4}, kSoftmaxPrecisionVSeed, 0.0f, 1.0f);
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(registry, std::move(node),
           "test_cc_attention_24_qk_matmul_output_mode3_softmax_precision", {opset24},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)},
                           {std::move(r.Y), std::move(r.qk_matmul_output)}};
           });
  }

  {
    const OpsetId opset24 = DefaultOpset(24);
    constexpr uint32_t kNonPadCompositionQSeed = 0x73a1u;
    constexpr uint32_t kNonPadCompositionKSeed = 0x73a2u;
    constexpr uint32_t kNonPadCompositionVSeed = 0x73a3u;
    Tensor Q = MakeDeterministicFloatTensor({2, 2, 2, 2}, kNonPadCompositionQSeed, 0.0f, 1.0f);
    Tensor K = MakeDeterministicFloatTensor({2, 2, 4, 2}, kNonPadCompositionKSeed, 0.0f, 1.0f);
    Tensor V = MakeDeterministicFloatTensor({2, 2, 4, 2}, kNonPadCompositionVSeed, 0.0f, 1.0f);
    Tensor mask = Tensor::FromFloat("", {2, 1, 2, 4},
                                    {0.0f, 0.0f, -2.0f, -4.0f, 0.0f, 0.0f, -1.0f, -2.0f, 0.0f,
                                     -0.5f, -2.0f, -4.0f, 0.0f, -0.5f, -1.5f, -3.0f});
    Tensor nonpad_kv_seqlen = Tensor::FromInt64("", {2}, {4, 3});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    auto r = attention(Q, K, V, attrs, &mask, /*past_key=*/nullptr, /*past_value=*/nullptr,
                       &nonpad_kv_seqlen);
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    // Upstream exposes distinct backend test names for scenario variants that
    // intentionally share the same numerics (attn_mask_composition,
    // batch_prefill, continued_prefill, negative_offset_structural_empty).
    // Register all aliases so the ONNX-vs-onnx-light name-parity check sees a
    // matching case for each name.  The lambda copies from its captures so it
    // can be safely passed to multiple Expect calls.
    auto make_nonpad_composition_io = [=]() -> IoData {
      return IoData{{Q, K, V, mask, nonpad_kv_seqlen}, {r.Y}};
    };
    Expect(registry, node, "test_cc_attention_4d_causal_nonpad_attn_mask_composition", {opset24},
           make_nonpad_composition_io);
    Expect(registry, node, "test_cc_attention_4d_causal_nonpad_batch_prefill", {opset24},
           make_nonpad_composition_io);
    Expect(registry, node, "test_cc_attention_4d_causal_nonpad_continued_prefill", {opset24},
           make_nonpad_composition_io);
    Expect(registry, std::move(node),
           "test_cc_attention_4d_causal_nonpad_negative_offset_structural_empty", {opset24},
           std::move(make_nonpad_composition_io));
  }

  {
    const OpsetId opset24 = DefaultOpset(24);
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor nonpad_kv_seqlen = Tensor::FromInt64("", {1}, {3});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, /*past_key=*/nullptr,
                         /*past_value=*/nullptr, &nonpad_kv_seqlen)
                   .Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_causal_nonpad_decode", {opset24},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(nonpad_kv_seqlen)},
                           {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor past_key =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor past_value =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f});
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_4d_causal_with_past_and_present", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // -------------------------------------------------------------------
  // Softcap + ``-inf`` mask ordering checks (mirror upstream
  // ``test_attention_4d_softcap_neginf_mask*``). These verify the kernel
  // applies softcap to the raw QK scores BEFORE adding mask/bias: if the
  // order were swapped, ``sc * tanh(-inf / sc) == -sc`` (finite) would
  // leak probability into masked positions.
  {
    // Shapes match upstream: (B=1, H=1, S_q=4, S_kv=6, D=8).
    Tensor Q = MakeDeterministicFloatTensor({1, 1, 4, 8}, 0x42a1u, 0.0f, 1.0f);
    Tensor K = MakeDeterministicFloatTensor({1, 1, 6, 8}, 0x42a2u, 0.0f, 1.0f);
    Tensor V = MakeDeterministicFloatTensor({1, 1, 6, 8}, 0x42a3u, 0.0f, 1.0f);
    std::vector<float> mask_data(static_cast<size_t>(4 * 6), 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < 4; ++i) {
      mask_data[static_cast<size_t>(i * 6 + 4)] = neg_inf;
      mask_data[static_cast<size_t>(i * 6 + 5)] = neg_inf;
    }
    Tensor mask = Tensor::FromFloat("", {4, 6}, mask_data);
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_softcap_neginf_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // Same as above but with "poison" values at the masked KV positions.
  // With correct ordering the masked positions contribute zero so the
  // output stays bounded; with wrong ordering they would dominate and
  // the magnitude would explode.
  {
    Tensor Q = MakeDeterministicFloatTensor({1, 1, 4, 8}, 0x42a1u, 0.0f, 1.0f);
    Tensor K = MakeDeterministicFloatTensor({1, 1, 6, 8}, 0x42a2u, 0.0f, 1.0f);
    Tensor V = MakeDeterministicFloatTensor({1, 1, 6, 8}, 0x42a3u, 0.0f, 1.0f);
    // Poison the last two KV positions with a large value.
    float *vptr = V.AsFloat();
    for (int64_t s = 4; s < 6; ++s) {
      for (int64_t d = 0; d < 8; ++d) {
        vptr[s * 8 + d] = 1000.0f;
      }
    }
    std::vector<float> mask_data(static_cast<size_t>(4 * 6), 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < 4; ++i) {
      mask_data[static_cast<size_t>(i * 6 + 4)] = neg_inf;
      mask_data[static_cast<size_t>(i * 6 + 5)] = neg_inf;
    }
    Tensor mask = Tensor::FromFloat("", {4, 6}, mask_data);
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_4d_softcap_neginf_mask_poison", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Additional rank-3 (fused layout) ``diff_heads_sizes`` variants
  // mirroring the upstream ``test_attention_3d_diff_heads_sizes_*`` cases.
  // V has a head_size of 3 (vs. 2 for Q/K), exercising the asymmetric
  // ``v_head_size`` path together with each feature attribute. The V
  // values are the rank-3 fused-layout reshape of the rank-4 ``(1, 2, 3,
  // 3)`` tensor used by the 4D ``diff_heads_sizes`` cases above.

  auto rank3_diff_heads_V = []() {
    // V: (1, 3, 6) ← collapse of (1, 2, 3, 3) — same payload as
    // ``MakeV_1_2_3_3`` written out in fused-layout order.
    return Tensor::FromFloat("", {1, 3, 6},
                             {1.0f, 0.0f, -1.0f, 2.0f, -2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.5f, 0.25f,
                              -0.25f, -1.0f, 1.0f, 0.5f, -0.5f, 0.0f, 1.0f});
  };

  // 3D diff_heads_sizes + scaled.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1e-2f);
    Expect(registry, std::move(node), "test_cc_attention_3d_diff_heads_sizes_scaled", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 3D diff_heads_sizes + softcap.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_diff_heads_sizes_softcap", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 3D diff_heads_sizes + attn_mask.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_diff_heads_sizes_attn_mask", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  // 3D diff_heads_sizes + causal — square q/kv lengths.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 3, 4}, {1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.5f, 0.5f, 0.25f, 0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "is_causal", 1);
    Expect(registry, std::move(node), "test_cc_attention_3d_diff_heads_sizes_causal", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // 3D diff_heads_sizes with past_key/past_value (and present_*) and an
  // attn_mask covering ``past_kv_seq_len + kv_seq_len``. Mirrors upstream's
  // ``test_attention_3d_diff_heads_with_past_and_present`` whose only
  // asymmetry is ``V`` carrying a larger head_size than Q/K.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    Tensor past_key = Tensor::FromFloat("", {1, 2, 2, 2},
                                        {0.5f, -0.5f, 0.0f, 0.5f,   // head 0
                                         1.0f, 0.0f, -0.5f, 1.0f}); // head 1
    Tensor past_value = Tensor::FromFloat("", {1, 2, 2, 3},
                                          {0.5f, 0.5f, -1.0f, 0.0f, 0.25f, 0.5f,     // head 0
                                           0.0f, 0.5f, 0.5f, -0.5f, 0.75f, -0.25f}); // head 1
    Tensor mask = Tensor::FromFloat("", {2, 5},
                                    {0.0f, -0.5f, -1.0f, 0.2f, 0.0f,   // q=0
                                     0.5f, 0.0f, -0.2f, -0.1f, 0.0f}); // q=1
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(registry, std::move(node), "test_cc_attention_3d_diff_heads_with_past_and_present",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(past_key), std::move(past_value)},
                           {std::move(r.Y), std::move(r.present_key), std::move(r.present_value)}};
           });
  }

  // 3D transpose verification — mirrors upstream's
  // ``test_attention_3d_transpose_verification``. Each query head carries
  // its own distinctive scalar pattern in the hidden dimension so the
  // rank-3 -> rank-4 reshape + transpose path can be inspected by reading
  // off ``Y``.
  {
    const int64_t q_num_heads = 3;
    const int64_t kv_num_heads = 3;
    const int64_t batch = 1;
    const int64_t q_seq = 2;
    const int64_t kv_seq = 2;
    const int64_t head_size = 4;
    const int64_t q_hidden = q_num_heads * head_size;
    const int64_t kv_hidden = kv_num_heads * head_size;
    std::vector<float> q_values(static_cast<size_t>(batch * q_seq * q_hidden), 0.0f);
    for (int64_t s = 0; s < q_seq; ++s) {
      for (int64_t h = 0; h < q_num_heads; ++h) {
        const float value = static_cast<float>(h + 1);
        for (int64_t d = 0; d < head_size; ++d) {
          q_values[static_cast<size_t>(s * q_hidden + h * head_size + d)] = value;
        }
      }
    }
    Tensor Q = Tensor::FromFloat("", {batch, q_seq, q_hidden}, q_values);
    Tensor K = Tensor::FromFloat(
        "", {batch, kv_seq, kv_hidden},
        std::vector<float>(static_cast<size_t>(batch * kv_seq * kv_hidden), 0.1f));
    Tensor V = Tensor::FromFloat(
        "", {batch, kv_seq, kv_hidden},
        std::vector<float>(static_cast<size_t>(batch * kv_seq * kv_hidden), 0.1f));
    onnx_kernels::kernel::Attention::Attributes attrs;
    attrs.q_num_heads = q_num_heads;
    attrs.kv_num_heads = kv_num_heads;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", q_num_heads);
    AddInt(node, "kv_num_heads", kv_num_heads);
    Expect(registry, std::move(node), "test_cc_attention_3d_transpose_verification", {opset},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  // -------------------------------------------------------------------
  // Rank-4 FP16 variants. The kernel itself runs in FP32; inputs are
  // round-tripped through the FP16 encoding so the expected output mirrors
  // the precision a true FP16 backend would deliver. Tolerances on the
  // registered cases are loosened to account for the resulting rounding
  // error.

  // 4D fp16 — basic MHA over deterministic small inputs.
  {
    Tensor Q32 = MakeDeterministicFloatTensor({2, 3, 4, 8}, 0x1234u, 0.0f, 1.0f);
    Tensor K32 = MakeDeterministicFloatTensor({2, 3, 6, 8}, 0x5678u, 0.0f, 1.0f);
    Tensor V32 = MakeDeterministicFloatTensor({2, 3, 6, 8}, 0x9abcu, 0.0f, 1.0f);
    Tensor Q_in = RoundToFloat16(Q32);
    Tensor K_in = RoundToFloat16(K32);
    Tensor V_in = RoundToFloat16(V32);
    Tensor Y32 = attention(Q_in, K_in, V_in);
    Tensor Q = FloatToFloat16Tensor("", Q_in);
    Tensor K = FloatToFloat16Tensor("", K_in);
    Tensor V = FloatToFloat16Tensor("", V_in);
    Tensor Y = FloatToFloat16Tensor("", Y32);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_attention_4d_fp16", {opset}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
    registry.back().atol = 5e-3;
    registry.back().rtol = 5e-3;
  }
  {
    const OpsetId opset24 = DefaultOpset(24);
    Tensor Q32 = MakeDeterministicFloatTensor({2, 9, 4, 8}, 0xdeadu, 0.0f, 1.0f);
    Tensor K32 = MakeDeterministicFloatTensor({2, 3, 6, 8}, 0xbeefu, 0.0f, 1.0f);
    Tensor V32 = MakeDeterministicFloatTensor({2, 3, 6, 8}, 0xfeedu, 0.0f, 1.0f);
    Tensor mask32 = MakeDeterministicFloatTensor({4, 18}, 0xcafeu, 0.0f, 1.0f);
    Tensor pk32 = MakeDeterministicFloatTensor({2, 3, 12, 8}, 0xface, 0.0f, 1.0f);
    Tensor pv32 = MakeDeterministicFloatTensor({2, 3, 12, 8}, 0xb16bu, 0.0f, 1.0f);
    Tensor Q_in = RoundToFloat16(Q32);
    Tensor K_in = RoundToFloat16(K32);
    Tensor V_in = RoundToFloat16(V32);
    Tensor mask_in = RoundToFloat16(mask32);
    Tensor pk_in = RoundToFloat16(pk32);
    Tensor pv_in = RoundToFloat16(pv32);
    onnx_kernels::kernel::Attention::Attributes attrs;
    auto r = attention(Q_in, K_in, V_in, attrs, &mask_in, &pk_in, &pv_in);
    Tensor Q = FloatToFloat16Tensor("", Q_in);
    Tensor K = FloatToFloat16Tensor("", K_in);
    Tensor V = FloatToFloat16Tensor("", V_in);
    Tensor mask = FloatToFloat16Tensor("", mask_in);
    Tensor pk = FloatToFloat16Tensor("", pk_in);
    Tensor pv = FloatToFloat16Tensor("", pv_in);
    Tensor Y = FloatToFloat16Tensor("", r.Y);
    Tensor present_key = FloatToFloat16Tensor("", r.present_key);
    Tensor present_value = FloatToFloat16Tensor("", r.present_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(registry, std::move(node), "test_cc_attention_4d_gqa_with_past_and_present_fp16",
           {opset}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask),
                            std::move(pk), std::move(pv)},
                           {std::move(Y), std::move(present_key), std::move(present_value)}};
           });
    registry.back().atol = 5e-3;
    registry.back().rtol = 5e-3;

    Tensor nonpad_kv_seqlen = Tensor::FromInt64("nonpad_kv_seqlen", {2}, {6, 5});
    onnx_kernels::kernel::Attention::Attributes decode_attrs;
    decode_attrs.is_causal = true;
    Tensor decode_y =
        attention(Q_in, K_in, V_in, decode_attrs, /*attn_mask=*/nullptr, /*past_key=*/nullptr,
                  /*past_value=*/nullptr, &nonpad_kv_seqlen)
            .Y;
    Tensor decode_Y = FloatToFloat16Tensor("", decode_y);
    NodeProto decode_node =
        MakeAttentionNode({"Q", "K", "V", "", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(decode_node, "is_causal", 1);
    Expect(registry, std::move(decode_node), "test_cc_attention_4d_gqa_causal_nonpad_decode_fp16",
           {opset24}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(nonpad_kv_seqlen)},
                           {std::move(decode_Y)}};
           });
    registry.back().atol = 5e-3;
    registry.back().rtol = 5e-3;
  }

  // -------------------------------------------------------------------
  // Attention-25 local-window cases from onnx/onnx#8108. Q and K are zero
  // so the independently constructed expected outputs are exact uniform
  // averages over the positions retained by cache, mask, causal, and window
  // rules. This avoids deriving expected values from the kernel under test.
  const OpsetId opset25 = DefaultOpset(25);

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 6, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 6, 8);
    Tensor Y = MakeUniformWindowReference4(2, 3, 3, 4, 6, 8, 2, -1, true, {}, {}, {}).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window", {opset25}, [=]() -> IoData {
      return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
    });
  }

  {
    Tensor Q = MakeConstantFloatTensor({1, 1, 5, 1}, 0.0f);
    Tensor K = MakeConstantFloatTensor({1, 1, 5, 1}, 0.0f);
    Tensor V = MakePositionValueTensor4(1, 1, 5, 1);
    Tensor Y = MakeUniformWindowReference4(1, 1, 1, 5, 5, 1, 1, 2, false, {}, {}, {}).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "left_window_size", 1);
    AddInt(node, "right_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_bidirectional_window", {opset25},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 6, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 6, 8);
    Tensor Y = MakeUniformWindowReference4(2, 3, 3, 4, 6, 8, -1, -1, false, {}, {}, {}).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "left_window_size", -1);
    AddInt(node, "right_window_size", -1);
    Expect(registry, std::move(node), "test_cc_attention_local_window_default", {opset25},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 6, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 6, 8);
    Tensor mask = Tensor::FromBool("", {6}, {1, 1, 1, 1, 0, 0});
    const WindowMaskPredicate mask_allows = [](int64_t, int64_t, int64_t, int64_t j) {
      return j < 4;
    };
    Tensor Y = MakeUniformWindowReference4(2, 3, 3, 4, 6, 8, 2, -1, true, {}, {}, mask_allows).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window_rank1_boolean_mask",
           {opset25}, [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 2, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 2, 8, 8);
    Tensor past_key = MakeConstantFloatTensor({2, 3, 8, 8}, 0.0f);
    Tensor past_value = MakePositionValueTensor4(2, 3, 8, 8);
    Tensor Y = MakeUniformWindowReference4(2, 3, 3, 4, 10, 8, 2, -1, true, {8, 8}, {}, {}).Y;
    Tensor present_key = MakeConstantFloatTensor({2, 3, 10, 8}, 0.0f);
    Tensor present_value = MakePositionValueTensor4(2, 3, 10, 8);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window_with_past", {opset25},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(past_key),
                            std::move(past_value)},
                           {std::move(Y), std::move(present_key), std::move(present_value)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 8, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 8, 8);
    std::vector<float> mask_values(static_cast<size_t>(3 * 4 * 8), 0.0f);
    for (int64_t h = 0; h < 3; ++h) {
      for (int64_t i = 0; i < 4; ++i) {
        mask_values[static_cast<size_t>((h * 4 + i) * 8 + h)] =
            -std::numeric_limits<float>::infinity();
      }
    }
    Tensor mask = Tensor::FromFloat("", {3, 4, 8}, mask_values);
    Tensor nonpad = Tensor::FromInt64("", {2}, {6, 7});
    const WindowMaskPredicate mask_allows = [](int64_t, int64_t h, int64_t, int64_t j) {
      return j != h;
    };
    Tensor Y =
        MakeUniformWindowReference4(2, 3, 3, 4, 8, 8, 2, -1, true, {2, 3}, {6, 7}, mask_allows).Y;
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window_ext_cache_rank3_head_mask",
           {opset25}, [=]() -> IoData {
             return IoData{
                 {std::move(Q), std::move(K), std::move(V), std::move(mask), std::move(nonpad)},
                 {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 8, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 8, 8);
    std::vector<float> mask_values(static_cast<size_t>(2 * 4 * 8), 0.0f);
    for (int64_t b = 0; b < 2; ++b) {
      for (int64_t i = 0; i < 4; ++i) {
        mask_values[static_cast<size_t>((b * 4 + i) * 8 + b)] =
            -std::numeric_limits<float>::infinity();
      }
    }
    Tensor mask = Tensor::FromFloat("", {2, 1, 4, 8}, mask_values);
    Tensor nonpad = Tensor::FromInt64("", {2}, {6, 7});
    const WindowMaskPredicate mask_allows = [](int64_t b, int64_t, int64_t, int64_t j) {
      return j != b;
    };
    Tensor Y =
        MakeUniformWindowReference4(2, 3, 3, 4, 8, 8, 2, -1, true, {2, 3}, {6, 7}, mask_allows).Y;
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window_ext_cache_rank4_batch_mask",
           {opset25}, [=]() -> IoData {
             return IoData{
                 {std::move(Q), std::move(K), std::move(V), std::move(mask), std::move(nonpad)},
                 {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 3, 8, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 3, 8, 8);
    Tensor mask = Tensor::FromFloat(
        "", {1, 8},
        {0.0f, -std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor nonpad = Tensor::FromInt64("", {2}, {6, 7});
    const WindowMaskPredicate mask_allows = [](int64_t, int64_t, int64_t, int64_t j) {
      return j != 1;
    };
    Tensor Y =
        MakeUniformWindowReference4(2, 3, 3, 4, 8, 8, 2, -1, true, {2, 3}, {6, 7}, mask_allows).Y;
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window_ext_cache_rank2_mask",
           {opset25}, [=]() -> IoData {
             return IoData{
                 {std::move(Q), std::move(K), std::move(V), std::move(mask), std::move(nonpad)},
                 {std::move(Y)}};
           });
  }

  {
    Tensor Q32 = MakeConstantFloatTensor({2, 3, 4, 8}, 0.0f);
    Tensor K32 = MakeConstantFloatTensor({2, 3, 8, 8}, 0.0f);
    Tensor V32 = MakeConstantFloatTensor({2, 3, 8, 8}, 1.0f);
    Tensor mask32 = MakeConstantFloatTensor({1, 8}, 0.0f);
    Tensor Y32 = MakeConstantFloatTensor({2, 3, 4, 8}, 1.0f);
    Tensor Q = FloatToFloat16Tensor("", Q32);
    Tensor K = FloatToFloat16Tensor("", K32);
    Tensor V = FloatToFloat16Tensor("", V32);
    Tensor mask = FloatToFloat16Tensor("", mask32);
    Tensor Y = FloatToFloat16Tensor("", Y32);
    Tensor nonpad = Tensor::FromInt64("", {2}, {6, 7});
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"}, {"Y"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_local_window_ext_cache_float16_mask",
           {opset25}, [=]() -> IoData {
             return IoData{
                 {std::move(Q), std::move(K), std::move(V), std::move(mask), std::move(nonpad)},
                 {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 4, 32}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 6, 8}, 0.0f);
    Tensor V = MakePositionValueTensor3(2, 1, 6, 6);
    Tensor Y4 = MakeUniformWindowReference4(2, 4, 1, 4, 6, 6, 2, -1, true, {}, {}, {}).Y;
    Tensor Y = CollapseReferenceToRank3(Y4);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 1);
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    Expect(registry, std::move(node), "test_cc_attention_3d_local_window", {opset25},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V)}, {std::move(Y)}};
           });
  }

  {
    Tensor Q = MakeConstantFloatTensor({2, 4, 4, 8}, 0.0f);
    Tensor K = MakeConstantFloatTensor({2, 2, 6, 8}, 0.0f);
    Tensor V = MakePositionValueTensor4(2, 2, 6, 6);
    std::vector<uint8_t> mask_values(static_cast<size_t>(2 * 4 * 4 * 6), 1);
    for (int64_t b = 0; b < 2; ++b) {
      for (int64_t h = 0; h < 4; ++h) {
        for (int64_t j = 0; j < 6; ++j) {
          mask_values[static_cast<size_t>(((b * 4 + h) * 4) * 6 + j)] = 0;
        }
      }
    }
    Tensor mask = Tensor::FromBool("", {2, 4, 4, 6}, mask_values);
    const WindowMaskPredicate mask_allows = [](int64_t, int64_t, int64_t i, int64_t) {
      return i != 0;
    };
    UniformWindowReference reference =
        MakeUniformWindowReference4(2, 4, 2, 4, 6, 6, 2, -1, true, {}, {}, mask_allows);
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y", "", "", "qk_matmul_output"});
    AddInt(node, "is_causal", 1);
    AddInt(node, "left_window_size", 2);
    AddFloat(node, "softcap", 2.0f);
    AddInt(node, "softmax_precision", static_cast<int64_t>(DataType::DOUBLE));
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(registry, std::move(node), "test_cc_attention_local_window_gqa_rank4_mask", {opset25},
           [=]() -> IoData {
             return IoData{{std::move(Q), std::move(K), std::move(V), std::move(mask)},
                           {std::move(reference.Y), std::move(reference.probabilities)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
