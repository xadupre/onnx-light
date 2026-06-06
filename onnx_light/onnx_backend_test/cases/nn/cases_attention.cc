// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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

// IEEE-754 binary32 -> binary16 conversion with round-to-nearest-even. Handles
// normals, zeros, subnormals, infinities and NaNs. Only used by the rank-4
// ``test_cc_attention_4d_fp16*`` cases so the helper is intentionally local.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t e32 = static_cast<int32_t>((x >> 23) & 0xffu);
  const uint32_t m32 = x & 0x007fffffu;
  if (e32 == 0xff) {
    // Inf or NaN — preserve sign; collapse the mantissa to a quiet-NaN
    // marker when it was non-zero.
    return static_cast<uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const int32_t e = e32 - 127 + 15;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u); // overflow -> +/-inf
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign); // too small -> +/-0
    }
    // Subnormal: build the implicit leading bit then shift.
    uint32_t m = (m32 | 0x00800000u) >> static_cast<uint32_t>(1 - e);
    const uint32_t round_bit = (m >> 12) & 1u;
    const uint32_t sticky = m & 0x00000fffu;
    uint16_t h = static_cast<uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<uint16_t>(h + 1);
    }
    return h;
  }
  // Normal: pack exponent + truncated mantissa, then round-to-nearest-even.
  const uint32_t low = m32 & 0x1fffu;
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<uint16_t>(h + 1); // mantissa carry naturally bumps exponent
  }
  return h;
}

// Inverse of ``FloatToFloat16Bits`` — decodes an IEEE-754 binary16 bit
// pattern into the corresponding ``float`` value. Matches the FLOAT16 -> FLOAT
// path used by ``kernel::Bernoulli``.
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
      f = (sign << 31) | (static_cast<uint32_t>(e + 127 + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
  }
  float fv;
  std::memcpy(&fv, &f, sizeof(float));
  return fv;
}

// Encodes a FLOAT tensor as a FLOAT16 tensor by round-tripping every element
// through ``FloatToFloat16Bits``. Caller-provided ``name`` becomes the
// tensor name on the resulting ``Tensor``.
Tensor FloatToFloat16Tensor(const std::string &name, const Tensor &f) {
  EXT_ENFORCE_INVALID(f.data_type == DataType::FLOAT, "FloatToFloat16Tensor: input must be FLOAT.");
  const int64_t n = f.element_count();
  std::vector<uint16_t> bits(static_cast<size_t>(n));
  const float *src = f.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    bits[static_cast<size_t>(i)] = FloatToFloat16Bits(src[i]);
  }
  Tensor t = Tensor::FromUint16(name, f.shape, bits);
  t.data_type = static_cast<int32_t>(DataType::FLOAT16);
  return t;
}

// Round-trips every element through ``FloatToFloat16Bits`` / decode and
// returns a fresh FLOAT tensor reflecting the FP16 storage precision. Used
// to simulate the input-side rounding that happens when FLOAT16 tensors are
// fed into a backend that internally promotes to FLOAT.
Tensor RoundToFloat16(const Tensor &f) {
  EXT_ENFORCE_INVALID(f.data_type == DataType::FLOAT, "RoundToFloat16: input must be FLOAT.");
  const int64_t n = f.element_count();
  std::vector<float> rounded(static_cast<size_t>(n));
  const float *src = f.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    rounded[static_cast<size_t>(i)] = Float16BitsToFloat(FloatToFloat16Bits(src[i]));
  }
  return Tensor::FromFloat(f.name, f.shape, rounded);
}

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

} // namespace

void RegisterAttentionCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(23);
  const kernel::KernelContext ctx{opset};
  const kernel::Attention attention{ctx};

  // -------------------------------------------------------------------
  // Case 1: basic MHA (matches upstream ``test_attention_4d``).
  {
    Tensor Q = MakeQ_1_2_2_2_basic();
    Tensor K = MakeKV_basic_K();
    Tensor V = MakeKV_basic_V();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d", {opset}, "backend-test", registry);
  }

  // Case 2: GQA (matches upstream ``test_attention_4d_gqa``).
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_gqa", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Case 3: explicit ``scale`` attribute (1e-2, far from the default
  // 1/sqrt(head_size)). Exercises the scale-attribute path.
  {
    Tensor Q = MakeQ_1_2_2_2_basic();
    Tensor K = MakeKV_basic_K();
    Tensor V = MakeKV_basic_V();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1e-2f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_scaled", {opset}, "backend-test", registry);
  }

  // Case 4: ``diff_heads_sizes`` — V has a head_size that differs from
  // Q/K's head_size, exercising the asymmetric ``v_head_size`` path.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_diff_heads_sizes", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_causal", {opset}, "backend-test", registry);
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask_4d", {opset}, "backend-test",
           registry);
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask_3d", {opset}, "backend-test",
           registry);
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask", {opset}, "backend-test",
           registry);
  }

  // Case 9: BOOL ``attn_mask`` — ``true`` = attend, ``false`` = ``-inf``.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromBool("", {1, 2, 2, 3},
                                   {1, 1, 0, 1, 0, 1,   // head 0
                                    1, 0, 1, 1, 1, 0}); // head 1
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask_bool", {opset},
           "backend-test", registry);
  }

  // Case 10: ``softcap > 0``. ``softcap * tanh(s / softcap)`` is applied
  // between the bias and the softmax. Scale is set to a non-default value
  // to make the saturating effect visible.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_softcap", {opset}, "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(node, {Q, K, V, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_4d_with_past_and_present", {opset}, "backend-test", registry);
  }

  // Case 12: ``qk_matmul_output`` exposed with mode 0 (raw QK^T * scale,
  // no bias / softcap / softmax applied).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    Expect(node, {Q, K, V}, {r.Y, r.qk_matmul_output}, "test_cc_attention_4d_with_qk_matmul",
           {opset}, "backend-test", registry);
  }

  // Case 13: ``qk_matmul_output`` mode 1 — after adding ``attn_mask``.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask);
    NodeProto node =
        MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y", "", "", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(node, {Q, K, V, mask}, {r.Y, r.qk_matmul_output},
           "test_cc_attention_4d_with_qk_matmul_bias", {opset}, "backend-test", registry);
  }

  // Case 14: ``qk_matmul_output`` mode 2 — after the softcap.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    attrs.qk_matmul_output_mode = 2;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 2);
    Expect(node, {Q, K, V}, {r.Y, r.qk_matmul_output},
           "test_cc_attention_4d_with_qk_matmul_softcap", {opset}, "backend-test", registry);
  }

  // Case 15: ``qk_matmul_output`` mode 3 — after softmax (probabilities).
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y", "", "", "qk_matmul_output"});
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(node, {Q, K, V}, {r.Y, r.qk_matmul_output},
           "test_cc_attention_4d_with_qk_matmul_softmax", {opset}, "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d", {opset}, "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_gqa", {opset}, "backend-test", registry);
  }

  // Case 18: rank-3 causal — square q/kv sequence lengths.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 3, 4}, {1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.5f, 0.5f, 0.25f, 0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_causal", {opset}, "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(node, {Q, K, V, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_3d_with_past_and_present", {opset}, "backend-test", registry);
  }

  // -------------------------------------------------------------------
  // Additional rank-3 (fused layout) variants mirroring the upstream
  // ``test_attention_3d_*`` cases.

  // 3D scaled.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1e-2f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_scaled", {opset}, "backend-test", registry);
  }

  // 3D softcap.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_softcap", {opset}, "backend-test", registry);
  }

  // 3D attn_mask (FLOAT, broadcast over batch/heads).
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_3d_attn_mask", {opset}, "backend-test",
           registry);
  }

  // 3D ``diff_heads_sizes`` — V has a different head_size than Q/K.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    // V: (1, 3, 6) ← collapse of (1, 2, 3, 3) so v_head_size=3.
    Tensor V = Tensor::FromFloat("", {1, 3, 6},
                                 {1.0f, 0.0f, -1.0f, 2.0f, -2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.5f,
                                  0.25f, -0.25f, -1.0f, 1.0f, 0.5f, -0.5f, 0.0f, 1.0f});
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_diff_heads_sizes", {opset}, "backend-test",
           registry);
  }

  // 3D GQA + scaled.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1e-2f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_gqa_scaled", {opset}, "backend-test",
           registry);
  }

  // 3D GQA + softcap.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_gqa_softcap", {opset}, "backend-test",
           registry);
  }

  // 3D GQA + attn_mask.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 8},
                                 {0.1f, 0.2f, -0.1f, 0.05f, 0.5f, 0.5f, 1.0f, 0.0f, 0.3f, 0.4f,
                                  0.2f, -0.3f, 0.0f, 1.0f, 0.5f, -0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_3d_gqa_attn_mask", {opset},
           "backend-test", registry);
  }

  // 3D GQA + causal — square q/kv sequence lengths.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 3, 8}, {0.1f, 0.2f,  -0.1f, 0.05f, 0.5f, 0.5f, 1.0f,  0.0f,
                                          0.3f, 0.4f,  0.2f,  -0.3f, 0.0f, 1.0f, 0.5f,  -0.5f,
                                          0.2f, -0.1f, 0.25f, 0.0f,  0.5f, 0.5f, -1.0f, 1.0f});
    Tensor K = rank3_K();
    Tensor V = rank3_V();
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_gqa_causal", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddInt(node, "q_num_heads", 4);
    AddInt(node, "kv_num_heads", 2);
    Expect(node, {Q, K, V, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_3d_gqa_with_past_and_present", {opset}, "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    Expect(node, {Q, K, V, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_3d_with_past_and_present_qk_matmul", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V, mask, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_3d_with_past_and_present_qk_matmul_bias", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_3d_with_past_and_present_qk_matmul_softcap", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "qk_matmul_output_mode", 3);
    Expect(node, {Q, K, V, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_3d_with_past_and_present_qk_matmul_softmax", {opset}, "backend-test",
           registry);
  }

  // -------------------------------------------------------------------
  // Additional rank-4 variants mirroring the upstream
  // ``test_attention_4d_*`` cases.

  // 4D GQA + scaled.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1e-2f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_gqa_scaled", {opset}, "backend-test",
           registry);
  }

  // 4D GQA + softcap.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_gqa_softcap", {opset}, "backend-test",
           registry);
  }

  // 4D GQA + attn_mask (FLOAT, broadcast over heads).
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_gqa_attn_mask", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_gqa_causal", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(node, {Q, K, V, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_4d_gqa_with_past_and_present", {opset}, "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask_4d_causal", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask_3d_causal", {opset},
           "backend-test", registry);
  }

  // 4D BOOL ``attn_mask`` with 4D shape.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromBool("", {1, 2, 2, 3},
                                   {1, 1, 0, 1, 0, 1,   // head 0
                                    1, 0, 1, 1, 1, 0}); // head 1
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_attn_mask_bool_4d", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    Expect(node, {Q, K, V, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_4d_with_past_and_present_qk_matmul", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(node, {Q, K, V, mask, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(node, {Q, K, V, mask, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value", "qk_matmul_output"});
    AddFloat(node, "scale", 0.5f);
    AddInt(node, "qk_matmul_output_mode", 1);
    Expect(node, {Q, K, V, mask, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V, mask, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V, mask, past_key, past_value},
           {r.Y, r.present_key, r.present_value, r.qk_matmul_output},
           "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal", {opset},
           "backend-test", registry);
  }

  // 4D ``diff_heads_sizes`` + scaled.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1e-2f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_diff_heads_sizes_scaled", {opset},
           "backend-test", registry);
  }

  // 4D ``diff_heads_sizes`` + softcap.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 1.0f;
    attrs.softcap = 0.5f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddFloat(node, "scale", 1.0f);
    AddFloat(node, "softcap", 0.5f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_diff_heads_sizes_softcap", {opset},
           "backend-test", registry);
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_diff_heads_sizes_attn_mask", {opset},
           "backend-test", registry);
  }

  // 4D ``diff_heads_sizes`` + causal — square q/kv lengths.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 2, 3, 2},
        {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 1.0f, -1.0f, 0.25f, 0.5f});
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    kernel::Attention::Attributes attrs;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_diff_heads_sizes_causal", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    Expect(node, {Q, K, V, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_4d_diff_heads_with_past_and_present", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_4d_diff_heads_with_past_and_present_mask3d", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    auto r = attention(Q, K, V, attrs, &mask, &past_key, &past_value);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask", "past_key", "past_value"},
                                       {"Y", "present_key", "present_value"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_4d_diff_heads_with_past_and_present_mask4d", {opset}, "backend-test",
           registry);
  }

  // 4D ``diff_heads`` with 4D mask serving as padded-KV mask.
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    // Mask is broadcastable over heads; final column is -inf-like (-1e4)
    // to emulate KV-padding suppression on the trailing position.
    Tensor mask = Tensor::FromFloat("", {1, 1, 2, 3}, {0.0f, 0.0f, -1.0e4f, 0.0f, 0.0f, -1.0e4f});
    kernel::Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_4d_diff_heads_mask4d_padded_kv", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 1e-2f;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 1e-2f);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_diff_heads_sizes_scaled", {opset},
           "backend-test", registry);
  }

  // 3D diff_heads_sizes + softcap.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_diff_heads_sizes_softcap", {opset},
           "backend-test", registry);
  }

  // 3D diff_heads_sizes + attn_mask.
  {
    Tensor Q = rank3_inputs();
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 0.5f;
    Tensor Y = attention(Q, K, V, attrs, &mask).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_3d_diff_heads_sizes_attn_mask", {opset},
           "backend-test", registry);
  }

  // 3D diff_heads_sizes + causal — square q/kv lengths.
  {
    Tensor Q = Tensor::FromFloat(
        "", {1, 3, 4}, {1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.5f, 0.5f, 0.25f, 0.5f});
    Tensor K = rank3_K();
    Tensor V = rank3_diff_heads_V();
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.is_causal = true;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", 2);
    AddInt(node, "kv_num_heads", 2);
    AddInt(node, "is_causal", 1);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_diff_heads_sizes_causal", {opset},
           "backend-test", registry);
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
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V, mask, past_key, past_value}, {r.Y, r.present_key, r.present_value},
           "test_cc_attention_3d_diff_heads_with_past_and_present", {opset}, "backend-test",
           registry);
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
    kernel::Attention::Attributes attrs;
    attrs.q_num_heads = q_num_heads;
    attrs.kv_num_heads = kv_num_heads;
    Tensor Y = attention(Q, K, V, attrs).Y;
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    AddInt(node, "q_num_heads", q_num_heads);
    AddInt(node, "kv_num_heads", kv_num_heads);
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_3d_transpose_verification", {opset},
           "backend-test", registry);
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_4d_fp16", {opset}, "backend-test", registry);
    registry.back().atol = 5e-3;
    registry.back().rtol = 5e-3;
  }

  // 4D GQA + past/present, fp16.
  {
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
    kernel::Attention::Attributes attrs;
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
    Expect(node, {Q, K, V, mask, pk, pv}, {Y, present_key, present_value},
           "test_cc_attention_4d_gqa_with_past_and_present_fp16", {opset}, "backend-test",
           registry);
    registry.back().atol = 5e-3;
    registry.back().rtol = 5e-3;
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
