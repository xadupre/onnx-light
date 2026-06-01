// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cmath>
#include <cstdint>
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
                               1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 1.0f,    // head 0
                               -1.0f, 1.0f, 1.0f, 1.0f, 0.25f, -0.5f, // head 1
                           });
}

Tensor MakeV_1_2_3_2() {
  // (batch=1, kv_heads=2, kv_seq=3, v_head_size=2)
  return Tensor::FromFloat("", {1, 2, 3, 2},
                           {
                               1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f,   // head 0
                               2.0f, -2.0f, 0.5f, 0.25f, -0.5f, 0.0f, // head 1
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
  // Case 1: basic MHA (kept for backward compatibility with the original
  // ``test_cc_attention_basic`` registration).
  {
    Tensor Q = MakeQ_1_2_2_2_basic();
    Tensor K = MakeKV_basic_K();
    Tensor V = MakeKV_basic_V();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_basic", {opset}, "backend-test", registry);
  }

  // Case 2: GQA (kept for backward compatibility with the original
  // ``test_cc_attention_gqa`` registration).
  {
    Tensor Q = MakeQ_1_4_2_2_gqa();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_gqa", {opset}, "backend-test", registry);
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_scaled", {opset}, "backend-test", registry);
  }

  // Case 4: ``diff_head_sizes`` — V has a head_size that differs from
  // Q/K's head_size, exercising the asymmetric ``v_head_size`` path.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_3();
    Tensor Y = attention(Q, K, V);
    NodeProto node = MakeAttentionNode({"Q", "K", "V"}, {"Y"});
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_diff_head_sizes", {opset}, "backend-test",
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_causal", {opset}, "backend-test", registry);
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_attn_mask_4d", {opset}, "backend-test",
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_attn_mask_3d", {opset}, "backend-test",
           registry);
  }

  // Case 8: FLOAT 2D ``attn_mask`` broadcast over batch and heads.
  {
    Tensor Q = MakeQ_1_2_2_2();
    Tensor K = MakeK_1_2_3_2();
    Tensor V = MakeV_1_2_3_2();
    Tensor mask = Tensor::FromFloat("", {2, 3}, {0.0f, -0.5f, -1.0f, 0.5f, 0.0f, -0.2f});
    Tensor Y = attention(Q, K, V, /*scale=*/0.5f, mask);
    NodeProto node = MakeAttentionNode({"Q", "K", "V", "attn_mask"}, {"Y"});
    AddFloat(node, "scale", 0.5f);
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_attn_mask_2d", {opset}, "backend-test",
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
    Expect(node, {Q, K, V, mask}, {Y}, "test_cc_attention_attn_mask_bool", {opset}, "backend-test",
           registry);
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
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_softcap", {opset}, "backend-test", registry);
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
           "test_cc_attention_with_past_and_present", {opset}, "backend-test", registry);
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
    Expect(node, {Q, K, V}, {r.Y, r.qk_matmul_output}, "test_cc_attention_with_qk_matmul", {opset},
           "backend-test", registry);
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
           "test_cc_attention_with_qk_matmul_bias", {opset}, "backend-test", registry);
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
    Expect(node, {Q, K, V}, {r.Y, r.qk_matmul_output}, "test_cc_attention_with_qk_matmul_softcap",
           {opset}, "backend-test", registry);
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
    Expect(node, {Q, K, V}, {r.Y, r.qk_matmul_output}, "test_cc_attention_with_qk_matmul_softmax",
           {opset}, "backend-test", registry);
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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
