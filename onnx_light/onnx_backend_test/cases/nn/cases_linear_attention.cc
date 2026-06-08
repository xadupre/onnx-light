// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeLinearAttentionNode(const std::vector<std::string> &inputs,
                                  const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("LinearAttention");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

void RegisterLinearAttentionCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(27);
  const kernel::KernelContext ctx{opset};
  const kernel::LinearAttention kernel{ctx};

  // Small shapes: B=1, T=2, H_q=2, H_kv=2, d_k=2, d_v=2
  // query: (1, 2, 4), key: (1, 2, 4), value: (1, 2, 4)

  Tensor query = Tensor::FromFloat("", {1, 2, 4},
                                   {1.0f, 0.0f, 0.5f, 0.5f,    // t=0: h0=[1,0], h1=[0.5,0.5]
                                    0.0f, 1.0f, 1.0f, -1.0f}); // t=1: h0=[0,1], h1=[1,-1]
  Tensor key = Tensor::FromFloat("", {1, 2, 4}, {1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 1.0f, -0.5f, 0.5f});
  Tensor value =
      Tensor::FromFloat("", {1, 2, 4}, {1.0f, 2.0f, 0.5f, -0.5f, 3.0f, 4.0f, -1.0f, 1.0f});

  // Case 1: linear update rule
  {
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value}, {result.output, result.present_state},
           "test_cc_linear_attention_linear", {opset}, "backend-test", registry);
  }

  // Case 2: gated update rule
  {
    Tensor decay = Tensor::FromFloat("", {1, 2, 2},
                                     {-0.1f, -0.2f,    // t=0: h0=-0.1, h1=-0.2
                                      -0.3f, -0.05f}); // t=1
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, decay}, {result.output, result.present_state},
           "test_cc_linear_attention_gated", {opset}, "backend-test", registry);
  }

  // Case 3: delta update rule
  {
    Tensor beta_t = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, nullptr, &beta_t);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, beta_t}, {result.output, result.present_state},
           "test_cc_linear_attention_delta", {opset}, "backend-test", registry);
  }

  // Case 4: gated_delta update rule (default)
  {
    Tensor decay = Tensor::FromFloat("", {1, 2, 2}, {-0.1f, -0.2f, -0.3f, -0.05f});
    Tensor beta_t = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay, &beta_t);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, decay, beta_t}, {result.output, result.present_state},
           "test_cc_linear_attention_gated_delta", {opset}, "backend-test", registry);
  }

  // Case 5: with past_state
  {
    Tensor past_state = Tensor::FromFloat("", {1, 2, 2, 2},
                                          {0.5f, -0.5f, 0.0f, 0.5f,   // h0
                                           1.0f, 0.0f, -0.5f, 1.0f}); // h1
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, &past_state);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "past_state"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, past_state}, {result.output, result.present_state},
           "test_cc_linear_attention_with_past_state", {opset}, "backend-test", registry);
  }

  // Case 6: GQA — q_num_heads=4, kv_num_heads=2
  {
    // query: (1, 2, 8) = 4 heads * d_k=2, key: (1, 2, 4) = 2 heads * d_k=2
    Tensor q_gqa = Tensor::FromFloat("", {1, 2, 8},
                                     {1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.2f, 0.3f, 0.0f, 1.0f,
                                      1.0f, -1.0f, 0.5f, 0.5f, -0.5f, 0.5f});
    // value: (1, 2, 4) = 2 heads * d_v=2
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    auto result = kernel(q_gqa, key, value, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {q_gqa, key, value}, {result.output, result.present_state},
           "test_cc_linear_attention_gqa", {opset}, "backend-test", registry);
  }

  // Case 7: gated with per-key-dim decay (decay_last = H_kv * d_k)
  {
    Tensor decay_perdim = Tensor::FromFloat(
        "", {1, 2, 4}, {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay_perdim);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, decay_perdim}, {result.output, result.present_state},
           "test_cc_linear_attention_gated_perdim_decay", {opset}, "backend-test", registry);
  }

  // Case 8: explicit scale attribute
  {
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    attrs.has_scale = true;
    attrs.scale = 2.0f;
    auto result = kernel(query, key, value, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    AddAttribute<float>(node, "scale", 2.0f);
    Expect(node, {query, key, value}, {result.output, result.present_state},
           "test_cc_linear_attention_explicit_scale", {opset}, "backend-test", registry);
  }

  // ---------------------------------------------------------------------------
  // Additional cases mirroring upstream ``test_linear_attention_*`` node tests.
  // These exist so the substring check in test_backend_test_names_onnx_vs_onnxlight
  // covers the non-_expanded ONNX entries listed in _backend_test_known_missing.txt.
  // ---------------------------------------------------------------------------

  // Case 9: gated_per_head_decay — gated rule with per-head scalar decay (last dim = H_kv).
  {
    Tensor decay_perhead = Tensor::FromFloat("", {1, 2, 2}, {-0.1f, -0.2f, -0.3f, -0.05f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay_perhead);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, decay_perhead}, {result.output, result.present_state},
           "test_cc_linear_attention_gated_per_head_decay", {opset}, "backend-test", registry);
  }

  // Case 10: gated_delta_beta_scalar — gated_delta with beta last dim = 1.
  {
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta_scalar = Tensor::FromFloat("", {1, 2, 1}, {0.8f, 0.7f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay, &beta_scalar);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, decay, beta_scalar}, {result.output, result.present_state},
           "test_cc_linear_attention_gated_delta_beta_scalar", {opset}, "backend-test", registry);
  }

  // Case 11: gated_delta_gqa — q_num_heads=4, kv_num_heads=2 with gated_delta rule.
  {
    Tensor q_gqa = Tensor::FromFloat("", {1, 2, 8},
                                     {1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.2f, 0.3f, 0.0f, 1.0f,
                                      1.0f, -1.0f, 0.5f, 0.5f, -0.5f, 0.5f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    auto result = kernel(q_gqa, key, value, attrs, nullptr, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {q_gqa, key, value, decay, beta}, {result.output, result.present_state},
           "test_cc_linear_attention_gated_delta_gqa", {opset}, "backend-test", registry);
  }

  // Case 12: gated_delta_mqa — kv_num_heads=1, q_num_heads=4 with gated_delta rule.
  {
    Tensor q_mqa = Tensor::FromFloat("", {1, 2, 8},
                                     {1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.2f, 0.3f, 0.0f, 1.0f,
                                      1.0f, -1.0f, 0.5f, 0.5f, -0.5f, 0.5f});
    Tensor k_mqa = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor v_mqa = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 2}, {-0.1f, -0.2f, -0.05f, -0.1f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 1}, {0.8f, 0.7f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 1;
    auto result = kernel(q_mqa, k_mqa, v_mqa, attrs, nullptr, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 1);
    Expect(node, {q_mqa, k_mqa, v_mqa, decay, beta}, {result.output, result.present_state},
           "test_cc_linear_attention_gated_delta_mqa", {opset}, "backend-test", registry);
  }

  // Case 13: decode_step — T=1 with past_state (gated_delta).
  {
    Tensor q1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 0.0f, 0.5f, 0.5f});
    Tensor k1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 0.0f, 0.5f, 0.5f});
    Tensor v1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 2.0f, 0.5f, -0.5f});
    Tensor past_state =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor decay = Tensor::FromFloat("", {1, 1, 4}, {-0.1f, -0.2f, -0.3f, -0.4f});
    Tensor beta = Tensor::FromFloat("", {1, 1, 2}, {0.8f, 0.9f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(q1, k1, v1, attrs, &past_state, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode(
        {"query", "key", "value", "past_state", "decay", "beta"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {q1, k1, v1, past_state, decay, beta}, {result.output, result.present_state},
           "test_cc_linear_attention_decode_step", {opset}, "backend-test", registry);
  }

  // Case 14: prefill_with_past — T>1 with non-zero past_state (gated_delta).
  {
    Tensor past_state =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, &past_state, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode(
        {"query", "key", "value", "past_state", "decay", "beta"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, past_state, decay, beta},
           {result.output, result.present_state}, "test_cc_linear_attention_prefill_with_past",
           {opset}, "backend-test", registry);
  }

  // Case 15: no_past_explicit_zeros — past_state provided but filled with zeros (gated_delta).
  {
    Tensor past_zeros =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, &past_zeros, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode(
        {"query", "key", "value", "past_state", "decay", "beta"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {query, key, value, past_zeros, decay, beta},
           {result.output, result.present_state}, "test_cc_linear_attention_no_past_explicit_zeros",
           {opset}, "backend-test", registry);
  }

  // Case 16: linear_t1_no_past — linear rule with T=1, no past_state.
  {
    Tensor q1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 0.0f, 0.5f, 0.5f});
    Tensor k1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 0.0f, 0.5f, 0.5f});
    Tensor v1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 2.0f, 0.5f, -0.5f});
    kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(q1, k1, v1, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(node, {q1, k1, v1}, {result.output, result.present_state},
           "test_cc_linear_attention_linear_t1_no_past", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
