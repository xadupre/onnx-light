// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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

void RegisterLinearAttentionCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(27);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::LinearAttention kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 8);
    AddAttribute<int64_t>(node, "kv_num_heads", 8);

    constexpr int64_t qkv_count = 1 * 512 * 512;
    constexpr int64_t state_count = 1 * 8 * 64 * 64;
    Expect(registry, std::move(node), "test_cc_linear_attention_linear_benchmark", {opset},
           {qkv_count, qkv_count, qkv_count}, {qkv_count, state_count}, [kernel]() -> IoData {
             Tensor query = RandnTensor(DataType::FLOAT, {1, 512, 512}, 2701);
             Tensor key = RandnTensor(DataType::FLOAT, {1, 512, 512}, 2702);
             Tensor value = RandnTensor(DataType::FLOAT, {1, 512, 512}, 2703);
             onnx_kernels::kernel::LinearAttention::Attributes attrs;
             attrs.update_rule = "linear";
             attrs.q_num_heads = 8;
             attrs.kv_num_heads = 8;
             auto result = kernel(query, key, value, attrs);
             return IoData{{std::move(query), std::move(key), std::move(value)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
    return;
  }

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
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_linear", {opset}, [=]() -> IoData {
      return IoData{{std::move(query), std::move(key), std::move(value)},
                    {std::move(result.output), std::move(result.present_state)}};
    });
  }

  // Case 2: gated update rule
  {
    Tensor decay = Tensor::FromFloat("", {1, 2, 2},
                                     {-0.1f, -0.2f,    // t=0: h0=-0.1, h1=-0.2
                                      -0.3f, -0.05f}); // t=1
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated", {opset}, [=]() -> IoData {
      return IoData{{std::move(query), std::move(key), std::move(value), std::move(decay)},
                    {std::move(result.output), std::move(result.present_state)}};
    });
  }

  // Case 3: delta update rule
  {
    Tensor beta_t = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, nullptr, &beta_t);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_delta", {opset}, [=]() -> IoData {
      return IoData{{std::move(query), std::move(key), std::move(value), std::move(beta_t)},
                    {std::move(result.output), std::move(result.present_state)}};
    });
  }

  // Case 4: gated_delta update rule (default)
  {
    Tensor decay = Tensor::FromFloat("", {1, 2, 2}, {-0.1f, -0.2f, -0.3f, -0.05f});
    Tensor beta_t = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay, &beta_t);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated_delta", {opset},
           [=]() -> IoData {
             return IoData{{std::move(query), std::move(key), std::move(value), std::move(decay),
                            std::move(beta_t)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 5: with past_state
  {
    Tensor past_state = Tensor::FromFloat("", {1, 2, 2, 2},
                                          {0.5f, -0.5f, 0.0f, 0.5f,   // h0
                                           1.0f, 0.0f, -0.5f, 1.0f}); // h1
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, &past_state);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "past_state"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_with_past_state", {opset},
           [=]() -> IoData {
             return IoData{
                 {std::move(query), std::move(key), std::move(value), std::move(past_state)},
                 {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 6: GQA — q_num_heads=4, kv_num_heads=2
  {
    // query: (1, 2, 8) = 4 heads * d_k=2, key: (1, 2, 4) = 2 heads * d_k=2
    Tensor q_gqa = Tensor::FromFloat("", {1, 2, 8},
                                     {1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.2f, 0.3f, 0.0f, 1.0f,
                                      1.0f, -1.0f, 0.5f, 0.5f, -0.5f, 0.5f});
    // value: (1, 2, 4) = 2 heads * d_v=2
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    auto result = kernel(q_gqa, key, value, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gqa", {opset}, [=]() -> IoData {
      return IoData{{std::move(q_gqa), std::move(key), std::move(value)},
                    {std::move(result.output), std::move(result.present_state)}};
    });
  }

  // Case 7: gated with per-key-dim decay (decay_last = H_kv * d_k)
  {
    Tensor decay_perdim = Tensor::FromFloat(
        "", {1, 2, 4}, {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay_perdim);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated_perdim_decay", {opset},
           [=]() -> IoData {
             return IoData{
                 {std::move(query), std::move(key), std::move(value), std::move(decay_perdim)},
                 {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 8: explicit scale attribute
  {
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
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
    Expect(registry, std::move(node), "test_cc_linear_attention_explicit_scale", {opset},
           [=]() -> IoData {
             return IoData{{std::move(query), std::move(key), std::move(value)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // ---------------------------------------------------------------------------
  // Additional cases mirroring upstream ``test_linear_attention_*`` node tests.
  // These exist so the substring check in test_backend_test_names_onnx_vs_onnxlight
  // covers the non-_expanded ONNX entries listed in _backend_test_known_missing.txt.
  // ---------------------------------------------------------------------------

  // Case 9: gated_per_head_decay — gated rule with per-head scalar decay (last dim = H_kv).
  {
    Tensor decay_perhead = Tensor::FromFloat("", {1, 2, 2}, {-0.1f, -0.2f, -0.3f, -0.05f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay_perhead);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated_per_head_decay", {opset},
           [=]() -> IoData {
             return IoData{
                 {std::move(query), std::move(key), std::move(value), std::move(decay_perhead)},
                 {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 10: gated_delta_beta_scalar — gated_delta with beta last dim = 1.
  {
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta_scalar = Tensor::FromFloat("", {1, 2, 1}, {0.8f, 0.7f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, nullptr, &decay, &beta_scalar);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated_delta_beta_scalar", {opset},
           [=]() -> IoData {
             return IoData{{std::move(query), std::move(key), std::move(value), std::move(decay),
                            std::move(beta_scalar)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 11: gated_delta_gqa — q_num_heads=4, kv_num_heads=2 with gated_delta rule.
  {
    Tensor q_gqa = Tensor::FromFloat("", {1, 2, 8},
                                     {1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.2f, 0.3f, 0.0f, 1.0f,
                                      1.0f, -1.0f, 0.5f, 0.5f, -0.5f, 0.5f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 2;
    auto result = kernel(q_gqa, key, value, attrs, nullptr, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated_delta_gqa", {opset},
           [=]() -> IoData {
             return IoData{{std::move(q_gqa), std::move(key), std::move(value), std::move(decay),
                            std::move(beta)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
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
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 4;
    attrs.kv_num_heads = 1;
    auto result = kernel(q_mqa, k_mqa, v_mqa, attrs, nullptr, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 4);
    AddAttribute<int64_t>(node, "kv_num_heads", 1);
    Expect(registry, std::move(node), "test_cc_linear_attention_gated_delta_mqa", {opset},
           [=]() -> IoData {
             return IoData{{std::move(q_mqa), std::move(k_mqa), std::move(v_mqa), std::move(decay),
                            std::move(beta)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
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
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(q1, k1, v1, attrs, &past_state, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode(
        {"query", "key", "value", "past_state", "decay", "beta"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_decode_step", {opset},
           [=]() -> IoData {
             return IoData{{std::move(q1), std::move(k1), std::move(v1), std::move(past_state),
                            std::move(decay), std::move(beta)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 14: prefill_with_past — T>1 with non-zero past_state (gated_delta).
  {
    Tensor past_state =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.5f, -0.5f, 0.0f, 0.5f, 1.0f, 0.0f, -0.5f, 1.0f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, &past_state, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode(
        {"query", "key", "value", "past_state", "decay", "beta"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_prefill_with_past", {opset},
           [=]() -> IoData {
             return IoData{{std::move(query), std::move(key), std::move(value),
                            std::move(past_state), std::move(decay), std::move(beta)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 15: no_past_explicit_zeros — past_state provided but filled with zeros (gated_delta).
  {
    Tensor past_zeros =
        Tensor::FromFloat("", {1, 2, 2, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor decay = Tensor::FromFloat("", {1, 2, 4},
                                     {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(query, key, value, attrs, &past_zeros, &decay, &beta);
    NodeProto node = MakeLinearAttentionNode(
        {"query", "key", "value", "past_state", "decay", "beta"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_no_past_explicit_zeros", {opset},
           [=]() -> IoData {
             return IoData{{std::move(query), std::move(key), std::move(value),
                            std::move(past_zeros), std::move(decay), std::move(beta)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 16: linear_t1_no_past — linear rule with T=1, no past_state.
  {
    Tensor q1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 0.0f, 0.5f, 0.5f});
    Tensor k1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 0.0f, 0.5f, 0.5f});
    Tensor v1 = Tensor::FromFloat("", {1, 1, 4}, {1.0f, 2.0f, 0.5f, -0.5f});
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "linear";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(q1, k1, v1, attrs);
    NodeProto node =
        MakeLinearAttentionNode({"query", "key", "value"}, {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "linear");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_linear_t1_no_past", {opset},
           [=]() -> IoData {
             return IoData{{std::move(q1), std::move(k1), std::move(v1)},
                           {std::move(result.output), std::move(result.present_state)}};
           });
  }

  // Case 17: fp16 — half-precision activations (gated_delta). The kernel
  // promotes FLOAT16 inputs to FLOAT32, runs the recurrence, then demotes the
  // outputs back to FLOAT16, so the reference outputs are computed from the
  // FP16-rounded inputs in float32 and rounded back to FLOAT16.
  {
    Tensor q16 = RoundToFloat16(query);
    Tensor k16 = RoundToFloat16(key);
    Tensor v16 = RoundToFloat16(value);
    Tensor decay32 = Tensor::FromFloat("", {1, 2, 4},
                                       {-0.1f, -0.2f, -0.3f, -0.4f, -0.05f, -0.1f, -0.15f, -0.2f});
    Tensor beta32 = Tensor::FromFloat("", {1, 2, 2}, {0.8f, 0.9f, 0.7f, 0.6f});
    Tensor decay16 = RoundToFloat16(decay32);
    Tensor beta16 = RoundToFloat16(beta32);
    onnx_kernels::kernel::LinearAttention::Attributes attrs;
    attrs.update_rule = "gated_delta";
    attrs.q_num_heads = 2;
    attrs.kv_num_heads = 2;
    auto result = kernel(q16, k16, v16, attrs, nullptr, &decay16, &beta16);
    NodeProto node = MakeLinearAttentionNode({"query", "key", "value", "", "decay", "beta"},
                                             {"output", "present_state"});
    AddAttribute<std::string>(node, "update_rule", "gated_delta");
    AddAttribute<int64_t>(node, "q_num_heads", 2);
    AddAttribute<int64_t>(node, "kv_num_heads", 2);
    Expect(registry, std::move(node), "test_cc_linear_attention_fp16", {opset}, [=]() -> IoData {
      return IoData{
          {std::move(FloatToFloat16Tensor("", q16)), std::move(FloatToFloat16Tensor("", k16)),
           std::move(FloatToFloat16Tensor("", v16)), std::move(FloatToFloat16Tensor("", decay16)),
           std::move(FloatToFloat16Tensor("", beta16))},
          {std::move(FloatToFloat16Tensor("", result.output)),
           std::move(FloatToFloat16Tensor("", result.present_state))}};
    });
    registry.back().atol = 5e-3;
    registry.back().rtol = 5e-3;
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
