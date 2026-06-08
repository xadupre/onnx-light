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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
