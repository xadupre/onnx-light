// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Attention — Y = Softmax((Q @ K^T) * scale, axis=-1) @ V (since opset 23
// in the ``ai.onnx`` domain).
//
// Two cases are registered, both exercising the un-modified baseline of
// the operator (no ``attn_mask``, ``past_key``/``past_value``,
// ``is_causal``, ``softcap`` or ``qk_matmul_output`` features):
//
//   * ``test_cc_attention_basic`` — Multi-Head Attention with
//     ``q_num_heads == kv_num_heads``.
//   * ``test_cc_attention_gqa`` — Grouped Query Attention with
//     ``q_num_heads > kv_num_heads``.
//
// Inputs are small, fully deterministic tensors so this library does not
// depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAttentionCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(23);
  const kernel::KernelContext ctx{opset};
  const kernel::Attention attention{ctx};

  auto make_node = []() {
    NodeProto node;
    node.set_op_type("Attention");
    node.add_input("Q");
    node.add_input("K");
    node.add_input("V");
    node.add_output("Y");
    return node;
  };

  // Case 1: basic MHA, batch_size=1, q_num_heads=kv_num_heads=2,
  // q_seq_len=kv_seq_len=2, head_size=v_head_size=2.
  {
    Tensor Q = Tensor::FromFloat("", {1, 2, 2, 2},
                                 {
                                     // head 0
                                     1.0f, 0.0f, // q0
                                     0.0f, 1.0f, // q1
                                                 // head 1
                                     0.5f, 0.5f, // q0
                                     1.0f, -1.0f // q1
                                 });
    Tensor K = Tensor::FromFloat("", {1, 2, 2, 2},
                                 {
                                     // head 0
                                     1.0f, 0.0f, // k0
                                     0.0f, 1.0f, // k1
                                                 // head 1
                                     1.0f, 1.0f, // k0
                                     -1.0f, 1.0f // k1
                                 });
    Tensor V = Tensor::FromFloat("", {1, 2, 2, 2},
                                 {
                                     // head 0
                                     1.0f, 2.0f,  // v0
                                     3.0f, 4.0f,  // v1
                                                  // head 1
                                     -1.0f, 0.0f, // v0
                                     0.0f, 1.0f   // v1
                                 });
    Tensor Y = attention(Q, K, V);
    NodeProto node = make_node();
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_basic", {opset}, "backend-test", registry);
  }

  // Case 2: GQA, batch_size=1, q_num_heads=4, kv_num_heads=2
  // (group_size=2), q_seq_len=2, kv_seq_len=3, head_size=v_head_size=2.
  {
    Tensor Q = Tensor::FromFloat("", {1, 4, 2, 2},
                                 {
                                     // head 0 (uses kv head 0)
                                     0.1f,
                                     0.2f,
                                     0.3f,
                                     0.4f,
                                     // head 1 (uses kv head 0)
                                     -0.1f,
                                     0.05f,
                                     0.2f,
                                     -0.3f,
                                     // head 2 (uses kv head 1)
                                     0.5f,
                                     0.5f,
                                     0.0f,
                                     1.0f,
                                     // head 3 (uses kv head 1)
                                     1.0f,
                                     0.0f,
                                     0.5f,
                                     -0.5f,
                                 });
    Tensor K = Tensor::FromFloat("", {1, 2, 3, 2},
                                 {
                                     // kv head 0
                                     1.0f,
                                     0.0f,
                                     0.5f,
                                     0.5f,
                                     0.0f,
                                     1.0f,
                                     // kv head 1
                                     -1.0f,
                                     1.0f,
                                     1.0f,
                                     1.0f,
                                     0.25f,
                                     -0.5f,
                                 });
    Tensor V = Tensor::FromFloat("", {1, 2, 3, 2},
                                 {
                                     // kv head 0
                                     1.0f,
                                     0.0f,
                                     0.0f,
                                     1.0f,
                                     -1.0f,
                                     1.0f,
                                     // kv head 1
                                     2.0f,
                                     -2.0f,
                                     0.5f,
                                     0.25f,
                                     -0.5f,
                                     0.0f,
                                 });
    Tensor Y = attention(Q, K, V);
    NodeProto node = make_node();
    Expect(node, {Q, K, V}, {Y}, "test_cc_attention_gqa", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
