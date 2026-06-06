// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/preview/include_preview_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Opset id for the ``ai.onnx.preview`` domain at the only released version
// of FlexAttention (v1). Mirrors ``onnx_op::preview::kOnnxPreviewDomain``
// but is duplicated here so this library does not need to depend on
// ``lib_onnx_op``.
constexpr const char *kOnnxPreviewDomain = "ai.onnx.preview";

OpsetId PreviewOpset(int64_t version) { return OpsetId(kOnnxPreviewDomain, version); }

} // namespace

// ---------------------------------------------------------------------------
// FlexAttention — Y = Softmax((Q @ K^T) * scale, axis=-1) @ V (since opset 1
// in the ``ai.onnx.preview`` domain).
//
// Two cases are registered, both exercising the unmodified baseline of the
// operator (no ``score_mod`` / ``prob_mod`` modifier subgraphs):
//
//   * ``test_cc_flex_attention_basic`` — Multi-Head Attention with
//     q_num_heads == kv_num_heads.
//   * ``test_cc_flex_attention_gqa`` — Grouped Query Attention with
//     q_num_heads > kv_num_heads.
//
// Inputs are small, fully deterministic tensors so this library does not
// depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterFlexAttentionCases(std::vector<TestCase> &registry) {
  const OpsetId opset = PreviewOpset(1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::FlexAttention flex{ctx};

  auto make_node = []() {
    NodeProto node;
    node.set_op_type("FlexAttention");
    node.set_domain(kOnnxPreviewDomain);
    node.add_input("Q");
    node.add_input("K");
    node.add_input("V");
    node.add_output("Y");
    return node;
  };

  // ----- Case 1: basic MHA, batch_size=1, q_num_heads=kv_num_heads=2,
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
                                     1.0f,
                                     0.0f, // k0
                                     0.0f,
                                     1.0f, // k1
                                           // head 1
                                     1.0f,
                                     1.0f, // k0
                                     -1.0f,
                                     1.0f, // k1
                                 });
    Tensor V = Tensor::FromFloat("", {1, 2, 2, 2},
                                 {
                                     // head 0
                                     1.0f,
                                     2.0f, // v0
                                     3.0f,
                                     4.0f, // v1
                                           // head 1
                                     -1.0f,
                                     0.0f, // v0
                                     0.0f,
                                     1.0f, // v1
                                 });
    Tensor Y = flex(Q, K, V);
    NodeProto node = make_node();
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_basic", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 2: GQA, batch_size=1, q_num_heads=4, kv_num_heads=2
  // (group_size=2), q_seq_len=2, kv_seq_len=3, head_size=v_head_size=2.
  {
    // Q: (1, 4, 2, 2)
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
    // K: (1, 2, 3, 2)
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
    // V: (1, 2, 3, 2)
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
    Tensor Y = flex(Q, K, V);
    NodeProto node = make_node();
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_gqa", {default_opset, opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
