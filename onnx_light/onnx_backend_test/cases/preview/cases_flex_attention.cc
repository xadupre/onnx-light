// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/preview/include_preview_kernels.h"

#include <cstring>
#include <string>
#include <utility>
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

// Adds a tensor-typed value-info (rank-4) named ``name`` to ``g``. Matches
// the FLOAT softmax_precision tensor shape ``(B, Hq, Lq, Lkv)`` carried in
// and out of the ``score_mod`` / ``prob_mod`` modifier subgraphs.
void AddModifierIO(ValueInfoProto *vi, const std::string &name, const std::vector<int64_t> &shape) {
  vi->set_name(name);
  TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto &sh = tt->ref_shape();
  for (int64_t d : shape) {
    sh.add_dim()->set_dim_value(d);
  }
}

// Builds a non-empty ``prob_mod`` subgraph that is mathematically the
// identity:
//
//   inputs : (probs [FLOAT, modifier_shape])
//   nodes  : out = Identity(probs)
//   outputs: (out [FLOAT, modifier_shape])
//
// Has ``node_size() > 0`` so ``IsIdentityModifierGraph`` returns false and
// the function-body builder inlines the call. Mathematically a no-op so
// the expected output equals the un-modified baseline.
GraphProto BuildIdentityProbMod(const std::vector<int64_t> &modifier_shape) {
  GraphProto g;
  g.set_name("prob_mod");
  AddModifierIO(g.add_input(), "probs", modifier_shape);

  NodeProto *n = g.add_node();
  n->set_op_type("Identity");
  n->add_input("probs");
  n->add_output("out");

  AddModifierIO(g.add_output(), "out", modifier_shape);
  return g;
}

// Builds a non-empty ``prob_mod`` subgraph that scales the probabilities
// by a scalar FLOAT constant ``factor``:
//
//   inputs : (probs [FLOAT, modifier_shape])
//   nodes  : factor = Constant(value=<scalar FLOAT>)
//            out = Mul(probs, factor)
//   outputs: (out [FLOAT, modifier_shape])
//
// Since matrix-multiplication with V is linear in ``probs``, the
// modified output equals ``factor`` times the un-modified baseline.
GraphProto BuildScaleProbMod(const std::vector<int64_t> &modifier_shape, float factor) {
  GraphProto g;
  g.set_name("prob_mod");
  AddModifierIO(g.add_input(), "probs", modifier_shape);

  // factor = Constant(value=<scalar FLOAT>)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Constant");
    n->add_output("factor");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->add_t();
    t->set_data_type(TensorProto::DataType::FLOAT);
    // scalar tensor: no dims.
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &factor, sizeof(float));
    t->set_raw_data(utils::ByteSpan(bytes));
  }
  // out = Mul(probs, factor)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Mul");
    n->add_input("probs");
    n->add_input("factor");
    n->add_output("out");
  }

  AddModifierIO(g.add_output(), "out", modifier_shape);
  return g;
}

// Adds a GRAPH-typed attribute named ``name`` to ``node`` whose subgraph
// is ``body``.
void AddGraphAttribute(NodeProto &node, const std::string &name, GraphProto body) {
  AttributeProto *a = node.add_attribute();
  a->set_name(name);
  a->set_type(AttributeProto::AttributeType::GRAPH);
  *a->add_g() = std::move(body);
}

} // namespace

// ---------------------------------------------------------------------------
// FlexAttention — Y = Softmax((Q @ K^T) * scale, axis=-1) @ V (since opset 1
// in the ``ai.onnx.preview`` domain).
//
// Four cases are registered:
//
//   * ``test_cc_flex_attention_basic`` — Multi-Head Attention with
//     q_num_heads == kv_num_heads. No modifier subgraphs.
//   * ``test_cc_flex_attention_gqa`` — Grouped Query Attention with
//     q_num_heads > kv_num_heads. No modifier subgraphs.
//   * ``test_cc_flex_attention_prob_mod_identity`` — basic MHA shape
//     with a non-empty ``prob_mod`` subgraph that is mathematically the
//     identity (single ``Identity`` node). Exercises the
//     function-body inlining of the modifier without changing the
//     expected output.
//   * ``test_cc_flex_attention_prob_mod_scale_half`` — basic MHA shape
//     with a non-empty ``prob_mod`` subgraph that multiplies the
//     post-Softmax probabilities by a scalar constant ``0.5``. The
//     expected output is ``0.5`` times the un-modified baseline.
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

  // ----- Case 3: basic MHA shape with a non-empty ``prob_mod`` subgraph
  // that is structurally non-trivial (a single ``Identity`` node, so
  // ``IsIdentityModifierGraph`` returns false and the function-body
  // builder inlines the call) but mathematically a no-op. The expected
  // output therefore matches the un-modified baseline.
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
                                     0.0f, 1.0f,  // v1
                                 });
    // Modifier shape: (B, Hq, Lq, Lkv) = (1, 2, 2, 2).
    const std::vector<int64_t> modifier_shape = {1, 2, 2, 2};
    Tensor Y = flex(Q, K, V);
    NodeProto node = make_node();
    AddGraphAttribute(node, "prob_mod", BuildIdentityProbMod(modifier_shape));
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_prob_mod_identity", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 4: basic MHA shape with a non-empty ``prob_mod`` subgraph
  // that scales the post-Softmax probabilities by a scalar constant
  // (``Mul(probs, 0.5)``). Since the final ``Probs @ V`` matmul is linear
  // in ``Probs``, the expected output is exactly ``0.5`` times the
  // un-modified baseline.
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
                                     0.0f, 1.0f,  // v1
                                 });
    const std::vector<int64_t> modifier_shape = {1, 2, 2, 2};
    constexpr float kScale = 0.5f;

    // Compute expected: baseline output scaled by ``kScale``.
    Tensor Y_baseline = flex(Q, K, V);
    Tensor Y = Tensor::FromFloat("", Y_baseline.shape,
                                 std::vector<float>(Y_baseline.element_count(), 0.0f));
    {
      const float *src = Y_baseline.AsFloat();
      float *dst = Y.AsFloat();
      for (int64_t i = 0; i < Y_baseline.element_count(); ++i) {
        dst[i] = kScale * src[i];
      }
    }

    NodeProto node = make_node();
    AddGraphAttribute(node, "prob_mod", BuildScaleProbMod(modifier_shape, kScale));
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_prob_mod_scale_half",
           {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
