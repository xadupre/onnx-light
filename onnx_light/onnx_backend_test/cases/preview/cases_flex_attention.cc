// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/preview/include_preview_kernels.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
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

// Appends a FLOAT scalar ``Constant`` initializer + ``Add(scores, factor)``
// pattern to ``g`` so the resulting subgraph computes ``scores + bias``.
//
//   inputs : (scores [FLOAT, modifier_shape])
//   nodes  : bias = Constant(value=<scalar FLOAT>)
//            scores_out = Add(scores, bias)
//   outputs: (scores_out [FLOAT, modifier_shape])
GraphProto BuildScoreModBias(const std::vector<int64_t> &modifier_shape, float bias_value) {
  GraphProto g;
  g.set_name("score_mod_bias");
  AddModifierIO(g.add_input(), "scores", modifier_shape);

  {
    NodeProto *n = g.add_node();
    n->set_op_type("Constant");
    n->add_output("bias");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->add_t();
    t->set_data_type(TensorProto::DataType::FLOAT);
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &bias_value, sizeof(float));
    t->set_raw_data(utils::ByteSpan(bytes));
  }
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Add");
    n->add_input("scores");
    n->add_input("bias");
    n->add_output("scores_out");
  }

  AddModifierIO(g.add_output(), "scores_out", modifier_shape);
  return g;
}

// Builds a ``score_mod`` subgraph that masks out positions where
// ``k_idx > q_idx`` by replacing the score with ``-inf``. Mirrors
// upstream's ``_make_score_mod_causal_mask_graph`` (Qwen/Gemma/Llama
// causal pattern).
//
//   nodes  : Shape(scores) -> Gather(L_dim, S_dim) -> Range(L), Range(S)
//            -> Reshape into broadcastable q_idx [1,1,L,1], k_idx [1,1,1,S]
//            -> GreaterOrEqual(q_idx, k_idx) -> mask
//            -> Where(mask, scores, neg_inf) -> scores_out
GraphProto BuildScoreModCausalMask(const std::vector<int64_t> &modifier_shape) {
  GraphProto g;
  g.set_name("score_mod_causal_mask");
  AddModifierIO(g.add_input(), "scores", modifier_shape);

  auto add_int64_initializer = [&](const std::string &name, const std::vector<int64_t> &dims,
                                   const std::vector<int64_t> &values) {
    TensorProto *t = g.add_initializer();
    t->set_name(name);
    t->set_data_type(TensorProto::DataType::INT64);
    for (int64_t d : dims) {
      t->add_dims(d);
    }
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    t->set_raw_data(utils::ByteSpan(bytes));
  };
  auto add_float_scalar_initializer = [&](const std::string &name, float value) {
    TensorProto *t = g.add_initializer();
    t->set_name(name);
    t->set_data_type(TensorProto::DataType::FLOAT);
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &value, sizeof(float));
    t->set_raw_data(utils::ByteSpan(bytes));
  };

  add_int64_initializer("zero", {}, {0});
  add_int64_initializer("one", {}, {1});
  add_int64_initializer("idx_2", {}, {2});
  add_int64_initializer("idx_3", {}, {3});
  add_int64_initializer("q_shape", {4}, {1, 1, -1, 1});
  add_int64_initializer("k_shape", {4}, {1, 1, 1, -1});
  add_float_scalar_initializer("neg_inf", -std::numeric_limits<float>::infinity());

  auto add_node = [&](const std::string &op_type, const std::vector<std::string> &inputs,
                      const std::vector<std::string> &outputs) -> NodeProto * {
    NodeProto *n = g.add_node();
    n->set_op_type(op_type);
    for (const auto &i : inputs) {
      n->add_input(i);
    }
    for (const auto &o : outputs) {
      n->add_output(o);
    }
    return n;
  };
  auto add_int_attr = [](NodeProto *n, const std::string &name, int64_t value) {
    AttributeProto *a = n->add_attribute();
    a->set_name(name);
    a->set_type(AttributeProto::AttributeType::INT);
    a->set_i(value);
  };

  add_node("Shape", {"scores"}, {"scores_shape"});
  add_int_attr(add_node("Gather", {"scores_shape", "idx_2"}, {"L_dim"}), "axis", 0);
  add_int_attr(add_node("Gather", {"scores_shape", "idx_3"}, {"S_dim"}), "axis", 0);
  add_node("Range", {"zero", "L_dim", "one"}, {"q_range"});
  add_node("Reshape", {"q_range", "q_shape"}, {"q_idx"});
  add_node("Range", {"zero", "S_dim", "one"}, {"k_range"});
  add_node("Reshape", {"k_range", "k_shape"}, {"k_idx"});
  add_node("GreaterOrEqual", {"q_idx", "k_idx"}, {"mask"});
  add_node("Where", {"mask", "scores", "neg_inf"}, {"scores_out"});

  AddModifierIO(g.add_output(), "scores_out", modifier_shape);
  return g;
}

// Builds a ``score_mod`` subgraph that applies soft capping
// ``scores_out = tanh(scores / cap) * cap`` (Gemma-2 pattern).
GraphProto BuildScoreModSoftCap(const std::vector<int64_t> &modifier_shape, float cap_value) {
  GraphProto g;
  g.set_name("score_mod_soft_cap");
  AddModifierIO(g.add_input(), "scores", modifier_shape);

  {
    TensorProto *t = g.add_initializer();
    t->set_name("cap");
    t->set_data_type(TensorProto::DataType::FLOAT);
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &cap_value, sizeof(float));
    t->set_raw_data(utils::ByteSpan(bytes));
  }

  auto add_node = [&](const std::string &op_type, const std::vector<std::string> &inputs,
                      const std::vector<std::string> &outputs) {
    NodeProto *n = g.add_node();
    n->set_op_type(op_type);
    for (const auto &i : inputs) {
      n->add_input(i);
    }
    for (const auto &o : outputs) {
      n->add_output(o);
    }
  };
  add_node("Div", {"scores", "cap"}, {"scaled"});
  add_node("Tanh", {"scaled"}, {"tanh_out"});
  add_node("Mul", {"tanh_out", "cap"}, {"scores_out"});

  AddModifierIO(g.add_output(), "scores_out", modifier_shape);
  return g;
}

// Builds a ``score_mod`` subgraph that adds the relative-position bias
// ``q_idx - k_idx`` (cast to FLOAT) to the scores.
GraphProto BuildScoreModRelativePositional(const std::vector<int64_t> &modifier_shape) {
  GraphProto g;
  g.set_name("score_mod_relative_positional");
  AddModifierIO(g.add_input(), "scores", modifier_shape);

  auto add_int64_initializer = [&](const std::string &name, const std::vector<int64_t> &dims,
                                   const std::vector<int64_t> &values) {
    TensorProto *t = g.add_initializer();
    t->set_name(name);
    t->set_data_type(TensorProto::DataType::INT64);
    for (int64_t d : dims) {
      t->add_dims(d);
    }
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    t->set_raw_data(utils::ByteSpan(bytes));
  };
  add_int64_initializer("zero", {}, {0});
  add_int64_initializer("one", {}, {1});
  add_int64_initializer("idx_2", {}, {2});
  add_int64_initializer("idx_3", {}, {3});
  add_int64_initializer("q_shape", {2}, {-1, 1});
  add_int64_initializer("k_shape", {2}, {1, -1});

  auto add_node = [&](const std::string &op_type, const std::vector<std::string> &inputs,
                      const std::vector<std::string> &outputs) -> NodeProto * {
    NodeProto *n = g.add_node();
    n->set_op_type(op_type);
    for (const auto &i : inputs) {
      n->add_input(i);
    }
    for (const auto &o : outputs) {
      n->add_output(o);
    }
    return n;
  };
  auto add_int_attr = [](NodeProto *n, const std::string &name, int64_t value) {
    AttributeProto *a = n->add_attribute();
    a->set_name(name);
    a->set_type(AttributeProto::AttributeType::INT);
    a->set_i(value);
  };

  add_node("Shape", {"scores"}, {"scores_shape"});
  add_int_attr(add_node("Gather", {"scores_shape", "idx_2"}, {"L_dim"}), "axis", 0);
  add_int_attr(add_node("Gather", {"scores_shape", "idx_3"}, {"S_dim"}), "axis", 0);
  add_node("Range", {"zero", "L_dim", "one"}, {"q_range"});
  add_node("Reshape", {"q_range", "q_shape"}, {"q_idx"});
  add_node("Range", {"zero", "S_dim", "one"}, {"k_range"});
  add_node("Reshape", {"k_range", "k_shape"}, {"k_idx"});
  add_node("Sub", {"q_idx", "k_idx"}, {"rel_pos"});
  add_int_attr(add_node("Cast", {"rel_pos"}, {"rel_pos_cast"}), "to",
               static_cast<int64_t>(TensorProto::DataType::FLOAT));
  add_node("Add", {"scores", "rel_pos_cast"}, {"scores_out"});

  AddModifierIO(g.add_output(), "scores_out", modifier_shape);
  return g;
}

// Reference implementation of FlexAttention with optional element-wise
// ``score_mod`` / ``prob_mod`` modifiers. Mirrors the math
// ``Y = (prob_mod(Softmax(score_mod((Q@K^T)*scale), axis=-1))) @ V`` over
// rank-4 FLOAT tensors with the same GQA semantics as
// ``kernel::FlexAttention``. The modifier callbacks receive the scalar
// value at position ``(b, h_q, i, j)`` (i.e. batch, query-head,
// query-position, key-position) and return the modified scalar.
//
// This helper computes expected outputs for the test cases without
// requiring ``kernel::FlexAttention`` itself to model ``score_mod``.
using ElementModFn =
    std::function<float(float value, int64_t b, int64_t h_q, int64_t i, int64_t j)>;
Tensor ComputeFlexAttentionExpected(const Tensor &Q, const Tensor &K, const Tensor &V, float scale,
                                    const ElementModFn &score_mod = {},
                                    const ElementModFn &prob_mod = {}) {
  const int64_t batch_size = Q.shape[0];
  const int64_t q_num_heads = Q.shape[1];
  const int64_t q_seq_len = Q.shape[2];
  const int64_t head_size = Q.shape[3];
  const int64_t kv_num_heads = K.shape[1];
  const int64_t kv_seq_len = K.shape[2];
  const int64_t v_head_size = V.shape[3];
  const int64_t group_size = q_num_heads / kv_num_heads;

  const float *pQ = Q.AsFloat();
  const float *pK = K.AsFloat();
  const float *pV = V.AsFloat();

  const int64_t q_head_stride = q_seq_len * head_size;
  const int64_t q_batch_stride = q_num_heads * q_head_stride;
  const int64_t k_head_stride = kv_seq_len * head_size;
  const int64_t k_batch_stride = kv_num_heads * k_head_stride;
  const int64_t v_head_stride = kv_seq_len * v_head_size;
  const int64_t v_batch_stride = kv_num_heads * v_head_stride;
  const int64_t y_head_stride = q_seq_len * v_head_size;
  const int64_t y_batch_stride = q_num_heads * y_head_stride;

  const int64_t out_count = batch_size * q_num_heads * q_seq_len * v_head_size;
  Tensor out("", DataType::FLOAT, {batch_size, q_num_heads, q_seq_len, v_head_size},
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float)));
  float *pY = out.AsFloat();

  std::vector<double> scores(static_cast<size_t>(kv_seq_len));
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < q_num_heads; ++h) {
      const int64_t kv_h = h / group_size;
      const float *Qbh = pQ + b * q_batch_stride + h * q_head_stride;
      const float *Kbh = pK + b * k_batch_stride + kv_h * k_head_stride;
      const float *Vbh = pV + b * v_batch_stride + kv_h * v_head_stride;
      float *Ybh = pY + b * y_batch_stride + h * y_head_stride;

      for (int64_t i = 0; i < q_seq_len; ++i) {
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          double s = 0.0;
          for (int64_t d = 0; d < head_size; ++d) {
            s += static_cast<double>(Qbh[i * head_size + d]) *
                 static_cast<double>(Kbh[j * head_size + d]);
          }
          s *= static_cast<double>(scale);
          if (score_mod) {
            s = static_cast<double>(score_mod(static_cast<float>(s), b, h, i, j));
          }
          scores[static_cast<size_t>(j)] = s;
        }
        // Softmax over the last (kv_seq_len) axis. Handle the all-``-inf``
        // row produced by an exhaustively masked position by leaving the
        // probabilities at zero, matching ONNX's reference semantics.
        double max_score = scores[0];
        for (int64_t j = 1; j < kv_seq_len; ++j) {
          if (scores[static_cast<size_t>(j)] > max_score) {
            max_score = scores[static_cast<size_t>(j)];
          }
        }
        double denom = 0.0;
        if (std::isfinite(max_score)) {
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            const double e = std::exp(scores[static_cast<size_t>(j)] - max_score);
            scores[static_cast<size_t>(j)] = e;
            denom += e;
          }
        } else {
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            scores[static_cast<size_t>(j)] = 0.0;
          }
        }
        const double inv_denom = denom != 0.0 ? 1.0 / denom : 0.0;
        for (int64_t j = 0; j < kv_seq_len; ++j) {
          double p = scores[static_cast<size_t>(j)] * inv_denom;
          if (prob_mod) {
            p = static_cast<double>(prob_mod(static_cast<float>(p), b, h, i, j));
          }
          scores[static_cast<size_t>(j)] = p;
        }

        for (int64_t dv = 0; dv < v_head_size; ++dv) {
          double y = 0.0;
          for (int64_t j = 0; j < kv_seq_len; ++j) {
            y += scores[static_cast<size_t>(j)] * static_cast<double>(Vbh[j * v_head_size + dv]);
          }
          Ybh[i * v_head_size + dv] = static_cast<float>(y);
        }
      }
    }
  }
  return out;
}

} // namespace

// ---------------------------------------------------------------------------
// FlexAttention — Y = Softmax((Q @ K^T) * scale, axis=-1) @ V (since opset 1
// in the ``ai.onnx.preview`` domain).
//
// Ten cases are registered, mirroring the cases shipped by upstream ONNX
// in ``onnx/backend/test/case/node/flexattention.py`` except for the
// ``fp16`` peer (the backend ``Tensor`` storage does not yet support
// ``FLOAT16``) and the ``double`` peer (``kernel::FlexAttention`` only
// supports FLOAT today):
//
//   * ``test_cc_flex_attention_basic`` — Multi-Head Attention with
//     q_num_heads == kv_num_heads. No modifier subgraphs.
//   * ``test_cc_flex_attention_scaled`` — basic MHA shape with an
//     explicit ``scale`` attribute overriding the default
//     ``1 / sqrt(head_size)``.
//   * ``test_cc_flex_attention_gqa`` — Grouped Query Attention with
//     q_num_heads > kv_num_heads. No modifier subgraphs.
//   * ``test_cc_flex_attention_diff_head_sizes`` — basic MHA with the
//     value head size differing from the query/key head size.
//   * ``test_cc_flex_attention_score_mod`` — basic MHA shape with a
//     ``score_mod`` subgraph that adds a scalar bias to every score.
//   * ``test_cc_flex_attention_prob_mod_identity`` — basic MHA shape
//     with a non-empty ``prob_mod`` subgraph that is mathematically the
//     identity (single ``Identity`` node). Exercises the
//     function-body inlining of the modifier without changing the
//     expected output.
//   * ``test_cc_flex_attention_prob_mod_scale_half`` — basic MHA shape
//     with a non-empty ``prob_mod`` subgraph that multiplies the
//     post-Softmax probabilities by a scalar constant ``0.5``. The
//     expected output is ``0.5`` times the un-modified baseline.
//   * ``test_cc_flex_attention_causal_mask`` — ``score_mod`` masks out
//     positions where ``k_idx > q_idx`` by setting their scores to
//     ``-inf`` (Qwen-3/Gemma-3/Llama-3 pattern).
//   * ``test_cc_flex_attention_soft_cap`` — ``score_mod`` applies
//     ``tanh(scores / cap) * cap`` to stabilize attention scores
//     (Gemma-2 pattern).
//   * ``test_cc_flex_attention_relative_positional`` — ``score_mod``
//     adds the relative positional bias ``q_idx - k_idx`` (cast to
//     FLOAT) to the scores.
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

  // ----- Case 5: basic MHA with an explicit ``scale`` attribute
  // overriding the default ``1 / sqrt(head_size)``. Mirrors upstream
  // ``test_flexattention_scaled``.
  {
    constexpr float kExplicitScale = 0.1f;
    Tensor Q =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
    Tensor K =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
    Tensor V =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, 1.0f});
    Tensor Y = flex(Q, K, V, kExplicitScale);
    NodeProto node = make_node();
    AttributeProto *a = node.add_attribute();
    a->set_name("scale");
    a->set_type(AttributeProto::AttributeType::FLOAT);
    a->set_f(kExplicitScale);
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_scaled", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 6: basic MHA with the value head size differing from the
  // query/key head size (v_head_size != head_size). Mirrors upstream
  // ``test_flexattention_diff_head_sizes``.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
    Tensor K =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
    // V: (B, Hkv, S, Dv) with Dv = 3 (different from head_size = 2).
    Tensor V = Tensor::FromFloat("", {1, 2, 2, 3},
                                 {
                                     // kv head 0
                                     1.0f, 2.0f, 3.0f,  // v0
                                     4.0f, 5.0f, 6.0f,  // v1
                                                        // kv head 1
                                     -1.0f, 0.0f, 1.0f, // v0
                                     0.0f, 1.0f, -1.0f, // v1
                                 });
    Tensor Y = flex(Q, K, V);
    NodeProto node = make_node();
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_diff_head_sizes", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 7: basic MHA shape with a ``score_mod`` subgraph that adds
  // a scalar bias of ``0.5`` to every attention score before Softmax.
  // Mirrors upstream ``test_flexattention_score_mod``.
  //
  // Adding a constant to every score in a row leaves the post-Softmax
  // probabilities unchanged, so the expected output mathematically
  // matches the un-modified baseline. We still compute it through
  // ``ComputeFlexAttentionExpected`` with the bias applied to keep the
  // helper in sync with the subgraph semantics.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
    Tensor K =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
    Tensor V =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, 1.0f});
    constexpr float kBias = 0.5f;
    const std::vector<int64_t> modifier_shape = {1, 2, 2, 2};
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(Q.shape[3]));
    Tensor Y = ComputeFlexAttentionExpected(
        Q, K, V, default_scale,
        [kBias](float s, int64_t, int64_t, int64_t, int64_t) { return s + kBias; });
    NodeProto node = make_node();
    AddGraphAttribute(node, "score_mod", BuildScoreModBias(modifier_shape, kBias));
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_score_mod", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 8: basic MHA shape with a ``score_mod`` subgraph that
  // implements causal masking by setting scores at positions where
  // ``k_idx > q_idx`` to ``-inf`` (Qwen-3/Gemma-3/Llama-3 pattern).
  // Mirrors upstream ``test_flexattention_causal_mask``.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
    Tensor K =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
    Tensor V =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, 1.0f});
    const std::vector<int64_t> modifier_shape = {1, 2, 2, 2};
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(Q.shape[3]));
    Tensor Y = ComputeFlexAttentionExpected(
        Q, K, V, default_scale, [](float s, int64_t, int64_t, int64_t i, int64_t j) {
          return j <= i ? s : -std::numeric_limits<float>::infinity();
        });
    NodeProto node = make_node();
    AddGraphAttribute(node, "score_mod", BuildScoreModCausalMask(modifier_shape));
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_causal_mask", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 9: basic MHA shape with a ``score_mod`` subgraph that
  // applies soft capping ``tanh(scores / cap) * cap`` (Gemma-2 pattern).
  // Mirrors upstream ``test_flexattention_soft_cap``.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
    Tensor K =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
    Tensor V =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, 1.0f});
    constexpr float kCap = 20.0f;
    const std::vector<int64_t> modifier_shape = {1, 2, 2, 2};
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(Q.shape[3]));
    Tensor Y = ComputeFlexAttentionExpected(
        Q, K, V, default_scale,
        [kCap](float s, int64_t, int64_t, int64_t, int64_t) { return std::tanh(s / kCap) * kCap; });
    NodeProto node = make_node();
    AddGraphAttribute(node, "score_mod", BuildScoreModSoftCap(modifier_shape, kCap));
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_soft_cap", {default_opset, opset},
           "backend-test", registry);
  }

  // ----- Case 10: basic MHA shape with a ``score_mod`` subgraph that
  // adds the relative positional bias ``q_idx - k_idx`` (cast to FLOAT)
  // to the scores. Mirrors upstream
  // ``test_flexattention_relative_positional``.
  {
    Tensor Q =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, -1.0f});
    Tensor K =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});
    Tensor V =
        Tensor::FromFloat("", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.0f, 1.0f});
    const std::vector<int64_t> modifier_shape = {1, 2, 2, 2};
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(Q.shape[3]));
    Tensor Y = ComputeFlexAttentionExpected(Q, K, V, default_scale,
                                            [](float s, int64_t, int64_t, int64_t i, int64_t j) {
                                              return s + static_cast<float>(i - j);
                                            });
    NodeProto node = make_node();
    AddGraphAttribute(node, "score_mod", BuildScoreModRelativePositional(modifier_shape));
    Expect(node, {Q, K, V}, {Y}, "test_cc_flex_attention_relative_positional",
           {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
