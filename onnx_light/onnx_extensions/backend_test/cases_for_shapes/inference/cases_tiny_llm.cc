// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/inference_random_weights.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 13;

// Tiny "Llama"-style configuration. The values are intentionally small so the
// generated model stays compact while keeping the architecture faithful:
// hidden_size == num_heads * head_size and the SwiGLU MLP widens to
// ``intermediate_size`` before projecting back to ``hidden_size``.
constexpr int64_t kVocabSize = 32;
constexpr int64_t kHiddenSize = 16;
constexpr int64_t kNumHeads = 4;
constexpr int64_t kHeadSize = kHiddenSize / kNumHeads; // 4
constexpr int64_t kIntermediateSize = 32;

} // namespace

// ---------------------------------------------------------------------------
// ``arnir0/Tiny-LLM`` — a single decoder layer of a tiny Llama-style causal
// language model translated to ONNX. The graph takes the four inputs a
// cached-generation step of such a model expects and produces the next-token
// logits together with the updated key/value cache:
//
//   input_ids       int64[batch, seq]
//   attention_mask  int64[batch, total_seq]
//   past_key        float[batch, n_heads, past_seq, head_size]
//   past_value      float[batch, n_heads, past_seq, head_size]
//
//   logits          float[batch, seq, vocab_size]
//   present_key     float[batch, n_heads, total_seq, head_size]
//   present_value   float[batch, n_heads, total_seq, head_size]
//
// Every dimension that varies at inference time (``batch``, ``seq``,
// ``past_seq``, ``total_seq``) is declared with a symbolic ``dim_param`` so
// the case exercises shape inference under fully dynamic shapes. All learnable
// tensors (token embedding, the four attention projections, the SwiGLU MLP
// matrices, the three RMSNorm scales and the LM head) are random initializers.
//
// Architecture (one Llama decoder block + LM head):
//
//   hidden  = Gather(embed_tokens, input_ids)
//   h1      = RMSNorm(hidden)
//   q,k,v   = MatMul(h1, {q,k,v}_proj)
//   attn    = Attention(q, k, v, attn_bias, past_key, past_value)
//   h       = hidden + MatMul(attn, o_proj)
//   h2      = RMSNorm(h)
//   mlp     = MatMul(SiLU(MatMul(h2, gate)) * MatMul(h2, up), down)
//   h       = h + mlp
//   logits  = MatMul(RMSNorm(h), lm_head)
//
// where ``attn_bias`` is the additive float mask derived from the 2-D
// ``attention_mask`` input. The case carries no ``DataSet``: it is a
// shape-inference-only model, so the generic
// ``BackendTestCaseShapeInference`` test simply checks that
// :cpp:func:`shape_inference::InferShapes` recovers the recorded
// ``value_info`` / output shapes.
// ---------------------------------------------------------------------------
void RegisterTinyLlmShapeInferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(23);

  const std::string name = "test_cc_shape_inference_tiny_llm";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  const TensorProto::DataType kFloat = DataType::FLOAT;
  const TensorProto::DataType kInt64 = DataType::INT64;

  // ---- Token embedding -------------------------------------------------
  AddNode(*graph, "Gather", {"embed_tokens.weight", "input_ids"}, {"hidden"});

  // ---- Input RMSNorm + QKV projections ---------------------------------
  AddNode(*graph, "RMSNormalization", {"hidden", "input_layernorm.weight"}, {"normed1"});
  AddNode(*graph, "MatMul", {"normed1", "q_proj.weight"}, {"query"});
  AddNode(*graph, "MatMul", {"normed1", "k_proj.weight"}, {"key"});
  AddNode(*graph, "MatMul", {"normed1", "v_proj.weight"}, {"value"});

  // ---- Additive attention mask derived from attention_mask -------------
  // attn_bias = (1 - cast(attention_mask)) * -1e9, broadcast to a 4-D mask.
  NodeProto &cast_node = AddNode(*graph, "Cast", {"attention_mask"}, {"mask_float"});
  AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(DataType::FLOAT));
  AddNode(*graph, "Unsqueeze", {"mask_float", "mask_axes"}, {"mask_4d"});
  AddNode(*graph, "Sub", {"mask_one", "mask_4d"}, {"mask_inv"});
  AddNode(*graph, "Mul", {"mask_inv", "mask_neg"}, {"attn_bias"});

  // ---- Self-attention with KV cache ------------------------------------
  NodeProto &attn_node =
      AddNode(*graph, "Attention", {"query", "key", "value", "attn_bias", "past_key", "past_value"},
              {"attn_out", "present_key", "present_value"});
  AddAttribute<int64_t>(attn_node, "q_num_heads", kNumHeads);
  AddAttribute<int64_t>(attn_node, "kv_num_heads", kNumHeads);
  AddAttribute<int64_t>(attn_node, "is_causal", static_cast<int64_t>(1));

  // ---- Output projection + residual ------------------------------------
  AddNode(*graph, "MatMul", {"attn_out", "o_proj.weight"}, {"attn_proj"});
  AddNode(*graph, "Add", {"hidden", "attn_proj"}, {"hidden2"});

  // ---- Post-attention RMSNorm + SwiGLU MLP -----------------------------
  AddNode(*graph, "RMSNormalization", {"hidden2", "post_attention_layernorm.weight"}, {"normed2"});
  AddNode(*graph, "MatMul", {"normed2", "gate_proj.weight"}, {"gate"});
  AddNode(*graph, "Sigmoid", {"gate"}, {"gate_sigmoid"});
  AddNode(*graph, "Mul", {"gate", "gate_sigmoid"}, {"gate_silu"});
  AddNode(*graph, "MatMul", {"normed2", "up_proj.weight"}, {"up"});
  AddNode(*graph, "Mul", {"gate_silu", "up"}, {"mlp_hidden"});
  AddNode(*graph, "MatMul", {"mlp_hidden", "down_proj.weight"}, {"mlp_out"});
  AddNode(*graph, "Add", {"hidden2", "mlp_out"}, {"hidden3"});

  // ---- Final RMSNorm + LM head -----------------------------------------
  AddNode(*graph, "RMSNormalization", {"hidden3", "norm.weight"}, {"normed_final"});
  AddNode(*graph, "MatMul", {"normed_final", "lm_head.weight"}, {"logits"});

  // ---- Random weight initializers --------------------------------------
  AddInitializer<float>(*graph, "embed_tokens.weight", {kVocabSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kVocabSize * kHiddenSize), 1u));
  AddInitializer<float>(*graph, "input_layernorm.weight", {kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize), 2u));
  AddInitializer<float>(*graph, "q_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 3u));
  AddInitializer<float>(*graph, "k_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 4u));
  AddInitializer<float>(*graph, "v_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 5u));
  AddInitializer<float>(*graph, "o_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 6u));
  AddInitializer<float>(*graph, "post_attention_layernorm.weight", {kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize), 7u));
  AddInitializer<float>(*graph, "gate_proj.weight", {kHiddenSize, kIntermediateSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kIntermediateSize), 8u));
  AddInitializer<float>(*graph, "up_proj.weight", {kHiddenSize, kIntermediateSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kIntermediateSize), 9u));
  AddInitializer<float>(*graph, "down_proj.weight", {kIntermediateSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kIntermediateSize * kHiddenSize), 10u));
  AddInitializer<float>(*graph, "norm.weight", {kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize), 11u));
  AddInitializer<float>(*graph, "lm_head.weight", {kHiddenSize, kVocabSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kVocabSize), 12u));

  // Small constant initializers backing the additive attention mask.
  AddInitializer<int64_t>(*graph, "mask_axes", {2}, {1, 2});
  AddInitializer<float>(*graph, "mask_one", {1}, {1.0f});
  AddInitializer<float>(*graph, "mask_neg", {1}, {-1e9f});

  // ---- Graph inputs (fully dynamic shapes) -----------------------------
  AppendValueInfo(*graph->add_input(), "input_ids", kInt64, {"batch", "seq"});
  AppendValueInfo(*graph->add_input(), "attention_mask", kInt64, {"batch", "total_seq"});
  AppendValueInfo(*graph->add_input(), "past_key", kFloat,
                  {"batch", DimSpec(kNumHeads), "past_seq", DimSpec(kHeadSize)});
  AppendValueInfo(*graph->add_input(), "past_value", kFloat,
                  {"batch", DimSpec(kNumHeads), "past_seq", DimSpec(kHeadSize)});

  // ---- Intermediate value_info shapes ----------------------------------
  // Declared in alphabetical order, mirroring the convention of the sibling
  // shape-inference cases. Only the dims shape inference can resolve from the
  // initializers are concrete; the rest stay symbolic.
  const std::vector<DimSpec> bsh = {"batch", "seq", DimSpec(kHiddenSize)};
  const std::vector<DimSpec> bsi = {"batch", "seq", DimSpec(kIntermediateSize)};
  AppendValueInfo(*graph->add_value_info(), "attn_bias", kFloat,
                  {"batch", DimSpec(int64_t{1}), DimSpec(int64_t{1}), "total_seq"});
  AppendValueInfo(*graph->add_value_info(), "attn_out", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "attn_proj", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "gate", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "gate_silu", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "gate_sigmoid", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "hidden", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "hidden2", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "hidden3", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "key", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "mask_4d", kFloat,
                  {"batch", DimSpec(int64_t{1}), DimSpec(int64_t{1}), "total_seq"});
  AppendValueInfo(*graph->add_value_info(), "mask_float", kFloat, {"batch", "total_seq"});
  AppendValueInfo(*graph->add_value_info(), "mask_inv", kFloat,
                  {"batch", DimSpec(int64_t{1}), DimSpec(int64_t{1}), "total_seq"});
  AppendValueInfo(*graph->add_value_info(), "mlp_hidden", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "mlp_out", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "normed1", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "normed2", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "normed_final", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "query", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "up", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "value", kFloat, bsh);

  // ---- Graph outputs ---------------------------------------------------
  AppendValueInfo(*graph->add_output(), "logits", kFloat, {"batch", "seq", DimSpec(kVocabSize)});
  AppendValueInfo(*graph->add_output(), "present_key", kFloat,
                  {"batch", DimSpec(kNumHeads), "total_seq", DimSpec(kHeadSize)});
  AppendValueInfo(*graph->add_output(), "present_value", kFloat,
                  {"batch", DimSpec(kNumHeads), "total_seq", DimSpec(kHeadSize)});

  registry.emplace_back(std::move(tc));
}

// ---------------------------------------------------------------------------
// ``arnir0/Tiny-LLM`` (inlined) — the same single decoder layer as
// :cpp:func:`RegisterTinyLlmShapeInferenceCases` but with the fused
// ``RMSNormalization`` and ``Attention`` operators **inlined** into their
// primitive subgraphs. This exercises shape inference through the longer chains
// a real exporter emits when those operators are decomposed:
//
//   RMSNorm(x, w)      = Mul(Div(x, Sqrt(Add(ReduceMean(Mul(x, x)), eps))), w)
//   Attention(q,k,v)   = Reshape/Transpose the heads, Concat the KV cache,
//                        scaled MatMul(q, kᵀ) + mask, Softmax, MatMul(·, v),
//                        Transpose/Reshape back to ``[batch, seq, hidden]``.
//
// The graph keeps the exact same four inputs and three outputs (with fully
// dynamic ``batch`` / ``seq`` / ``past_seq`` / ``total_seq`` dims) as the fused
// companion above.
// ---------------------------------------------------------------------------
void RegisterTinyLlmInlinedShapeInferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(23);

  const std::string name = "test_cc_shape_inference_tiny_llm_inlined";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  const TensorProto::DataType kFloat = DataType::FLOAT;
  const TensorProto::DataType kInt64 = DataType::INT64;

  // Inlines ``RMSNormalization(x, weight)`` as
  // ``Mul(Div(x, Sqrt(Add(ReduceMean(Mul(x, x)), eps))), weight)``. ``prefix``
  // disambiguates the intermediate value names across the three call sites.
  const auto add_rmsnorm = [&](const std::string &x, const std::string &weight,
                               const std::string &out, const std::string &prefix) {
    AddNode(*graph, "Mul", {x, x}, {prefix + "_sq"});
    AddNode(*graph, "ReduceMean", {prefix + "_sq", "rms_axes"}, {prefix + "_mean"});
    AddNode(*graph, "Add", {prefix + "_mean", "rms_eps"}, {prefix + "_meaneps"});
    AddNode(*graph, "Sqrt", {prefix + "_meaneps"}, {prefix + "_rms"});
    AddNode(*graph, "Div", {x, prefix + "_rms"}, {prefix + "_norm"});
    AddNode(*graph, "Mul", {prefix + "_norm", weight}, {out});
  };

  // ---- Token embedding -------------------------------------------------
  AddNode(*graph, "Gather", {"embed_tokens.weight", "input_ids"}, {"hidden"});

  // ---- Input RMSNorm (inlined) + QKV projections -----------------------
  add_rmsnorm("hidden", "input_layernorm.weight", "normed1", "ln1");
  AddNode(*graph, "MatMul", {"normed1", "q_proj.weight"}, {"query"});
  AddNode(*graph, "MatMul", {"normed1", "k_proj.weight"}, {"key"});
  AddNode(*graph, "MatMul", {"normed1", "v_proj.weight"}, {"value"});

  // ---- Additive attention mask derived from attention_mask -------------
  // attn_bias = (1 - cast(attention_mask)) * -1e9, broadcast to a 4-D mask.
  NodeProto &cast_node = AddNode(*graph, "Cast", {"attention_mask"}, {"mask_float"});
  AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(DataType::FLOAT));
  AddNode(*graph, "Unsqueeze", {"mask_float", "mask_axes"}, {"mask_4d"});
  AddNode(*graph, "Sub", {"mask_one", "mask_4d"}, {"mask_inv"});
  AddNode(*graph, "Mul", {"mask_inv", "mask_neg"}, {"attn_bias"});

  // ---- Self-attention with KV cache (inlined) --------------------------
  // Split the fused [batch, seq, hidden] projections into heads, append the
  // KV cache, run scaled dot-product attention and merge the heads back.
  AddNode(*graph, "Reshape", {"query", "head_shape"}, {"query_4d"});
  NodeProto &q_transpose = AddNode(*graph, "Transpose", {"query_4d"}, {"query_heads"});
  AddAttribute<std::vector<int64_t>>(q_transpose, "perm", {0, 2, 1, 3});
  AddNode(*graph, "Reshape", {"key", "head_shape"}, {"key_4d"});
  NodeProto &k_transpose = AddNode(*graph, "Transpose", {"key_4d"}, {"key_heads"});
  AddAttribute<std::vector<int64_t>>(k_transpose, "perm", {0, 2, 1, 3});
  AddNode(*graph, "Reshape", {"value", "head_shape"}, {"value_4d"});
  NodeProto &v_transpose = AddNode(*graph, "Transpose", {"value_4d"}, {"value_heads"});
  AddAttribute<std::vector<int64_t>>(v_transpose, "perm", {0, 2, 1, 3});

  NodeProto &key_concat = AddNode(*graph, "Concat", {"past_key", "key_heads"}, {"present_key"});
  AddAttribute<int64_t>(key_concat, "axis", 2);
  NodeProto &value_concat =
      AddNode(*graph, "Concat", {"past_value", "value_heads"}, {"present_value"});
  AddAttribute<int64_t>(value_concat, "axis", 2);

  NodeProto &key_transpose = AddNode(*graph, "Transpose", {"present_key"}, {"key_heads_t"});
  AddAttribute<std::vector<int64_t>>(key_transpose, "perm", {0, 1, 3, 2});
  AddNode(*graph, "MatMul", {"query_heads", "key_heads_t"}, {"scores"});
  AddNode(*graph, "Mul", {"scores", "attn_scale"}, {"scores_scaled"});
  AddNode(*graph, "Add", {"scores_scaled", "attn_bias"}, {"scores_biased"});
  NodeProto &softmax_node = AddNode(*graph, "Softmax", {"scores_biased"}, {"attn_weights"});
  AddAttribute<int64_t>(softmax_node, "axis", -1);
  AddNode(*graph, "MatMul", {"attn_weights", "present_value"}, {"context"});
  NodeProto &context_transpose = AddNode(*graph, "Transpose", {"context"}, {"context_t"});
  AddAttribute<std::vector<int64_t>>(context_transpose, "perm", {0, 2, 1, 3});
  AddNode(*graph, "Reshape", {"context_t", "merge_shape"}, {"attn_out"});

  // ---- Output projection + residual ------------------------------------
  AddNode(*graph, "MatMul", {"attn_out", "o_proj.weight"}, {"attn_proj"});
  AddNode(*graph, "Add", {"hidden", "attn_proj"}, {"hidden2"});

  // ---- Post-attention RMSNorm (inlined) + SwiGLU MLP -------------------
  add_rmsnorm("hidden2", "post_attention_layernorm.weight", "normed2", "ln2");
  AddNode(*graph, "MatMul", {"normed2", "gate_proj.weight"}, {"gate"});
  AddNode(*graph, "Sigmoid", {"gate"}, {"gate_sigmoid"});
  AddNode(*graph, "Mul", {"gate", "gate_sigmoid"}, {"gate_silu"});
  AddNode(*graph, "MatMul", {"normed2", "up_proj.weight"}, {"up"});
  AddNode(*graph, "Mul", {"gate_silu", "up"}, {"mlp_hidden"});
  AddNode(*graph, "MatMul", {"mlp_hidden", "down_proj.weight"}, {"mlp_out"});
  AddNode(*graph, "Add", {"hidden2", "mlp_out"}, {"hidden3"});

  // ---- Final RMSNorm (inlined) + LM head -------------------------------
  add_rmsnorm("hidden3", "norm.weight", "normed_final", "lnf");
  AddNode(*graph, "MatMul", {"normed_final", "lm_head.weight"}, {"logits"});

  // ---- Random weight initializers --------------------------------------
  AddInitializer<float>(*graph, "embed_tokens.weight", {kVocabSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kVocabSize * kHiddenSize), 1u));
  AddInitializer<float>(*graph, "input_layernorm.weight", {kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize), 2u));
  AddInitializer<float>(*graph, "q_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 3u));
  AddInitializer<float>(*graph, "k_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 4u));
  AddInitializer<float>(*graph, "v_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 5u));
  AddInitializer<float>(*graph, "o_proj.weight", {kHiddenSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kHiddenSize), 6u));
  AddInitializer<float>(*graph, "post_attention_layernorm.weight", {kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize), 7u));
  AddInitializer<float>(*graph, "gate_proj.weight", {kHiddenSize, kIntermediateSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kIntermediateSize), 8u));
  AddInitializer<float>(*graph, "up_proj.weight", {kHiddenSize, kIntermediateSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kIntermediateSize), 9u));
  AddInitializer<float>(*graph, "down_proj.weight", {kIntermediateSize, kHiddenSize},
                        RandomWeights(static_cast<size_t>(kIntermediateSize * kHiddenSize), 10u));
  AddInitializer<float>(*graph, "norm.weight", {kHiddenSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize), 11u));
  AddInitializer<float>(*graph, "lm_head.weight", {kHiddenSize, kVocabSize},
                        RandomWeights(static_cast<size_t>(kHiddenSize * kVocabSize), 12u));

  // Small constant initializers backing the inlined subgraphs.
  AddInitializer<int64_t>(*graph, "rms_axes", {1}, {-1});
  AddInitializer<float>(*graph, "rms_eps", {1}, {1e-5f});
  AddInitializer<int64_t>(*graph, "head_shape", {4}, {0, 0, kNumHeads, kHeadSize});
  AddInitializer<int64_t>(*graph, "merge_shape", {3}, {0, 0, kHiddenSize});
  AddInitializer<float>(*graph, "attn_scale", {1},
                        {1.0f / static_cast<float>(std::sqrt(static_cast<double>(kHeadSize)))});

  // Constant initializers backing the additive attention mask.
  AddInitializer<int64_t>(*graph, "mask_axes", {2}, {1, 2});
  AddInitializer<float>(*graph, "mask_one", {1}, {1.0f});
  AddInitializer<float>(*graph, "mask_neg", {1}, {-1e9f});

  // ---- Graph inputs (fully dynamic shapes) -----------------------------
  AppendValueInfo(*graph->add_input(), "input_ids", kInt64, {"batch", "seq"});
  AppendValueInfo(*graph->add_input(), "attention_mask", kInt64, {"batch", "total_seq"});
  AppendValueInfo(*graph->add_input(), "past_key", kFloat,
                  {"batch", DimSpec(kNumHeads), "past_seq", DimSpec(kHeadSize)});
  AppendValueInfo(*graph->add_input(), "past_value", kFloat,
                  {"batch", DimSpec(kNumHeads), "past_seq", DimSpec(kHeadSize)});

  // ---- Intermediate value_info shapes ----------------------------------
  // Declared in alphabetical order, mirroring the convention of the sibling
  // shape-inference cases. Only the dims shape inference can resolve from the
  // initializers are concrete; the rest stay symbolic.
  const std::vector<DimSpec> bsh = {"batch", "seq", DimSpec(kHiddenSize)};
  const std::vector<DimSpec> bsi = {"batch", "seq", DimSpec(kIntermediateSize)};
  const std::vector<DimSpec> bs1 = {"batch", "seq", DimSpec(int64_t{1})};
  const std::vector<DimSpec> mask4 = {"batch", DimSpec(int64_t{1}), DimSpec(int64_t{1}),
                                      "total_seq"};
  const std::vector<DimSpec> b_s_nh_hs = {"batch", "seq", DimSpec(kNumHeads), DimSpec(kHeadSize)};
  const std::vector<DimSpec> b_nh_s_hs = {"batch", DimSpec(kNumHeads), "seq", DimSpec(kHeadSize)};
  const std::vector<DimSpec> b_nh_hs_total = {"batch", DimSpec(kNumHeads), DimSpec(kHeadSize),
                                              "total_seq"};
  const std::vector<DimSpec> b_nh_s_total = {"batch", DimSpec(kNumHeads), "seq", "total_seq"};

  AppendValueInfo(*graph->add_value_info(), "attn_bias", kFloat, mask4);
  AppendValueInfo(*graph->add_value_info(), "attn_out", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "attn_proj", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "attn_weights", kFloat, b_nh_s_total);
  AppendValueInfo(*graph->add_value_info(), "context", kFloat, b_nh_s_hs);
  AppendValueInfo(*graph->add_value_info(), "context_t", kFloat, b_s_nh_hs);
  AppendValueInfo(*graph->add_value_info(), "gate", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "gate_silu", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "gate_sigmoid", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "hidden", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "hidden2", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "hidden3", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "key", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "key_4d", kFloat, b_s_nh_hs);
  AppendValueInfo(*graph->add_value_info(), "key_heads", kFloat, b_nh_s_hs);
  AppendValueInfo(*graph->add_value_info(), "key_heads_t", kFloat, b_nh_hs_total);
  // All five intermediates of the inlined input RMSNorm. The post-attention
  // (``ln2_*``) and final (``lnf_*``) RMSNorms produce the same shapes; the
  // ``_mean`` / ``_meaneps`` / ``_rms`` reductions collapse the last axis to 1.
  AppendValueInfo(*graph->add_value_info(), "ln1_mean", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "ln1_meaneps", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "ln1_norm", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "ln1_rms", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "ln1_sq", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "ln2_mean", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "ln2_meaneps", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "ln2_norm", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "ln2_rms", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "ln2_sq", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "lnf_mean", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "lnf_meaneps", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "lnf_norm", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "lnf_rms", kFloat, bs1);
  AppendValueInfo(*graph->add_value_info(), "lnf_sq", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "mask_4d", kFloat, mask4);
  AppendValueInfo(*graph->add_value_info(), "mask_float", kFloat, {"batch", "total_seq"});
  AppendValueInfo(*graph->add_value_info(), "mask_inv", kFloat, mask4);
  AppendValueInfo(*graph->add_value_info(), "mlp_hidden", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "mlp_out", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "normed1", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "normed2", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "normed_final", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "query", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "query_4d", kFloat, b_s_nh_hs);
  AppendValueInfo(*graph->add_value_info(), "query_heads", kFloat, b_nh_s_hs);
  AppendValueInfo(*graph->add_value_info(), "scores", kFloat, b_nh_s_total);
  AppendValueInfo(*graph->add_value_info(), "scores_biased", kFloat, b_nh_s_total);
  AppendValueInfo(*graph->add_value_info(), "scores_scaled", kFloat, b_nh_s_total);
  AppendValueInfo(*graph->add_value_info(), "up", kFloat, bsi);
  AppendValueInfo(*graph->add_value_info(), "value", kFloat, bsh);
  AppendValueInfo(*graph->add_value_info(), "value_4d", kFloat, b_s_nh_hs);
  AppendValueInfo(*graph->add_value_info(), "value_heads", kFloat, b_nh_s_hs);

  // ---- Graph outputs ---------------------------------------------------
  AppendValueInfo(*graph->add_output(), "logits", kFloat, {"batch", "seq", DimSpec(kVocabSize)});
  AppendValueInfo(*graph->add_output(), "present_key", kFloat,
                  {"batch", DimSpec(kNumHeads), "total_seq", DimSpec(kHeadSize)});
  AppendValueInfo(*graph->add_output(), "present_value", kFloat,
                  {"batch", DimSpec(kNumHeads), "total_seq", DimSpec(kHeadSize)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
