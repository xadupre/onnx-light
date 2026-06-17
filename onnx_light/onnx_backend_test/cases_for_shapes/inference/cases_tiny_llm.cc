// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

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

// Builds a deterministic pseudo-random FLOAT weight vector of ``count``
// elements in ``[-0.05, 0.05]`` (a typical initialization range for small
// language-model weights). A Numerical-Recipes LCG seeded by ``seed`` keeps
// the data reproducible across runs and platforms without depending on a
// global RNG.
std::vector<float> RandomWeights(size_t count, uint32_t seed) {
  std::vector<float> values(count);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s & 0x00ffffffu) / static_cast<float>(0x01000000u);
    values[i] = -0.05f + 0.1f * u;
  }
  return values;
}

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
void RegisterTinyLlmShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(23);

  const std::string name = "test_cc_shape_inference_tiny_llm";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  ModelProto &model = tc.model;
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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
