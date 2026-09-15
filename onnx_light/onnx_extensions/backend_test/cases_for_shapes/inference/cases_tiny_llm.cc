// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/inference_random_weights.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
void RegisterTinyLlmShapeInferenceCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(23);

  const std::string name = "test_cc_shape_inference_tiny_llm";

  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
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

  // ---- Pre-embedded metadata (shape-tag, inplace-reuse, release-after) ----
  // These are the expected outputs of WriteValueAndNodeTagsToMetadata and
  // WriteInPlaceReuseToMetadata for this model. They allow
  // TestBackendMetadataCoverage to verify that the computed metadata matches.
  //
  // Value-tag seeds: all graph inputs and initializers → "weight"; mask_axes
  // upgraded to "axes" by Unsqueeze's input[1] rule.
  // mask_float and mask_4d are tagged "weight" via Sub backward propagation:
  // Sub(mask_one, mask_4d)→mask_inv tags mask_4d "weight", then Unsqueeze
  // backward tags mask_float "weight".  Cast backward (Cast propagates from its
  // first input) then tags attention_mask "weight".
  {
    namespace ann = core::compute;

    // Helper: add a metadata entry to the i-th node.
    const auto node_meta = [&graph](int i, const char *key, const std::string &value) {
      (*graph->mutable_node())[static_cast<std::size_t>(i)].add_metadata(key, value);
    };

    // node[0]  Gather(embed_tokens.weight, input_ids) → hidden
    node_meta(0, ann::kNodeTagMetadataKey, "weight");
    // node[1]  RMSNormalization(hidden, input_layernorm.weight) → normed1
    node_meta(1, ann::kNodeTagMetadataKey, "weight");
    // node[2]  MatMul(normed1, q_proj.weight) → query
    node_meta(2, ann::kNodeTagMetadataKey, "weight");
    // node[3]  MatMul(normed1, k_proj.weight) → key
    node_meta(3, ann::kNodeTagMetadataKey, "weight");
    // node[4]  MatMul(normed1, v_proj.weight) → value
    //   normed1 last used here → released; value reuses normed1's buffer.
    node_meta(4, ann::kNodeTagMetadataKey, "weight");
    node_meta(4, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(4, ann::kReleaseAfterMetadataKey, "normed1");
    // node[5]  Cast(attention_mask) → mask_float
    //   attention_mask is tagged "weight" via Cast backward propagation
    //   (mask_float→"weight"→Cast backward→attention_mask="weight").
    node_meta(5, ann::kNodeTagMetadataKey, "weight");
    // node[6]  Unsqueeze(mask_float, mask_axes) → mask_4d
    //   mask_float inherits "weight" via Sub backward propagation; mask_float
    //   last used here → released; same total byte size → inplace.
    node_meta(6, ann::kNodeTagMetadataKey, "weight");
    node_meta(6, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(6, ann::kReleaseAfterMetadataKey, "mask_float");
    // node[7]  Sub(mask_one, mask_4d) → mask_inv
    //   mask_4d last used here (output 0 = mask_inv reuses mask_4d buffer).
    node_meta(7, ann::kNodeTagMetadataKey, "weight");
    node_meta(7, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(7, ann::kReleaseAfterMetadataKey, "mask_4d");
    // node[8]  Mul(mask_inv, mask_neg) → attn_bias
    node_meta(8, ann::kNodeTagMetadataKey, "weight");
    node_meta(8, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(8, ann::kReleaseAfterMetadataKey, "mask_inv");
    // node[9]  Attention(query, key, value, attn_bias, past_key, past_value)
    //           → {attn_out, present_key, present_value}
    //   query, key, value, attn_bias all last used here → released.
    //   attn_out (output 0) reuses query's buffer (same [batch,seq,hidden]).
    node_meta(9, ann::kNodeTagMetadataKey, "weight");
    node_meta(9, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(9, ann::kReleaseAfterMetadataKey, "query;key;value;attn_bias");
    // node[10] MatMul(attn_out, o_proj.weight) → attn_proj
    node_meta(10, ann::kNodeTagMetadataKey, "weight");
    node_meta(10, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(10, ann::kReleaseAfterMetadataKey, "attn_out");
    // node[11] Add(hidden, attn_proj) → hidden2
    node_meta(11, ann::kNodeTagMetadataKey, "weight");
    node_meta(11, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(11, ann::kReleaseAfterMetadataKey, "hidden;attn_proj");
    // node[12] RMSNormalization(hidden2, post_attention_layernorm.weight) → normed2
    //   hidden2 last_use=19 (used again at Add below) → not released here.
    node_meta(12, ann::kNodeTagMetadataKey, "weight");
    // node[13] MatMul(normed2, gate_proj.weight) → gate
    //   normed2 last_use=16 → not released here.
    node_meta(13, ann::kNodeTagMetadataKey, "weight");
    // node[14] Sigmoid(gate) → gate_sigmoid
    //   gate last_use=15 → not released here.
    node_meta(14, ann::kNodeTagMetadataKey, "weight");
    // node[15] Mul(gate, gate_sigmoid) → gate_silu
    node_meta(15, ann::kNodeTagMetadataKey, "weight");
    node_meta(15, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(15, ann::kReleaseAfterMetadataKey, "gate;gate_sigmoid");
    // node[16] MatMul(normed2, up_proj.weight) → up
    //   normed2 last used here → released; up=[batch,seq,32] ≠ normed2=[batch,seq,16] → no inplace.
    node_meta(16, ann::kNodeTagMetadataKey, "weight");
    node_meta(16, ann::kReleaseAfterMetadataKey, "normed2");
    // node[17] Mul(gate_silu, up) → mlp_hidden
    node_meta(17, ann::kNodeTagMetadataKey, "weight");
    node_meta(17, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(17, ann::kReleaseAfterMetadataKey, "gate_silu;up");
    // node[18] MatMul(mlp_hidden, down_proj.weight) → mlp_out
    //   mlp_hidden=[batch,seq,32] ≠ mlp_out=[batch,seq,16] → no inplace.
    node_meta(18, ann::kNodeTagMetadataKey, "weight");
    node_meta(18, ann::kReleaseAfterMetadataKey, "mlp_hidden");
    // node[19] Add(hidden2, mlp_out) → hidden3
    node_meta(19, ann::kNodeTagMetadataKey, "weight");
    node_meta(19, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(19, ann::kReleaseAfterMetadataKey, "hidden2;mlp_out");
    // node[20] RMSNormalization(hidden3, norm.weight) → normed_final
    node_meta(20, ann::kNodeTagMetadataKey, "weight");
    node_meta(20, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(20, ann::kReleaseAfterMetadataKey, "hidden3");
    // node[21] MatMul(normed_final, lm_head.weight) → logits
    //   normed_final=[batch,seq,16] ≠ logits=[batch,seq,32] → no inplace.
    node_meta(21, ann::kNodeTagMetadataKey, "weight");
    node_meta(21, ann::kReleaseAfterMetadataKey, "normed_final");

    // Per-value kValueTagMetadataKey on value_info (alphabetical order, matching
    // the AppendValueInfo calls above).
    // value_info[0]  attn_bias
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[1]  attn_out
    {
      StringStringEntryProto *entry = graph->mutable_value_info(1)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[2]  attn_proj
    {
      StringStringEntryProto *entry = graph->mutable_value_info(2)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[3]  gate
    {
      StringStringEntryProto *entry = graph->mutable_value_info(3)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[4]  gate_silu
    {
      StringStringEntryProto *entry = graph->mutable_value_info(4)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[5]  gate_sigmoid
    {
      StringStringEntryProto *entry = graph->mutable_value_info(5)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[6]  hidden
    {
      StringStringEntryProto *entry = graph->mutable_value_info(6)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[7]  hidden2
    {
      StringStringEntryProto *entry = graph->mutable_value_info(7)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[8]  hidden3
    {
      StringStringEntryProto *entry = graph->mutable_value_info(8)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[9]  key
    {
      StringStringEntryProto *entry = graph->mutable_value_info(9)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[10] mask_4d → "weight" (backward-tagged via Sub)
    {
      StringStringEntryProto *entry = graph->mutable_value_info(10)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[11] mask_float → "weight" (backward-tagged via Unsqueeze → Sub)
    {
      StringStringEntryProto *entry = graph->mutable_value_info(11)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[12] mask_inv
    {
      StringStringEntryProto *entry = graph->mutable_value_info(12)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[13] mlp_hidden
    {
      StringStringEntryProto *entry = graph->mutable_value_info(13)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[14] mlp_out
    {
      StringStringEntryProto *entry = graph->mutable_value_info(14)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[15] normed1
    {
      StringStringEntryProto *entry = graph->mutable_value_info(15)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[16] normed2
    {
      StringStringEntryProto *entry = graph->mutable_value_info(16)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[17] normed_final
    {
      StringStringEntryProto *entry = graph->mutable_value_info(17)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[18] query
    {
      StringStringEntryProto *entry = graph->mutable_value_info(18)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[19] up
    {
      StringStringEntryProto *entry = graph->mutable_value_info(19)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[20] value
    {
      StringStringEntryProto *entry = graph->mutable_value_info(20)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Per-value kValueTagMetadataKey on graph outputs.
    // output[0] logits → "weight"
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // output[1] present_key → "weight"
    {
      StringStringEntryProto *entry = graph->mutable_output(1)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // output[2] present_value → "weight"
    {
      StringStringEntryProto *entry = graph->mutable_output(2)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Per-value kValueTagMetadataKey on graph inputs.
    // input[0] input_ids → "weight" (direct graph input seed)
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // input[1] attention_mask → "weight" (Cast backward propagation)
    {
      StringStringEntryProto *entry = graph->mutable_input(1)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // input[2] past_key → "weight" (direct graph input seed)
    {
      StringStringEntryProto *entry = graph->mutable_input(2)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // input[3] past_value → "weight" (direct graph input seed)
    {
      StringStringEntryProto *entry = graph->mutable_input(3)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Per-value kValueTagMetadataKey on initializers (insertion order, see
    // AddInitializer calls above).
    const auto init_meta = [&graph](std::size_t i, const char *tag) {
      auto *entry = graph->mutable_initializer(i)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value(tag);
    };
    // initializer[0]  embed_tokens.weight → "weight"
    init_meta(0, "weight");
    // initializer[1]  input_layernorm.weight → "weight"
    init_meta(1, "weight");
    // initializer[2]  q_proj.weight → "weight"
    init_meta(2, "weight");
    // initializer[3]  k_proj.weight → "weight"
    init_meta(3, "weight");
    // initializer[4]  v_proj.weight → "weight"
    init_meta(4, "weight");
    // initializer[5]  o_proj.weight → "weight"
    init_meta(5, "weight");
    // initializer[6]  post_attention_layernorm.weight → "weight"
    init_meta(6, "weight");
    // initializer[7]  gate_proj.weight → "weight"
    init_meta(7, "weight");
    // initializer[8]  up_proj.weight → "weight"
    init_meta(8, "weight");
    // initializer[9]  down_proj.weight → "weight"
    init_meta(9, "weight");
    // initializer[10] norm.weight → "weight"
    init_meta(10, "weight");
    // initializer[11] lm_head.weight → "weight"
    init_meta(11, "weight");
    // initializer[12] mask_axes → "axes"
    init_meta(12, "axes");
    // initializer[13] mask_one → "weight"
    init_meta(13, "weight");
    // initializer[14] mask_neg → "weight"
    init_meta(14, "weight");

    // Constant information: every initializer is a build-time constant. No
    // intermediate value or node is constant because they all depend on the
    // runtime graph inputs (input_ids, attention_mask, past_key, past_value).
    const auto init_const = [&graph](std::size_t i) {
      auto *entry = graph->mutable_initializer(i)->add_metadata_props();
      entry->set_key(ann::kConstantMetadataKey);
      entry->set_value("1");
    };
    for (std::size_t i = 0; i < 15; ++i) {
      init_const(i);
    }
  }

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
void RegisterTinyLlmInlinedShapeInferenceCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(23);

  const std::string name = "test_cc_shape_inference_tiny_llm_inlined";

  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
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
  const auto add_rmsnorm = [&graph](const std::string &x, const std::string &weight,
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

  // ---- Pre-embedded metadata (shape-tag, inplace-reuse, release-after) ----
  // present_key/present_value are now tagged "weight" because Concat gains
  // "weight" tag when at least one input is "weight" (key_heads/value_heads).
  // The 52 nodes are indexed 0-51 in AddNode insertion order.
  {
    namespace ann = core::compute;

    // Notes:
    //   attn_scale, rms_eps, rms_axes, head_shape, merge_shape, mask_* are
    //   initializers that get seed "weight" (or upgraded to "axes"/"shape").
    //   mask_float and mask_4d are now tagged "weight" via Sub backward
    //   propagation (Sub(mask_one, mask_4d)→mask_inv with mask_one="weight").
    //   attention_mask gets "weight" via Cast backward propagation.
    //   past_key/past_value get "weight" via Concat backward (key_heads/
    //   value_heads are "weight" → Concat output "weight" → backward).
    //   key_heads_t gets "weight" via Transpose of present_key (which is
    //   "weight").

    // Helper: add a metadata entry to the i-th node.
    const auto node_meta = [&graph](int i, const char *key, const std::string &value) {
      (*graph->mutable_node())[static_cast<std::size_t>(i)].add_metadata(key, value);
    };

    // Node index / operation / inputs → outputs
    // node[0]  Gather(embed_tokens.weight, input_ids) → hidden
    node_meta(0, ann::kNodeTagMetadataKey, "weight");
    // ---- First RMSNorm (inlined): nodes 1-6 --------------------------------
    // node[1]  Mul(hidden, hidden) → ln1_sq
    node_meta(1, ann::kNodeTagMetadataKey, "weight");
    // node[2]  ReduceMean(ln1_sq, rms_axes) → ln1_mean
    //   ln1_sq last used here → released; shapes differ → no inplace.
    node_meta(2, ann::kNodeTagMetadataKey, "weight");
    node_meta(2, ann::kReleaseAfterMetadataKey, "ln1_sq");
    // node[3]  Add(ln1_mean, rms_eps) → ln1_meaneps
    //   ln1_mean last used here → released; same [batch,seq,1] → inplace.
    node_meta(3, ann::kNodeTagMetadataKey, "weight");
    node_meta(3, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(3, ann::kReleaseAfterMetadataKey, "ln1_mean");
    // node[4]  Sqrt(ln1_meaneps) → ln1_rms
    node_meta(4, ann::kNodeTagMetadataKey, "weight");
    node_meta(4, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(4, ann::kReleaseAfterMetadataKey, "ln1_meaneps");
    // node[5]  Div(hidden, ln1_rms) → ln1_norm
    //   hidden last_use=31 (residual Add below) → no inplace on hidden.
    //   ln1_rms last used here → released; shapes differ → no inplace.
    node_meta(5, ann::kNodeTagMetadataKey, "weight");
    node_meta(5, ann::kReleaseAfterMetadataKey, "ln1_rms");
    // node[6]  Mul(ln1_norm, input_layernorm.weight) → normed1
    node_meta(6, ann::kNodeTagMetadataKey, "weight");
    node_meta(6, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(6, ann::kReleaseAfterMetadataKey, "ln1_norm");
    // ---- QKV projections ---------------------------------------------------
    // node[7]  MatMul(normed1, q_proj.weight) → query
    //   normed1 last_use=9 → not released here.
    node_meta(7, ann::kNodeTagMetadataKey, "weight");
    // node[8]  MatMul(normed1, k_proj.weight) → key
    node_meta(8, ann::kNodeTagMetadataKey, "weight");
    // node[9]  MatMul(normed1, v_proj.weight) → value
    //   normed1 last used here → released; same [batch,seq,hidden] → inplace.
    node_meta(9, ann::kNodeTagMetadataKey, "weight");
    node_meta(9, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(9, ann::kReleaseAfterMetadataKey, "normed1");
    // ---- Attention mask ----------------------------------------------------
    // node[10] Cast(attention_mask) → mask_float
    //   attention_mask is tagged "weight" via Cast backward propagation.
    node_meta(10, ann::kNodeTagMetadataKey, "weight");
    // node[11] Unsqueeze(mask_float, mask_axes) → mask_4d
    //   mask_float inherits "weight" via Sub backward propagation; mask_float
    //   released here; same total byte size → inplace.
    node_meta(11, ann::kNodeTagMetadataKey, "weight");
    node_meta(11, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(11, ann::kReleaseAfterMetadataKey, "mask_float");
    // node[12] Sub(mask_one, mask_4d) → mask_inv
    //   mask_4d last used here; mask_inv reuses mask_4d (input index=1).
    node_meta(12, ann::kNodeTagMetadataKey, "weight");
    node_meta(12, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(12, ann::kReleaseAfterMetadataKey, "mask_4d");
    // node[13] Mul(mask_inv, mask_neg) → attn_bias
    node_meta(13, ann::kNodeTagMetadataKey, "weight");
    node_meta(13, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(13, ann::kReleaseAfterMetadataKey, "mask_inv");
    // ---- Split projections into heads ----------------------------------------
    // node[14] Reshape(query, head_shape) → query_4d
    //   [b,s,h]→[b,s,nh,hs]: same total byte size → kEqual.
    node_meta(14, ann::kNodeTagMetadataKey, "weight");
    node_meta(14, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(14, ann::kReleaseAfterMetadataKey, "query");
    // node[15] Transpose(query_4d) → query_heads
    //   [b,s,nh,hs]→[b,nh,s,hs]: same total byte size → kEqual.
    node_meta(15, ann::kNodeTagMetadataKey, "weight");
    node_meta(15, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(15, ann::kReleaseAfterMetadataKey, "query_4d");
    // node[16] Reshape(key, head_shape) → key_4d
    node_meta(16, ann::kNodeTagMetadataKey, "weight");
    node_meta(16, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(16, ann::kReleaseAfterMetadataKey, "key");
    // node[17] Transpose(key_4d) → key_heads
    node_meta(17, ann::kNodeTagMetadataKey, "weight");
    node_meta(17, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(17, ann::kReleaseAfterMetadataKey, "key_4d");
    // node[18] Reshape(value, head_shape) → value_4d
    node_meta(18, ann::kNodeTagMetadataKey, "weight");
    node_meta(18, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(18, ann::kReleaseAfterMetadataKey, "value");
    // node[19] Transpose(value_4d) → value_heads
    node_meta(19, ann::kNodeTagMetadataKey, "weight");
    node_meta(19, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(19, ann::kReleaseAfterMetadataKey, "value_4d");
    // ---- Concat KV cache -------------------------------------------------
    // node[20] Concat(past_key, key_heads) → present_key
    //   key_heads has "weight" → Concat output = "weight"; backward tags
    //   past_key "weight".
    //   key_heads released here ([b,nh,s,hs] ≠ [b,nh,total,hs] → no inplace).
    node_meta(20, ann::kNodeTagMetadataKey, "weight");
    node_meta(20, ann::kReleaseAfterMetadataKey, "key_heads");
    // node[21] Concat(past_value, value_heads) → present_value
    //   value_heads has "weight" → Concat output = "weight"; backward tags
    //   past_value "weight".
    node_meta(21, ann::kNodeTagMetadataKey, "weight");
    node_meta(21, ann::kReleaseAfterMetadataKey, "value_heads");
    // node[22] Transpose(present_key) → key_heads_t
    //   present_key has "weight" → key_heads_t = "weight".
    //   present_key is a graph output (keep) → not released.
    node_meta(22, ann::kNodeTagMetadataKey, "weight");
    // ---- Scaled dot-product attention ------------------------------------
    // node[23] MatMul(query_heads, key_heads_t) → scores
    node_meta(23, ann::kNodeTagMetadataKey, "weight");
    node_meta(23, ann::kReleaseAfterMetadataKey, "query_heads;key_heads_t");
    // node[24] Mul(scores, attn_scale) → scores_scaled
    node_meta(24, ann::kNodeTagMetadataKey, "weight");
    node_meta(24, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(24, ann::kReleaseAfterMetadataKey, "scores");
    // node[25] Add(scores_scaled, attn_bias) → scores_biased
    //   Both inputs last used here; scores_scaled reused (inplace).
    //   Release order: scores_scaled (input[0]), attn_bias (input[1]).
    node_meta(25, ann::kNodeTagMetadataKey, "weight");
    node_meta(25, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(25, ann::kReleaseAfterMetadataKey, "scores_scaled;attn_bias");
    // node[26] Softmax(scores_biased) → attn_weights
    node_meta(26, ann::kNodeTagMetadataKey, "weight");
    node_meta(26, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(26, ann::kReleaseAfterMetadataKey, "scores_biased");
    // node[27] MatMul(attn_weights, present_value) → context
    //   [b,nh,s,total] × [b,nh,total,hs] → [b,nh,s,hs]: shapes differ → no inplace.
    //   present_value is a graph output (keep) → not released.
    node_meta(27, ann::kNodeTagMetadataKey, "weight");
    node_meta(27, ann::kReleaseAfterMetadataKey, "attn_weights");
    // node[28] Transpose(context) → context_t
    //   [b,nh,s,hs]→[b,s,nh,hs]: same total byte size → kEqual.
    node_meta(28, ann::kNodeTagMetadataKey, "weight");
    node_meta(28, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(28, ann::kReleaseAfterMetadataKey, "context");
    // node[29] Reshape(context_t, merge_shape) → attn_out
    //   [b,s,nh,hs]→[b,s,h]: same total byte size → kEqual.
    node_meta(29, ann::kNodeTagMetadataKey, "weight");
    node_meta(29, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(29, ann::kReleaseAfterMetadataKey, "context_t");
    // ---- Output projection + first residual --------------------------------
    // node[30] MatMul(attn_out, o_proj.weight) → attn_proj
    node_meta(30, ann::kNodeTagMetadataKey, "weight");
    node_meta(30, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(30, ann::kReleaseAfterMetadataKey, "attn_out");
    // node[31] Add(hidden, attn_proj) → hidden2
    //   hidden last used here (last_use=31); same shape → inplace.
    node_meta(31, ann::kNodeTagMetadataKey, "weight");
    node_meta(31, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(31, ann::kReleaseAfterMetadataKey, "hidden;attn_proj");
    // ---- Second RMSNorm (inlined): nodes 32-37 --------------------------------
    // node[32] Mul(hidden2, hidden2) → ln2_sq
    //   hidden2 last_use=44 (Add below) → no inplace here.
    node_meta(32, ann::kNodeTagMetadataKey, "weight");
    // node[33] ReduceMean(ln2_sq, rms_axes) → ln2_mean
    node_meta(33, ann::kNodeTagMetadataKey, "weight");
    node_meta(33, ann::kReleaseAfterMetadataKey, "ln2_sq");
    // node[34] Add(ln2_mean, rms_eps) → ln2_meaneps
    node_meta(34, ann::kNodeTagMetadataKey, "weight");
    node_meta(34, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(34, ann::kReleaseAfterMetadataKey, "ln2_mean");
    // node[35] Sqrt(ln2_meaneps) → ln2_rms
    node_meta(35, ann::kNodeTagMetadataKey, "weight");
    node_meta(35, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(35, ann::kReleaseAfterMetadataKey, "ln2_meaneps");
    // node[36] Div(hidden2, ln2_rms) → ln2_norm
    //   hidden2 last_use=44 → no inplace; ln2_rms released.
    node_meta(36, ann::kNodeTagMetadataKey, "weight");
    node_meta(36, ann::kReleaseAfterMetadataKey, "ln2_rms");
    // node[37] Mul(ln2_norm, post_attention_layernorm.weight) → normed2
    node_meta(37, ann::kNodeTagMetadataKey, "weight");
    node_meta(37, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(37, ann::kReleaseAfterMetadataKey, "ln2_norm");
    // ---- SwiGLU MLP -------------------------------------------------------
    // node[38] MatMul(normed2, gate_proj.weight) → gate
    //   normed2 last_use=41 → not released here.
    node_meta(38, ann::kNodeTagMetadataKey, "weight");
    // node[39] Sigmoid(gate) → gate_sigmoid
    //   gate last_use=40 → not released here.
    node_meta(39, ann::kNodeTagMetadataKey, "weight");
    // node[40] Mul(gate, gate_sigmoid) → gate_silu
    node_meta(40, ann::kNodeTagMetadataKey, "weight");
    node_meta(40, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(40, ann::kReleaseAfterMetadataKey, "gate;gate_sigmoid");
    // node[41] MatMul(normed2, up_proj.weight) → up
    //   normed2 last used here → released; [b,s,h] ≠ [b,s,i] → no inplace.
    node_meta(41, ann::kNodeTagMetadataKey, "weight");
    node_meta(41, ann::kReleaseAfterMetadataKey, "normed2");
    // node[42] Mul(gate_silu, up) → mlp_hidden
    node_meta(42, ann::kNodeTagMetadataKey, "weight");
    node_meta(42, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(42, ann::kReleaseAfterMetadataKey, "gate_silu;up");
    // node[43] MatMul(mlp_hidden, down_proj.weight) → mlp_out
    //   [b,s,i] ≠ [b,s,h] → no inplace.
    node_meta(43, ann::kNodeTagMetadataKey, "weight");
    node_meta(43, ann::kReleaseAfterMetadataKey, "mlp_hidden");
    // node[44] Add(hidden2, mlp_out) → hidden3
    node_meta(44, ann::kNodeTagMetadataKey, "weight");
    node_meta(44, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(44, ann::kReleaseAfterMetadataKey, "hidden2;mlp_out");
    // ---- Final RMSNorm (inlined): nodes 45-50 --------------------------------
    // node[45] Mul(hidden3, hidden3) → lnf_sq
    //   hidden3 last_use=49 (Div below) → no inplace here.
    node_meta(45, ann::kNodeTagMetadataKey, "weight");
    // node[46] ReduceMean(lnf_sq, rms_axes) → lnf_mean
    node_meta(46, ann::kNodeTagMetadataKey, "weight");
    node_meta(46, ann::kReleaseAfterMetadataKey, "lnf_sq");
    // node[47] Add(lnf_mean, rms_eps) → lnf_meaneps
    node_meta(47, ann::kNodeTagMetadataKey, "weight");
    node_meta(47, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(47, ann::kReleaseAfterMetadataKey, "lnf_mean");
    // node[48] Sqrt(lnf_meaneps) → lnf_rms
    node_meta(48, ann::kNodeTagMetadataKey, "weight");
    node_meta(48, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(48, ann::kReleaseAfterMetadataKey, "lnf_meaneps");
    // node[49] Div(hidden3, lnf_rms) → lnf_norm
    //   hidden3 and lnf_rms both last used here; hidden3 reused (inplace).
    //   Release order: hidden3 (input[0]), lnf_rms (input[1]).
    node_meta(49, ann::kNodeTagMetadataKey, "weight");
    node_meta(49, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(49, ann::kReleaseAfterMetadataKey, "hidden3;lnf_rms");
    // node[50] Mul(lnf_norm, norm.weight) → normed_final
    node_meta(50, ann::kNodeTagMetadataKey, "weight");
    node_meta(50, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(50, ann::kReleaseAfterMetadataKey, "lnf_norm");
    // node[51] MatMul(normed_final, lm_head.weight) → logits
    //   [b,s,h] ≠ [b,s,vocab] → no inplace.
    node_meta(51, ann::kNodeTagMetadataKey, "weight");
    node_meta(51, ann::kReleaseAfterMetadataKey, "normed_final");

    // Per-value kValueTagMetadataKey on value_info (alphabetical order,
    // matching the AppendValueInfo calls above).
    // value_info[0]  attn_bias
    {
      StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[1]  attn_out
    {
      StringStringEntryProto *entry = graph->mutable_value_info(1)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[2]  attn_proj
    {
      StringStringEntryProto *entry = graph->mutable_value_info(2)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[3]  attn_weights
    {
      StringStringEntryProto *entry = graph->mutable_value_info(3)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[4]  context
    {
      StringStringEntryProto *entry = graph->mutable_value_info(4)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[5]  context_t
    {
      StringStringEntryProto *entry = graph->mutable_value_info(5)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[6]  gate
    {
      StringStringEntryProto *entry = graph->mutable_value_info(6)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[7]  gate_silu
    {
      StringStringEntryProto *entry = graph->mutable_value_info(7)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[8]  gate_sigmoid
    {
      StringStringEntryProto *entry = graph->mutable_value_info(8)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[9]  hidden
    {
      StringStringEntryProto *entry = graph->mutable_value_info(9)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[10] hidden2
    {
      StringStringEntryProto *entry = graph->mutable_value_info(10)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[11] hidden3
    {
      StringStringEntryProto *entry = graph->mutable_value_info(11)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[12] key
    {
      StringStringEntryProto *entry = graph->mutable_value_info(12)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[13] key_4d
    {
      StringStringEntryProto *entry = graph->mutable_value_info(13)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[14] key_heads
    {
      StringStringEntryProto *entry = graph->mutable_value_info(14)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[15] key_heads_t → "weight" (Transpose of present_key which is "weight")
    {
      StringStringEntryProto *entry = graph->mutable_value_info(15)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[16] ln1_mean
    {
      StringStringEntryProto *entry = graph->mutable_value_info(16)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[17] ln1_meaneps
    {
      StringStringEntryProto *entry = graph->mutable_value_info(17)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[18] ln1_norm
    {
      StringStringEntryProto *entry = graph->mutable_value_info(18)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[19] ln1_rms
    {
      StringStringEntryProto *entry = graph->mutable_value_info(19)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[20] ln1_sq
    {
      StringStringEntryProto *entry = graph->mutable_value_info(20)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[21] ln2_mean
    {
      StringStringEntryProto *entry = graph->mutable_value_info(21)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[22] ln2_meaneps
    {
      StringStringEntryProto *entry = graph->mutable_value_info(22)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[23] ln2_norm
    {
      StringStringEntryProto *entry = graph->mutable_value_info(23)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[24] ln2_rms
    {
      StringStringEntryProto *entry = graph->mutable_value_info(24)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[25] ln2_sq
    {
      StringStringEntryProto *entry = graph->mutable_value_info(25)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[26] lnf_mean
    {
      StringStringEntryProto *entry = graph->mutable_value_info(26)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[27] lnf_meaneps
    {
      StringStringEntryProto *entry = graph->mutable_value_info(27)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[28] lnf_norm
    {
      StringStringEntryProto *entry = graph->mutable_value_info(28)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[29] lnf_rms
    {
      StringStringEntryProto *entry = graph->mutable_value_info(29)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[30] lnf_sq
    {
      StringStringEntryProto *entry = graph->mutable_value_info(30)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[31] mask_4d → "weight" (backward-tagged via Sub)
    {
      StringStringEntryProto *entry = graph->mutable_value_info(31)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[32] mask_float → "weight" (backward-tagged via Unsqueeze → Sub)
    {
      StringStringEntryProto *entry = graph->mutable_value_info(32)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[33] mask_inv
    {
      StringStringEntryProto *entry = graph->mutable_value_info(33)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[34] mlp_hidden
    {
      StringStringEntryProto *entry = graph->mutable_value_info(34)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[35] mlp_out
    {
      StringStringEntryProto *entry = graph->mutable_value_info(35)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[36] normed1
    {
      StringStringEntryProto *entry = graph->mutable_value_info(36)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[37] normed2
    {
      StringStringEntryProto *entry = graph->mutable_value_info(37)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[38] normed_final
    {
      StringStringEntryProto *entry = graph->mutable_value_info(38)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[39] query
    {
      StringStringEntryProto *entry = graph->mutable_value_info(39)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[40] query_4d
    {
      StringStringEntryProto *entry = graph->mutable_value_info(40)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[41] query_heads
    {
      StringStringEntryProto *entry = graph->mutable_value_info(41)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[42] scores
    {
      StringStringEntryProto *entry = graph->mutable_value_info(42)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[43] scores_biased
    {
      StringStringEntryProto *entry = graph->mutable_value_info(43)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[44] scores_scaled
    {
      StringStringEntryProto *entry = graph->mutable_value_info(44)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[45] up
    {
      StringStringEntryProto *entry = graph->mutable_value_info(45)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[46] value
    {
      StringStringEntryProto *entry = graph->mutable_value_info(46)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[47] value_4d
    {
      StringStringEntryProto *entry = graph->mutable_value_info(47)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // value_info[48] value_heads
    {
      StringStringEntryProto *entry = graph->mutable_value_info(48)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Per-value kValueTagMetadataKey on graph outputs.
    // output[0] logits → "weight"
    {
      StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // output[1] present_key → "weight" (Concat with key_heads which is "weight")
    {
      StringStringEntryProto *entry = graph->mutable_output(1)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // output[2] present_value → "weight" (Concat with value_heads which is "weight")
    {
      StringStringEntryProto *entry = graph->mutable_output(2)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Per-value kValueTagMetadataKey on graph inputs.
    // input[0] input_ids → "weight" (direct graph input seed)
    {
      StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // input[1] attention_mask → "weight" (Cast backward propagation)
    {
      StringStringEntryProto *entry = graph->mutable_input(1)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // input[2] past_key → "weight" (Concat backward from present_key="weight")
    {
      StringStringEntryProto *entry = graph->mutable_input(2)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }
    // input[3] past_value → "weight" (Concat backward from present_value="weight")
    {
      StringStringEntryProto *entry = graph->mutable_input(3)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value("weight");
    }

    // Per-value kValueTagMetadataKey on initializers (insertion order, see
    // AddInitializer calls above).
    const auto init_meta = [&graph](std::size_t i, const char *tag) {
      auto *entry = graph->mutable_initializer(i)->add_metadata_props();
      entry->set_key(ann::kValueTagMetadataKey);
      entry->set_value(tag);
    };
    // initializer[0]  embed_tokens.weight → "weight"
    init_meta(0, "weight");
    // initializer[1]  input_layernorm.weight → "weight"
    init_meta(1, "weight");
    // initializer[2]  q_proj.weight → "weight"
    init_meta(2, "weight");
    // initializer[3]  k_proj.weight → "weight"
    init_meta(3, "weight");
    // initializer[4]  v_proj.weight → "weight"
    init_meta(4, "weight");
    // initializer[5]  o_proj.weight → "weight"
    init_meta(5, "weight");
    // initializer[6]  post_attention_layernorm.weight → "weight"
    init_meta(6, "weight");
    // initializer[7]  gate_proj.weight → "weight"
    init_meta(7, "weight");
    // initializer[8]  up_proj.weight → "weight"
    init_meta(8, "weight");
    // initializer[9]  down_proj.weight → "weight"
    init_meta(9, "weight");
    // initializer[10] norm.weight → "weight"
    init_meta(10, "weight");
    // initializer[11] lm_head.weight → "weight"
    init_meta(11, "weight");
    // initializer[12] rms_axes → "axes"
    init_meta(12, "axes");
    // initializer[13] rms_eps → "weight"
    init_meta(13, "weight");
    // initializer[14] head_shape → "shape"
    init_meta(14, "shape");
    // initializer[15] merge_shape → "shape"
    init_meta(15, "shape");
    // initializer[16] attn_scale → "weight"
    init_meta(16, "weight");
    // initializer[17] mask_axes → "axes"
    init_meta(17, "axes");
    // initializer[18] mask_one → "weight"
    init_meta(18, "weight");
    // initializer[19] mask_neg → "weight"
    init_meta(19, "weight");
  }

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
