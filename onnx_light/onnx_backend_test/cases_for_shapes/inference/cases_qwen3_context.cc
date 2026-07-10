// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kQ3IrVersion = 13;

// Tiny "Qwen3"-style configuration.  The values are intentionally small so
// the generated model stays compact while keeping the architecture faithful.
// Key Qwen3-specific characteristics reproduced here:
//
//  * Causal mask built from ``Range`` outputs: one Range (0..total_seq-1)
//    provides key positions; a second Range (past_seq..total_seq-1) provides
//    query positions.  ``Unsqueeze + LessOrEqual`` turn them into a 4-D
//    boolean mask of shape ``[Range_dim0, 1, 1, Range_dim0]``.
//
//  * ``Where(mask, attention_scores, -inf)`` broadcasts this mask against the
//    ``[batch_size, num_heads, seq, total_seq]`` scores.  Because ``Range_dim0``
//    appears in dim 0 of the mask while ``batch_size`` appears in dim 0 of the
//    scores, the broadcast produces ``broadcast(Range_dim0, batch_size)`` in
//    the batch dimension of the masked scores — and of subsequent tensors.
//
//  * Over two decoder layers this accumulates into a 240-character symbolic
//    expression in the batch dimension of the final hidden state, which is the
//    pathological input to ``ByteSizeExpr`` that previously made the in-place
//    reuse analysis appear to hang.  The fix (a 128-character threshold guard
//    in ``ByteSizeExpr``) causes it to return ``nullopt`` for those tensors,
//    keeping the memory-profile pass bounded.
//
//   input_ids        int64[batch_size, sequence_length]
//   past_key_0       float[batch_size, 2, past_sequence_length, 4]
//   past_value_0     float[batch_size, 2, past_sequence_length, 4]
//   past_key_1       float[batch_size, 2, past_sequence_length, 4]
//   past_value_1     float[batch_size, 2, past_sequence_length, 4]
//
//   logits           float[b2, sequence_length, vocab_size]
//   present_key_0    float[batch_size, 2, past_seq+seq, 4]
//   present_value_0  float[batch_size, 2, past_seq+seq, 4]
//   present_key_1    float[batch_size, 2, past_seq+seq, 4]
//   present_value_1  float[batch_size, 2, past_seq+seq, 4]
//
// where ``b2`` is the 240-character broadcast expression accumulated after
// two decoder layers.
constexpr int64_t kQ3VocabSize = 16;
constexpr int64_t kQ3HiddenSize = 8;
constexpr int64_t kQ3NumHeads = 2;
constexpr int64_t kQ3HeadSize = 4; // kQ3HiddenSize / kQ3NumHeads
constexpr int64_t kQ3IntermediateSize = 12;

// Deterministic pseudo-random weights in [-0.05, 0.05], LCG seeded by seed.
std::vector<float> Q3RandomWeights(std::size_t count, uint32_t seed) {
  std::vector<float> values(count);
  uint32_t s = seed;
  for (std::size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s & 0x00ffffffu) / static_cast<float>(0x01000000u);
    values[i] = -0.05f + 0.1f * u;
  }
  return values;
}

} // namespace

// ---------------------------------------------------------------------------
// Minimal two-decoder-layer Qwen3-style causal language model (context
// encoding phase).  The graph is a faithful but scaled-down translation of
// the bench_qwen3_compute_context_memory model that exposed the performance
// regression in in-place reuse analysis (issue #3356).  External weights in
// the original model are replaced by deterministic pseudo-random initializers.
//
// Architecture (two Qwen3 decoder blocks + LM head):
//
//   --- Shared setup (causal mask) ---
//   k_range     = Range(0, total_seq, 1)               # [Range_dim0]
//   q_range     = Range(past_seq, total_seq, 1)        # [Range_dim0]
//   k_pos       = Unsqueeze(k_range, [1,2,3])          # [Range_dim0,1,1,1]
//   q_col       = Unsqueeze(q_range, [0,1,2])          # [1,1,1,Range_dim0]
//   causal_mask = LessOrEqual(k_pos, q_col)            # [Range_dim0,1,1,Range_dim0]
//
//   --- Decoder layer 0 ---
//   embed      = Gather(embed_w, input_ids)            # [b, s, H]
//   normed0    = RMSNorm(embed, n0_w)                  # [b, s, H]
//   q/k/v      = MatMul(normed0, {q,k,v}0_w)          # [b, s, H]
//   (reshape + transpose to multi-head)
//   pres_k0    = Concat(past_k0, k0_t, axis=2)        # [b, 2, past+s, 4]
//   pres_v0    = Concat(past_v0, v0_t, axis=2)        # [b, 2, past+s, 4]
//   scores0    = MatMul(q0_t, Transpose(pres_k0))     # [b, 2, s, past+s]
//   masked0    = Where(causal_mask, scores0, -inf)     # [bcast(R,b), 2, s, bcast(R,past+s)]
//   probs0     = Softmax(masked0)                      # same
//   attn0      = MatMul(probs0, pres_v0)              # [bcast(bcast(R,b),b), 2, s, 4]
//   h1         = embed + o_proj(attn0_flat)            # batch dim = b1 (79 chars)
//   h2         = h1 + MLP(RMSNorm(h1))                # batch dim = b1
//
//   --- Decoder layer 1 (same structure) ---
//   h3         = h2 + o_proj(attn1_flat)              # batch dim = b2 (240 chars)
//   h4         = h3 + MLP(RMSNorm(h3))                # batch dim = b2
//
//   logits     = MatMul(RMSNorm(h4), lm_head_w)       # [b2, s, vocab]
//
// The ``Where`` operation in each layer is the root cause of the batch-dim
// explosion: the 4-D causal mask has ``Range_dim0`` in its first dimension,
// which broadcasts with ``batch_size`` and accumulates across layers.
// After two layers the batch-dim expression exceeds 128 characters, at which
// point the fixed ``ByteSizeExpr`` returns ``nullopt`` instead of building an
// astronomically long symbolic product.
//
// The case carries no DataSet; it is shape-inference-only.
// ---------------------------------------------------------------------------
void RegisterQwen3ContextShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(23);

  const std::string name = "test_cc_shape_inference_qwen3_context";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  ModelProto &model = tc.model;
  InitModel(model, kQ3IrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  const auto kFloat = static_cast<int32_t>(DataType::FLOAT);
  const auto kInt64 = static_cast<int32_t>(DataType::INT64);
  const auto kBool = static_cast<int32_t>(DataType::BOOL);

  // ---- Small constant initializers ----------------------------------------
  // Scalar and axes initializers reused across the graph.
  AddInitializer<int64_t>(*graph, "zero_sc", {}, {0});  // Range start (0)
  AddInitializer<int64_t>(*graph, "one_sc", {}, {1});   // Range step (1)
  AddInitializer<int64_t>(*graph, "idx_sc_1", {}, {1}); // Gather index 1 → seq_len
  AddInitializer<int64_t>(*graph, "idx_sc_2", {}, {2}); // Gather index 2 → past_seq
  AddInitializerShape(*graph, "axes_123", {1, 2, 3});   // Unsqueeze axes for k_range
  AddInitializerShape(*graph, "axes_012", {0, 1, 2});   // Unsqueeze axes for q_range
  AddInitializerShape(*graph, "head_shape_q3", {0, 0, kQ3NumHeads, kQ3HeadSize});
  AddInitializerShape(*graph, "merge_shape_q3", {0, 0, kQ3HiddenSize});
  AddInitializer<float>(*graph, "neg_inf_q3", {}, {-1e9f}); // Where fill value

  // ---- Shared causal mask setup -------------------------------------------
  // Extract sequence lengths from the input shapes.
  AddNode(*graph, "Shape", {"input_ids"}, {"ids_shape"});
  AddNode(*graph, "Shape", {"past_key_0"}, {"pk_shape"});
  AddNode(*graph, "Gather", {"ids_shape", "idx_sc_1"}, {"seq_sc"});
  AddNode(*graph, "Gather", {"pk_shape", "idx_sc_2"}, {"past_sc"});
  AddNode(*graph, "Add", {"seq_sc", "past_sc"}, {"total_sc"});

  // Range for key positions (0..total_seq-1) and query positions (past..total-1).
  AddNode(*graph, "Range", {"zero_sc", "total_sc", "one_sc"}, {"k_range"});
  AddNode(*graph, "Range", {"past_sc", "total_sc", "one_sc"}, {"q_range"});

  // Unsqueeze to 4-D for broadcasting against [batch, heads, seq, total_seq].
  AddNode(*graph, "Unsqueeze", {"k_range", "axes_123"}, {"k_pos"});
  AddNode(*graph, "Unsqueeze", {"q_range", "axes_012"}, {"q_col"});

  // Causal mask: k <= q  ⇒  key position can be attended by query position.
  // Shape: [Range_dim0, 1, 1, Range_dim0].  When Where'd against scores of
  // shape [batch_size, heads, seq, total_seq], the batch dimension picks up
  // broadcast(Range_dim0, batch_size).
  AddNode(*graph, "LessOrEqual", {"k_pos", "q_col"}, {"causal_mask"});

  // ---- Decoder layer 0 ----------------------------------------------------

  // Token embedding.
  AddNode(*graph, "Gather", {"embed_w", "input_ids"}, {"embed"});

  // Input RMSNorm + Q/K/V projections.
  AddNode(*graph, "RMSNormalization", {"embed", "norm0_w"}, {"normed0"});
  AddNode(*graph, "MatMul", {"normed0", "q0_w"}, {"q0"});
  AddNode(*graph, "MatMul", {"normed0", "k0_w"}, {"k0"});
  AddNode(*graph, "MatMul", {"normed0", "v0_w"}, {"v0"});

  // Reshape and transpose Q, K, V to multi-head layout.
  AddNode(*graph, "Reshape", {"q0", "head_shape_q3"}, {"q0_r"});
  NodeProto &q0_tp = AddNode(*graph, "Transpose", {"q0_r"}, {"q0_t"});
  AddAttribute<std::vector<int64_t>>(q0_tp, "perm", {0, 2, 1, 3});

  AddNode(*graph, "Reshape", {"k0", "head_shape_q3"}, {"k0_r"});
  NodeProto &k0_tp = AddNode(*graph, "Transpose", {"k0_r"}, {"k0_t"});
  AddAttribute<std::vector<int64_t>>(k0_tp, "perm", {0, 2, 1, 3});

  NodeProto &kc0 = AddNode(*graph, "Concat", {"past_key_0", "k0_t"}, {"present_key_0"});
  AddAttribute<int64_t>(kc0, "axis", 2);

  AddNode(*graph, "Reshape", {"v0", "head_shape_q3"}, {"v0_r"});
  NodeProto &v0_tp = AddNode(*graph, "Transpose", {"v0_r"}, {"v0_t"});
  AddAttribute<std::vector<int64_t>>(v0_tp, "perm", {0, 2, 1, 3});

  NodeProto &vc0 = AddNode(*graph, "Concat", {"past_value_0", "v0_t"}, {"present_value_0"});
  AddAttribute<int64_t>(vc0, "axis", 2);

  // Attention: MatMul(Q, K^T) → Where(causal) → Softmax → MatMul(V).
  NodeProto &pk0_tp = AddNode(*graph, "Transpose", {"present_key_0"}, {"pres_k0_t"});
  AddAttribute<std::vector<int64_t>>(pk0_tp, "perm", {0, 1, 3, 2});

  AddNode(*graph, "MatMul", {"q0_t", "pres_k0_t"}, {"scores0"});
  AddNode(*graph, "Where", {"causal_mask", "scores0", "neg_inf_q3"}, {"masked0"});
  NodeProto &sf0 = AddNode(*graph, "Softmax", {"masked0"}, {"probs0"});
  AddAttribute<int64_t>(sf0, "axis", -1);
  AddNode(*graph, "MatMul", {"probs0", "present_value_0"}, {"attn0"});

  // Merge heads and apply output projection.
  NodeProto &a0_tp = AddNode(*graph, "Transpose", {"attn0"}, {"attn0_t"});
  AddAttribute<std::vector<int64_t>>(a0_tp, "perm", {0, 2, 1, 3});
  AddNode(*graph, "Reshape", {"attn0_t", "merge_shape_q3"}, {"attn0_flat"});
  AddNode(*graph, "MatMul", {"attn0_flat", "o0_w"}, {"attn0_proj"});
  AddNode(*graph, "Add", {"embed", "attn0_proj"}, {"h1"});

  // Post-attention RMSNorm + SwiGLU MLP.
  AddNode(*graph, "RMSNormalization", {"h1", "norm1_w"}, {"normed1"});
  AddNode(*graph, "MatMul", {"normed1", "gate0_w"}, {"gate0"});
  AddNode(*graph, "Sigmoid", {"gate0"}, {"gate_sig0"});
  AddNode(*graph, "Mul", {"gate0", "gate_sig0"}, {"silu0"});
  AddNode(*graph, "MatMul", {"normed1", "up0_w"}, {"up0"});
  AddNode(*graph, "Mul", {"silu0", "up0"}, {"mlp_h0"});
  AddNode(*graph, "MatMul", {"mlp_h0", "down0_w"}, {"mlp_out0"});
  AddNode(*graph, "Add", {"h1", "mlp_out0"}, {"h2"});

  // ---- Decoder layer 1 ----------------------------------------------------
  // Same structure.  The batch dimension of h2 (and all layer-1 intermediates)
  // is b1 = broadcast(batch_size, broadcast(broadcast(Range_dim0, batch_size),
  //          batch_size)), a 79-character expression inherited from layer 0's
  // attention + residual add.
  AddNode(*graph, "RMSNormalization", {"h2", "norm2_w"}, {"normed2"});
  AddNode(*graph, "MatMul", {"normed2", "q1_w"}, {"q1"});
  AddNode(*graph, "MatMul", {"normed2", "k1_w"}, {"k1"});
  AddNode(*graph, "MatMul", {"normed2", "v1_w"}, {"v1"});

  AddNode(*graph, "Reshape", {"q1", "head_shape_q3"}, {"q1_r"});
  NodeProto &q1_tp = AddNode(*graph, "Transpose", {"q1_r"}, {"q1_t"});
  AddAttribute<std::vector<int64_t>>(q1_tp, "perm", {0, 2, 1, 3});

  AddNode(*graph, "Reshape", {"k1", "head_shape_q3"}, {"k1_r"});
  NodeProto &k1_tp = AddNode(*graph, "Transpose", {"k1_r"}, {"k1_t"});
  AddAttribute<std::vector<int64_t>>(k1_tp, "perm", {0, 2, 1, 3});

  NodeProto &kc1 = AddNode(*graph, "Concat", {"past_key_1", "k1_t"}, {"present_key_1"});
  AddAttribute<int64_t>(kc1, "axis", 2);

  AddNode(*graph, "Reshape", {"v1", "head_shape_q3"}, {"v1_r"});
  NodeProto &v1_tp = AddNode(*graph, "Transpose", {"v1_r"}, {"v1_t"});
  AddAttribute<std::vector<int64_t>>(v1_tp, "perm", {0, 2, 1, 3});

  NodeProto &vc1 = AddNode(*graph, "Concat", {"past_value_1", "v1_t"}, {"present_value_1"});
  AddAttribute<int64_t>(vc1, "axis", 2);

  NodeProto &pk1_tp = AddNode(*graph, "Transpose", {"present_key_1"}, {"pres_k1_t"});
  AddAttribute<std::vector<int64_t>>(pk1_tp, "perm", {0, 1, 3, 2});

  // Layer 1 attention.  scores1 has batch = broadcast(b1, batch_size) (102 chars).
  // After Where with the causal mask, the batch dim reaches 125 chars.
  // After MatMul with present_value_1 (batch = batch_size), it reaches 148 chars —
  // above the kMaxSymbolicDimExprLength threshold.  ByteSizeExpr returns nullopt
  // from here on, confirming the fix prevents the quadratic run-time.
  AddNode(*graph, "MatMul", {"q1_t", "pres_k1_t"}, {"scores1"});
  AddNode(*graph, "Where", {"causal_mask", "scores1", "neg_inf_q3"}, {"masked1"});
  NodeProto &sf1 = AddNode(*graph, "Softmax", {"masked1"}, {"probs1"});
  AddAttribute<int64_t>(sf1, "axis", -1);
  AddNode(*graph, "MatMul", {"probs1", "present_value_1"}, {"attn1"});

  NodeProto &a1_tp = AddNode(*graph, "Transpose", {"attn1"}, {"attn1_t"});
  AddAttribute<std::vector<int64_t>>(a1_tp, "perm", {0, 2, 1, 3});
  AddNode(*graph, "Reshape", {"attn1_t", "merge_shape_q3"}, {"attn1_flat"});
  AddNode(*graph, "MatMul", {"attn1_flat", "o1_w"}, {"attn1_proj"});
  AddNode(*graph, "Add", {"h2", "attn1_proj"}, {"h3"});

  AddNode(*graph, "RMSNormalization", {"h3", "norm3_w"}, {"normed3"});
  AddNode(*graph, "MatMul", {"normed3", "gate1_w"}, {"gate1"});
  AddNode(*graph, "Sigmoid", {"gate1"}, {"gate_sig1"});
  AddNode(*graph, "Mul", {"gate1", "gate_sig1"}, {"silu1"});
  AddNode(*graph, "MatMul", {"normed3", "up1_w"}, {"up1"});
  AddNode(*graph, "Mul", {"silu1", "up1"}, {"mlp_h1"});
  AddNode(*graph, "MatMul", {"mlp_h1", "down1_w"}, {"mlp_out1"});
  AddNode(*graph, "Add", {"h3", "mlp_out1"}, {"h4"});

  // ---- Final RMSNorm + LM head --------------------------------------------
  AddNode(*graph, "RMSNormalization", {"h4", "norm_final_w"}, {"normed_final"});
  AddNode(*graph, "MatMul", {"normed_final", "lm_head_w"}, {"logits"});

  // ---- Random weight initializers -----------------------------------------
  uint32_t seed = 1u;
  const auto nw = [&](const char *nm, const std::vector<int64_t> &dims) {
    std::size_t count = 1;
    for (int64_t d : dims) {
      count *= static_cast<std::size_t>(d);
    }
    AddInitializer<float>(*graph, nm, dims, Q3RandomWeights(count, seed++));
  };

  nw("embed_w", {kQ3VocabSize, kQ3HiddenSize});
  nw("norm0_w", {kQ3HiddenSize});
  nw("q0_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("k0_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("v0_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("o0_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("norm1_w", {kQ3HiddenSize});
  nw("gate0_w", {kQ3HiddenSize, kQ3IntermediateSize});
  nw("up0_w", {kQ3HiddenSize, kQ3IntermediateSize});
  nw("down0_w", {kQ3IntermediateSize, kQ3HiddenSize});
  nw("norm2_w", {kQ3HiddenSize});
  nw("q1_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("k1_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("v1_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("o1_w", {kQ3HiddenSize, kQ3HiddenSize});
  nw("norm3_w", {kQ3HiddenSize});
  nw("gate1_w", {kQ3HiddenSize, kQ3IntermediateSize});
  nw("up1_w", {kQ3HiddenSize, kQ3IntermediateSize});
  nw("down1_w", {kQ3IntermediateSize, kQ3HiddenSize});
  nw("norm_final_w", {kQ3HiddenSize});
  nw("lm_head_w", {kQ3HiddenSize, kQ3VocabSize});

  // ---- Graph inputs (fully dynamic shapes) --------------------------------
  AppendValueInfo(*graph->add_input(), "input_ids", kInt64, {"batch_size", "sequence_length"});
  AppendValueInfo(
      *graph->add_input(), "past_key_0", kFloat,
      {"batch_size", DimSpec(kQ3NumHeads), "past_sequence_length", DimSpec(kQ3HeadSize)});
  AppendValueInfo(
      *graph->add_input(), "past_value_0", kFloat,
      {"batch_size", DimSpec(kQ3NumHeads), "past_sequence_length", DimSpec(kQ3HeadSize)});
  AppendValueInfo(
      *graph->add_input(), "past_key_1", kFloat,
      {"batch_size", DimSpec(kQ3NumHeads), "past_sequence_length", DimSpec(kQ3HeadSize)});
  AppendValueInfo(
      *graph->add_input(), "past_value_1", kFloat,
      {"batch_size", DimSpec(kQ3NumHeads), "past_sequence_length", DimSpec(kQ3HeadSize)});

  // ---- Intermediate value_info shapes -------------------------------------
  // Symbolic dim shorthands used throughout.
  const DimSpec kBatch{"batch_size"};
  const DimSpec kSeq{"sequence_length"};
  const DimSpec kPast{"past_sequence_length"};
  const DimSpec kTotal{"past_sequence_length+sequence_length"};
  const DimSpec kRangeD{"Range_dim0"};
  const DimSpec kH{kQ3HiddenSize};
  const DimSpec kNH{kQ3NumHeads};
  const DimSpec kHS{kQ3HeadSize};
  const DimSpec kInter{kQ3IntermediateSize};
  const DimSpec k1{int64_t{1}};

  // Batch-dimension broadcast expressions that accumulate across layers.
  // Layer 0: the Where + MatMul + residual-Add chain.
  // b0 = "batch_size"                                                     (10 chars)
  // r  = "Range_dim0"                                                     (10 chars)
  // Layer 0 Where batch:   broadcast(Range_dim0, batch_size)              (33 chars)
  // Layer 0 attn  batch:   broadcast(broadcast(Range_dim0, batch_size), batch_size)
  //                                                                        (56 chars)
  // b1 (layer 0 output):
  //   broadcast(batch_size, broadcast(broadcast(Range_dim0, batch_size), batch_size))
  //                                                                        (79 chars)
  //
  // Layer 1 scores batch:
  //   broadcast(b1, batch_size)                                           (102 chars)
  // Layer 1 Where  batch:
  //   broadcast(Range_dim0, broadcast(b1, batch_size))                   (125 chars)
  // Layer 1 attn   batch (> kMaxSymbolicDimExprLength=128):
  //   broadcast(broadcast(Range_dim0, broadcast(b1, batch_size)), batch_size)
  //                                                                       (148 chars)
  // b2 (layer 1 output):
  //   broadcast(b1, broadcast(broadcast(Range_dim0, broadcast(b1, batch_size)), batch_size))
  //                                                                       (240 chars)
  const std::string kWhereB0s = "broadcast(Range_dim0, batch_size)";
  const std::string kWhereT0s = "broadcast(Range_dim0, past_sequence_length+sequence_length)";
  const std::string kAttnB0s = "broadcast(broadcast(Range_dim0, batch_size), batch_size)";
  const std::string kB1s =
      "broadcast(batch_size, broadcast(broadcast(Range_dim0, batch_size), batch_size))";
  const std::string kScoreB1s =
      "broadcast(broadcast(batch_size, broadcast(broadcast(Range_dim0, batch_size), batch_size)),"
      " batch_size)";
  const std::string kWhereB1s =
      "broadcast(Range_dim0, broadcast(broadcast(batch_size, broadcast(broadcast(Range_dim0,"
      " batch_size), batch_size)), batch_size))";
  const std::string kAttnB1s =
      "broadcast(broadcast(Range_dim0, broadcast(broadcast(batch_size, broadcast(broadcast("
      "Range_dim0, batch_size), batch_size)), batch_size)), batch_size)";
  const std::string kB2s =
      "broadcast(broadcast(batch_size, broadcast(broadcast(Range_dim0, batch_size), batch_size)),"
      " broadcast(broadcast(Range_dim0, broadcast(broadcast(batch_size, broadcast(broadcast("
      "Range_dim0, batch_size), batch_size)), batch_size)), batch_size))";

  const DimSpec kWhereB0{kWhereB0s};
  const DimSpec kWhereT0{kWhereT0s};
  const DimSpec kAttnB0{kAttnB0s};
  const DimSpec kB1{kB1s};
  const DimSpec kScoreB1{kScoreB1s};
  const DimSpec kWhereB1{kWhereB1s};
  const DimSpec kAttnB1{kAttnB1s};
  const DimSpec kB2{kB2s};

  // Shared setup shapes.
  AppendValueInfo(*graph->add_value_info(), "ids_shape", kInt64, {DimSpec{int64_t{2}}});
  AppendValueInfo(*graph->add_value_info(), "pk_shape", kInt64, {DimSpec{int64_t{4}}});
  AppendValueInfo(*graph->add_value_info(), "seq_sc", kInt64, {});
  AppendValueInfo(*graph->add_value_info(), "past_sc", kInt64, {});
  AppendValueInfo(*graph->add_value_info(), "total_sc", kInt64, {});
  AppendValueInfo(*graph->add_value_info(), "k_range", kInt64, {kRangeD});
  AppendValueInfo(*graph->add_value_info(), "q_range", kInt64, {kRangeD});
  AppendValueInfo(*graph->add_value_info(), "k_pos", kInt64, {kRangeD, k1, k1, k1});
  AppendValueInfo(*graph->add_value_info(), "q_col", kInt64, {k1, k1, k1, kRangeD});
  AppendValueInfo(*graph->add_value_info(), "causal_mask", kBool, {kRangeD, k1, k1, kRangeD});

  // Layer 0 intermediates.
  AppendValueInfo(*graph->add_value_info(), "embed", kFloat, {kBatch, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "normed0", kFloat, {kBatch, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "q0", kFloat, {kBatch, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "k0", kFloat, {kBatch, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "v0", kFloat, {kBatch, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "q0_r", kFloat, {kBatch, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "q0_t", kFloat, {kBatch, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "k0_r", kFloat, {kBatch, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "k0_t", kFloat, {kBatch, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "v0_r", kFloat, {kBatch, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "v0_t", kFloat, {kBatch, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "pres_k0_t", kFloat, {kBatch, kNH, kHS, kTotal});
  AppendValueInfo(*graph->add_value_info(), "scores0", kFloat, {kBatch, kNH, kSeq, kTotal});
  AppendValueInfo(*graph->add_value_info(), "masked0", kFloat, {kWhereB0, kNH, kSeq, kWhereT0});
  AppendValueInfo(*graph->add_value_info(), "probs0", kFloat, {kWhereB0, kNH, kSeq, kWhereT0});
  AppendValueInfo(*graph->add_value_info(), "attn0", kFloat, {kAttnB0, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "attn0_t", kFloat, {kAttnB0, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "attn0_flat", kFloat, {kAttnB0, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "attn0_proj", kFloat, {kAttnB0, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "h1", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "normed1", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "gate0", kFloat, {kB1, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "gate_sig0", kFloat, {kB1, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "silu0", kFloat, {kB1, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "up0", kFloat, {kB1, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "mlp_h0", kFloat, {kB1, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "mlp_out0", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "h2", kFloat, {kB1, kSeq, kH});

  // Layer 1 intermediates.
  AppendValueInfo(*graph->add_value_info(), "normed2", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "q1", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "k1", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "v1", kFloat, {kB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "q1_r", kFloat, {kB1, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "q1_t", kFloat, {kB1, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "k1_r", kFloat, {kB1, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "k1_t", kFloat, {kB1, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "v1_r", kFloat, {kB1, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "v1_t", kFloat, {kB1, kNH, kSeq, kHS});
  // present_key_1 / present_value_1 batch dim is batch_size (Concat MergeDim keeps
  // the first input's symbolic dim when the two differ).
  AppendValueInfo(*graph->add_value_info(), "pres_k1_t", kFloat, {kBatch, kNH, kHS, kTotal});
  AppendValueInfo(*graph->add_value_info(), "scores1", kFloat, {kScoreB1, kNH, kSeq, kTotal});
  AppendValueInfo(*graph->add_value_info(), "masked1", kFloat, {kWhereB1, kNH, kSeq, kWhereT0});
  AppendValueInfo(*graph->add_value_info(), "probs1", kFloat, {kWhereB1, kNH, kSeq, kWhereT0});
  AppendValueInfo(*graph->add_value_info(), "attn1", kFloat, {kAttnB1, kNH, kSeq, kHS});
  AppendValueInfo(*graph->add_value_info(), "attn1_t", kFloat, {kAttnB1, kSeq, kNH, kHS});
  AppendValueInfo(*graph->add_value_info(), "attn1_flat", kFloat, {kAttnB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "attn1_proj", kFloat, {kAttnB1, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "h3", kFloat, {kB2, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "normed3", kFloat, {kB2, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "gate1", kFloat, {kB2, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "gate_sig1", kFloat, {kB2, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "silu1", kFloat, {kB2, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "up1", kFloat, {kB2, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "mlp_h1", kFloat, {kB2, kSeq, kInter});
  AppendValueInfo(*graph->add_value_info(), "mlp_out1", kFloat, {kB2, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "h4", kFloat, {kB2, kSeq, kH});
  AppendValueInfo(*graph->add_value_info(), "normed_final", kFloat, {kB2, kSeq, kH});

  // ---- Graph outputs -------------------------------------------------------
  // KV cache outputs retain batch_size (not b1/b2) because Concat MergeDim
  // keeps the first input's symbolic dim.
  AppendValueInfo(*graph->add_output(), "logits", kFloat, {kB2, kSeq, DimSpec{kQ3VocabSize}});
  AppendValueInfo(*graph->add_output(), "present_key_0", kFloat, {kBatch, kNH, kTotal, kHS});
  AppendValueInfo(*graph->add_output(), "present_value_0", kFloat, {kBatch, kNH, kTotal, kHS});
  AppendValueInfo(*graph->add_output(), "present_key_1", kFloat, {kBatch, kNH, kTotal, kHS});
  AppendValueInfo(*graph->add_output(), "present_value_1", kFloat, {kBatch, kNH, kTotal, kHS});

  registry.push_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
