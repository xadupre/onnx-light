// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// ``qwen3_4_layers_like`` -- a 4-layer Qwen3-style causal language model,
// registered in two variants selected by a single ``fused`` switch:
//
//   * unfused (``test_cc_shape_inference_big_qwen3_4_layers_like``, opset 21):
//     RMSNorm is spelled out (Cast + Pow + ReduceMean + Add + Sqrt +
//     Reciprocal + Mul + Cast + Mul) and the grouped-query attention core is
//     an explicit subgraph (KV-cache concat, head broadcast, scaled QK^T,
//     causal masking, Softmax and attention @ V);
//   * fused (``test_cc_shape_inference_big_qwen3_4_layers_like_fused``,
//     opset 23): every RMSNorm collapses to one ``RMSNormalization`` node and
//     the whole attention core collapses to one ``Attention`` node (which also
//     emits the per-layer present key/value cache).
//
// Both variants share the same public signature; RoPE and the causal-mask
// construction stay explicit in both because they have no fused equivalent.
// The unfused variant additionally carries the golden in-place-reuse,
// value-tag and constant metadata verified by the ``BigModels*`` tests.
//
// Graph signature (identical for both variants):
//
//   Inputs:
//     input_ids               INT64[batch_size, sequence_length]
//     attention_mask          INT64[batch_size, total_sequence_length]
//     past_key_values_key_N   FP16[batch_size, 8, past_sequence_length, 128]
//     past_key_values_value_N FP16[batch_size, 8, past_sequence_length, 128]
//
//   Outputs:
//     output_0                FP16[batch_size, sequence_length, 32000]
//     present_key_values_key_N    FP16[batch_size, 8, past_seq+seq, 128]
//     present_key_values_value_N  FP16[batch_size, 8, past_seq+seq, 128]
// ---------------------------------------------------------------------------
void RegisterQwen3_4LayersLikeShapeInferenceCases(std::vector<TestCase> &registry,
                                                  TestMode /*mode*/) {
  // Produce both the inlined and the fused model from a single switch: the only
  // structural differences are the ops used for RMSNorm / attention (and the
  // opset that provides them).
  for (bool fused : {false, true}) {
    const OpsetId opset = DefaultOpset(fused ? 23 : 21);
    const std::string name = fused ? "test_cc_shape_inference_big_qwen3_4_layers_like_fused"
                                   : "test_cc_shape_inference_big_qwen3_4_layers_like";

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
    tc.rtol = 1e-3f;
    tc.atol = 1e-5f;

    ModelProto &model = tc.emplace_model();
    InitModel(model, /*ir_version=*/10, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    if (fused) {
      // ---- Shape-math / reshape constant initializers -------------------------
      AddInitializer<int64_t>(*graph, "init7_s1_1", {INT64_C(1)}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s1_-1", {INT64_C(1)}, {INT64_C(-1)});
      AddInitializer<int64_t>(*graph, "init7_s_1__1", {}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s2_0_1__1", {INT64_C(2)}, {INT64_C(0), INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_-1_1__1", {INT64_C(3)},
                              {INT64_C(0), INT64_C(-1), INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s_0__2", {}, {INT64_C(0)});
      AddInitializer<int64_t>(*graph, "init7_s_1__2", {}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_1_2__2", {INT64_C(3)},
                              {INT64_C(0), INT64_C(1), INT64_C(2)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_1_3__2", {INT64_C(3)},
                              {INT64_C(0), INT64_C(1), INT64_C(3)});
      AddInitializer<int64_t>(*graph, "init7_s_0__3", {}, {INT64_C(0)});
      AddInitializer<int64_t>(*graph, "init7_s_1__3", {}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_1_2__3", {INT64_C(3)},
                              {INT64_C(0), INT64_C(1), INT64_C(2)});
      AddInitializer<int64_t>(*graph, "init7_s3_1_2_3__3", {INT64_C(3)},
                              {INT64_C(1), INT64_C(2), INT64_C(3)});
      AddInitializer<int64_t>(*graph, "init7_s4_0_0_16_128", {INT64_C(4)},
                              {INT64_C(0), INT64_C(0), INT64_C(16), INT64_C(128)});
      AddInitializer<int64_t>(*graph, "init7_s4_0_0_8_128", {INT64_C(4)},
                              {INT64_C(0), INT64_C(0), INT64_C(8), INT64_C(128)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_0_2048", {INT64_C(3)},
                              {INT64_C(0), INT64_C(0), INT64_C(2048)});

      // ---- Per-layer projection weight initializers ---------------------------
      for (int layer = 0; layer < 4; ++layer) {
        const std::string weight_prefix = "p_model_layers_" + std::to_string(layer);
        const std::string q_proj = weight_prefix + "_self_attn_q_proj_weight::T10";
        const std::string k_proj = weight_prefix + "_self_attn_k_proj_weight::T10";
        const std::string v_proj = weight_prefix + "_self_attn_v_proj_weight::T10";
        const std::string o_proj = weight_prefix + "_self_attn_o_proj_weight::T10";
        const std::string gate_proj = weight_prefix + "_mlp_gate_proj_weight::T10";
        const std::string up_proj = weight_prefix + "_mlp_up_proj_weight::T10";
        const std::string down_proj = weight_prefix + "_mlp_down_proj_weight::T10";
        AddInitializer<uint16_t>(*graph, q_proj.c_str(), {INT64_C(1024), INT64_C(2048)}, {});
        AddInitializer<uint16_t>(*graph, k_proj.c_str(), {INT64_C(1024), INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, v_proj.c_str(), {INT64_C(1024), INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, o_proj.c_str(), {INT64_C(2048), INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, gate_proj.c_str(), {INT64_C(1024), INT64_C(3072)}, {});
        AddInitializer<uint16_t>(*graph, up_proj.c_str(), {INT64_C(1024), INT64_C(3072)}, {});
        AddInitializer<uint16_t>(*graph, down_proj.c_str(), {INT64_C(3072), INT64_C(1024)}, {});
      }

      AddInitializer<uint16_t>(*graph, "p_lm_head_weight::T10", {INT64_C(1024), INT64_C(32000)},
                               {});
      AddInitializer<float>(
          *graph, "to_322", {INT64_C(1), INT64_C(1), INT64_C(64)},
          {1.0f,           0.80566406f,    0.64941406f,    0.5234375f,     0.42163086f,
           0.33984375f,    0.27392578f,    0.22070312f,    0.17785645f,    0.14331055f,
           0.11547852f,    0.093078613f,   0.075012207f,   0.060424805f,   0.048706055f,
           0.039245605f,   0.031616211f,   0.025482178f,   0.02053833f,    0.016555786f,
           0.013336182f,   0.010749817f,   0.0086593628f,  0.0069770813f,  0.0056228638f,
           0.0045318604f,  0.0036525726f,  0.0029430389f,  0.0023708344f,  0.0019111633f,
           0.001540184f,   0.0012407303f,  0.0010004044f,  0.0008058548f,  0.00064945221f,
           0.00052309036f, 0.00042176247f, 0.00033974648f, 0.00027394295f, 0.00022065639f,
           0.00017786026f, 0.00014328957f, 0.0001154542f,  9.304285e-05f,  7.4982643e-05f,
           6.043911e-05f,  4.8696995e-05f, 3.9219856e-05f, 3.1650066e-05f, 2.5510788e-05f,
           2.0563602e-05f, 1.6570091e-05f, 1.335144e-05f,  1.0728836e-05f, 8.6426735e-06f,
           6.9737434e-06f, 5.6028366e-06f, 4.529953e-06f,  3.6358833e-06f, 2.9206276e-06f,
           2.3841858e-06f, 1.9073486e-06f, 1.5497208e-06f, 1.2516975e-06f});

      // ---- Per-layer norm weight initializers ---------------------------------
      for (int layer = 0; layer < 4; ++layer) {
        const std::string norm_prefix = "model.layers." + std::to_string(layer);
        const std::string q_norm = norm_prefix + ".self_attn.q_norm.weight";
        const std::string k_norm = norm_prefix + ".self_attn.k_norm.weight";
        const std::string input_ln = norm_prefix + ".input_layernorm.weight";
        const std::string post_ln = norm_prefix + ".post_attention_layernorm.weight";
        // All-same FP16 initializer (value=0x3C00).
        AddInitializer<uint16_t>(*graph, q_norm.c_str(), {INT64_C(128)},
                                 std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
        AddInitializer<uint16_t>(*graph, k_norm.c_str(), {INT64_C(128)},
                                 std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
        AddInitializer<uint16_t>(*graph, input_ln.c_str(), {INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, post_ln.c_str(), {INT64_C(1024)}, {});
      }

      AddInitializer<uint16_t>(*graph, "model.norm.weight", {INT64_C(1024)}, {});
      AddInitializer<uint16_t>(*graph, "lm_head.weight", {INT64_C(32000), INT64_C(1024)}, {});

      // ---- RoPE tables + causal mask (shared across the transformer layers) ----
      // This section is identical to the inlined case: it derives the RoPE
      // cos/sin tables (``unsqueeze_16`` / ``unsqueeze_17``) and the boolean
      // causal mask (``and_2``) that every layer feeds into ``Attention``.
      {
        NodeProto &n = AddNode(*graph, "Shape", {"input_ids"}, {"input_ids::Shape1:2"});
        AddAttribute<int64_t>(n, "end", INT64_C(2));
        AddAttribute<int64_t>(n, "start", INT64_C(1));
      }
      {
        NodeProto &n = AddNode(*graph, "Shape", {"past_key_values_key_0"},
                               {"past_key_values_key_0::Shape2:3"});
        AddAttribute<int64_t>(n, "end", INT64_C(3));
        AddAttribute<int64_t>(n, "start", INT64_C(2));
      }
      {
        NodeProto &n = AddNode(*graph, "Shape", {"past_key_values_value_2"},
                               {"past_key_values_value_2::Shape:1"});
        AddAttribute<int64_t>(n, "end", INT64_C(1));
        AddAttribute<int64_t>(n, "start", INT64_C(0));
      }
      AddNode(*graph, "Add", {"input_ids::Shape1:2", "past_key_values_key_0::Shape2:3"},
              {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"});
      AddNode(*graph, "Squeeze", {"past_key_values_key_0::Shape2:3"}, {"dim1::Sq__1"});
      AddNode(*graph, "Squeeze", {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"},
              {"dim2::Sq__1"});
      AddNode(*graph, "Range", {"dim1::Sq__1", "dim2::Sq__1", "init7_s_1__1"},
              {"_onx_range_dim1::Sq__1"});
      AddNode(*graph, "Unsqueeze", {"_onx_range_dim1::Sq__1", "init7_s2_0_1__1"},
              {"_onx_range_dim1::Sq::UnSq0x1__1"});
      {
        NodeProto &n = AddNode(*graph, "Cast", {"_onx_range_dim1::Sq::UnSq0x1__1"},
                               {"_onx_range_dim1::Sq::UnSq0x1::C1__1"});
        AddAttribute<int64_t>(n, "to", INT64_C(1));
      }
      AddNode(*graph, "Reshape", {"_onx_range_dim1::Sq::UnSq0x1::C1__1", "init7_s3_0_-1_1__1"},
              {"_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1"});
      AddNode(*graph, "Mul", {"to_322", "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1"},
              {"_onx_mul_weights__1"});
      AddNode(*graph, "Cos", {"_onx_mul_weights__1"}, {"_onx_cos_mul_weights__1"});
      AddNode(*graph, "Sin", {"_onx_mul_weights__1"}, {"_onx_sin_mul_weights__1"});
      {
        NodeProto &n = AddNode(*graph, "Cast", {"_onx_cos_mul_weights__1"}, {"uoutput_0"});
        AddAttribute<int64_t>(n, "to", INT64_C(10));
      }
      {
        NodeProto &n = AddNode(*graph, "Cast", {"_onx_sin_mul_weights__1"}, {"uoutput_1"});
        AddAttribute<int64_t>(n, "to", INT64_C(10));
      }
      AddNode(*graph, "Squeeze", {"past_key_values_key_0::Shape2:3"}, {"A::Sq__2"});
      AddNode(*graph, "Squeeze", {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"},
              {"B::Sq__2"});
      AddNode(*graph, "Range", {"init7_s_0__2", "B::Sq__2", "init7_s_1__2"},
              {"_onx_range_init7_s_0__2"});
      AddNode(*graph, "Range", {"A::Sq__2", "B::Sq__2", "init7_s_1__2"}, {"_onx_range_A::Sq__2"});
      AddNode(*graph, "Unsqueeze", {"_onx_range_init7_s_0__2", "init7_s3_0_1_2__2"},
              {"_onx_range_init7_s_0::UnSq0x1x2__2"});
      AddNode(*graph, "Unsqueeze", {"_onx_range_A::Sq__2", "init7_s3_0_1_3__2"},
              {"_onx_range_A::Sq::UnSq0x1x3__2"});
      AddNode(*graph, "LessOrEqual",
              {"_onx_range_init7_s_0::UnSq0x1x2__2", "_onx_range_A::Sq::UnSq0x1x3__2"}, {"le_3"});
      AddNode(*graph, "Gather", {"lm_head.weight", "input_ids"}, {"embedding"});
      {
        NodeProto &n = AddNode(*graph, "Cast", {"attention_mask"}, {"to"});
        AddAttribute<int64_t>(n, "to", INT64_C(9));
      }
      {
        NodeProto &n = AddNode(*graph, "Shape", {"to"}, {"to::Shape-1:"});
        AddAttribute<int64_t>(n, "start", INT64_C(-1));
      }
      AddNode(*graph, "Squeeze", {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"},
              {"A::Sq__3"});
      AddNode(*graph, "Squeeze", {"past_key_values_value_2::Shape:1"}, {"B::Sq__3"});
      AddNode(*graph, "Range", {"init7_s_0__3", "A::Sq__3", "init7_s_1__3"},
              {"_onx_range_init7_s_0__3"});
      AddNode(*graph, "Range", {"init7_s_0__3", "B::Sq__3", "init7_s_1__3"},
              {"_onx_range_init7_s_02__3"});
      AddNode(*graph, "Unsqueeze", {"_onx_range_init7_s_0__3", "init7_s3_0_1_2__3"},
              {"_onx_range_init7_s_0::UnSq0x1x2__3"});
      AddNode(*graph, "Unsqueeze", {"_onx_range_init7_s_02__3", "init7_s3_1_2_3__3"},
              {"_onx_range_init7_s_02::UnSq1x2x3__3"});
      AddNode(*graph, "Mul", {"_onx_range_init7_s_02::UnSq1x2x3__3", "to::Shape-1:"},
              {"_onx_mul_range_init7_s_02::UnSq1x2x3__3"});
      AddNode(*graph, "Add",
              {"_onx_mul_range_init7_s_02::UnSq1x2x3__3", "_onx_range_init7_s_0::UnSq0x1x2__3"},
              {"_onx_add_unsqueeze_12"});
      AddNode(*graph, "Shape", {"_onx_add_unsqueeze_12"}, {"_onx_add_unsqueeze_12::Shape:"});
      AddNode(*graph, "Reshape", {"to", "init7_s1_-1"}, {"to::RSh-1"});
      AddNode(*graph, "Reshape", {"_onx_add_unsqueeze_12", "init7_s1_-1"},
              {"_onx_add_unsqueeze_12::RSh-1"});
      AddNode(*graph, "Gather", {"to::RSh-1", "_onx_add_unsqueeze_12::RSh-1"},
              {"_onx_gather_to::RSh-1"});
      AddNode(*graph, "Reshape", {"_onx_gather_to::RSh-1", "_onx_add_unsqueeze_12::Shape:"},
              {"index"});
      AddNode(*graph, "And", {"le_3", "index"}, {"and_2"});
      AddNode(*graph, "Unsqueeze", {"uoutput_0", "init7_s1_1"}, {"uunsqueeze_16"});
      {
        NodeProto &n =
            AddNode(*graph, "Concat", {"uunsqueeze_16", "uunsqueeze_16"}, {"unsqueeze_16"});
        AddAttribute<int64_t>(n, "axis", INT64_C(-1));
      }
      AddNode(*graph, "Unsqueeze", {"uoutput_1", "init7_s1_1"}, {"uunsqueeze_17"});
      {
        NodeProto &n =
            AddNode(*graph, "Concat", {"uunsqueeze_17", "uunsqueeze_17"}, {"unsqueeze_17"});
        AddAttribute<int64_t>(n, "axis", INT64_C(-1));
      }

      // ---- Transformer layers -------------------------------------------------
      // Each layer applies: pre-attention RMSNormalization, GQA Attention with
      // RoPE-rotated Q/K and per-head Q/K RMSNormalization, post-attention
      // RMSNormalization and a SwiGLU MLP.
      std::string layer_input = "embedding";
      for (int layer = 0; layer < 4; ++layer) {
        const std::string li = std::to_string(layer);
        const std::string weight_prefix = "p_model_layers_" + li; // projection weight prefix
        const std::string norm_prefix = "model.layers." + li;     // norm weight prefix
        const std::string layer_suffix = "layer_" + li;           // per-layer node name suffix
        auto make_layer_name = [&layer_suffix](const std::string &s) -> std::string {
          return layer_suffix + "_" + s;
        };

        // Pre-attention RMSNorm (input_layernorm).
        AddNode(*graph, "RMSNormalization", {layer_input, norm_prefix + ".input_layernorm.weight"},
                {make_layer_name("normed")});

        // Q projection + per-head q_norm + transpose to (batch, heads, seq, head).
        AddNode(*graph, "MatMul",
                {make_layer_name("normed"), weight_prefix + "_self_attn_q_proj_weight::T10"},
                {make_layer_name("q_mm")});
        AddNode(*graph, "Reshape", {make_layer_name("q_mm"), "init7_s4_0_0_16_128"},
                {make_layer_name("q_4d")});
        AddNode(*graph, "RMSNormalization",
                {make_layer_name("q_4d"), norm_prefix + ".self_attn.q_norm.weight"},
                {make_layer_name("q_normed")});
        {
          NodeProto &n =
              AddNode(*graph, "Transpose", {make_layer_name("q_normed")}, {make_layer_name("q_T")});
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }

        // K projection + per-head k_norm + transpose.
        AddNode(*graph, "MatMul",
                {make_layer_name("normed"), weight_prefix + "_self_attn_k_proj_weight::T10"},
                {make_layer_name("k_mm")});
        AddNode(*graph, "Reshape", {make_layer_name("k_mm"), "init7_s4_0_0_8_128"},
                {make_layer_name("k_4d")});
        AddNode(*graph, "RMSNormalization",
                {make_layer_name("k_4d"), norm_prefix + ".self_attn.k_norm.weight"},
                {make_layer_name("k_normed")});
        {
          NodeProto &n =
              AddNode(*graph, "Transpose", {make_layer_name("k_normed")}, {make_layer_name("k_T")});
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }

        // V projection + transpose.
        AddNode(*graph, "MatMul",
                {make_layer_name("normed"), weight_prefix + "_self_attn_v_proj_weight::T10"},
                {make_layer_name("v_mm")});
        AddNode(*graph, "Reshape", {make_layer_name("v_mm"), "init7_s4_0_0_8_128"},
                {make_layer_name("v_4d")});
        {
          NodeProto &n =
              AddNode(*graph, "Transpose", {make_layer_name("v_4d")}, {make_layer_name("v_T")});
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }

        // RoPE for Q.
        {
          NodeProto &n = AddNode(*graph, "Split", {make_layer_name("q_T")},
                                 {make_layer_name("q_half0"), make_layer_name("q_half1")});
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
          AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
        }
        AddNode(*graph, "Neg", {make_layer_name("q_half1")}, {make_layer_name("neg_q")});
        {
          NodeProto &n =
              AddNode(*graph, "Concat", {make_layer_name("neg_q"), make_layer_name("q_half0")},
                      {make_layer_name("q_rot")});
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
        }
        AddNode(*graph, "Mul", {make_layer_name("q_T"), "unsqueeze_16"},
                {make_layer_name("q_cos")});
        AddNode(*graph, "Mul", {make_layer_name("q_rot"), "unsqueeze_17"},
                {make_layer_name("q_sin")});
        AddNode(*graph, "Add", {make_layer_name("q_cos"), make_layer_name("q_sin")},
                {make_layer_name("q_rope")});

        // RoPE for K.
        {
          NodeProto &n = AddNode(*graph, "Split", {make_layer_name("k_T")},
                                 {make_layer_name("k_half0"), make_layer_name("k_half1")});
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
          AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
        }
        AddNode(*graph, "Neg", {make_layer_name("k_half1")}, {make_layer_name("neg_k")});
        {
          NodeProto &n =
              AddNode(*graph, "Concat", {make_layer_name("neg_k"), make_layer_name("k_half0")},
                      {make_layer_name("k_rot")});
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
        }
        AddNode(*graph, "Mul", {make_layer_name("k_T"), "unsqueeze_16"},
                {make_layer_name("k_cos")});
        AddNode(*graph, "Mul", {make_layer_name("k_rot"), "unsqueeze_17"},
                {make_layer_name("k_sin")});
        AddNode(*graph, "Add", {make_layer_name("k_cos"), make_layer_name("k_sin")},
                {make_layer_name("k_rope")});

        // Fused grouped-query attention with KV cache. The single ``Attention``
        // node replaces the KV-cache concat, head broadcast, scaled ``QK^T``,
        // masking, ``Softmax`` and ``attention @ V`` steps and emits the layer's
        // present key/value cache. GQA (16 Q heads / 8 KV heads) is inferred from
        // the rank-4 input shapes; the boolean ``and_2`` acts as the attention
        // mask.
        AddNode(*graph, "Attention",
                {make_layer_name("q_rope"), make_layer_name("k_rope"), make_layer_name("v_T"),
                 "and_2", "past_key_values_key_" + li, "past_key_values_value_" + li},
                {make_layer_name("attn_out"), "present_key_values_key_" + li,
                 "present_key_values_value_" + li});

        // Merge the heads back and project the attention output.
        {
          NodeProto &n = AddNode(*graph, "Transpose", {make_layer_name("attn_out")},
                                 {make_layer_name("attn_out_T")});
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }
        AddNode(*graph, "Reshape", {make_layer_name("attn_out_T"), "init7_s3_0_0_2048"},
                {make_layer_name("attn_2d")});
        AddNode(*graph, "MatMul",
                {make_layer_name("attn_2d"), weight_prefix + "_self_attn_o_proj_weight::T10"},
                {make_layer_name("attn_proj")});
        AddNode(*graph, "Add", {layer_input, make_layer_name("attn_proj")},
                {make_layer_name("resid_attn")});

        // Post-attention RMSNorm (post_attention_layernorm).
        AddNode(*graph, "RMSNormalization",
                {make_layer_name("resid_attn"), norm_prefix + ".post_attention_layernorm.weight"},
                {make_layer_name("mlp_in")});

        // SwiGLU MLP.
        AddNode(*graph, "MatMul",
                {make_layer_name("mlp_in"), weight_prefix + "_mlp_gate_proj_weight::T10"},
                {make_layer_name("gate")});
        AddNode(*graph, "Sigmoid", {make_layer_name("gate")}, {make_layer_name("gate_act")});
        AddNode(*graph, "Mul", {make_layer_name("gate"), make_layer_name("gate_act")},
                {make_layer_name("silu")});
        AddNode(*graph, "MatMul",
                {make_layer_name("mlp_in"), weight_prefix + "_mlp_up_proj_weight::T10"},
                {make_layer_name("up")});
        AddNode(*graph, "Mul", {make_layer_name("silu"), make_layer_name("up")},
                {make_layer_name("swiglu")});
        AddNode(*graph, "MatMul",
                {make_layer_name("swiglu"), weight_prefix + "_mlp_down_proj_weight::T10"},
                {make_layer_name("down")});
        AddNode(*graph, "Add", {make_layer_name("resid_attn"), make_layer_name("down")},
                {make_layer_name("out")});

        layer_input = make_layer_name("out");
      }

      // ---- Final RMSNorm (model.norm) + language-model head -------------------
      AddNode(*graph, "RMSNormalization", {layer_input, "model.norm.weight"}, {"final_normed"});
      AddNode(*graph, "MatMul", {"final_normed", "p_lm_head_weight::T10"}, {"output_0"});

      // ---- Intermediate value_info shapes -------------------------------------
      // Every intermediate result records its expected shape so shape inference is
      // validated on the full fused model. Concrete dims are matched exactly;
      // symbolic dims (batch_size / sequence_length / past_sequence_length and
      // arithmetic expressions thereof) are tolerated.
      AppendValueInfo(*graph->add_value_info(), "A::Sq__2", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "A::Sq__3", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "B::Sq__2", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "B::Sq__3", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(),
                      "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25",
                      DataType::INT64, {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_add_unsqueeze_12", DataType::INT64,
                      {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_add_unsqueeze_12::RSh-1", DataType::INT64,
                      {DimSpec("batch_size*(past_sequence_length+sequence_length)")});
      AppendValueInfo(*graph->add_value_info(), "_onx_add_unsqueeze_12::Shape:", DataType::INT64,
                      {DimSpec(INT64_C(4))});
      AppendValueInfo(*graph->add_value_info(), "_onx_cos_mul_weights__1", DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "_onx_gather_to::RSh-1", DataType::BOOL,
                      {DimSpec("batch_size*(past_sequence_length+sequence_length)")});
      AppendValueInfo(
          *graph->add_value_info(), "_onx_mul_range_init7_s_02::UnSq1x2x3__3", DataType::INT64,
          {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_mul_weights__1", DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_A::Sq::UnSq0x1x3__2", DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_A::Sq__2", DataType::INT64,
                      {DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1",
                      DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1::C1__1",
                      DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1__1", DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq__1", DataType::INT64,
                      {DimSpec("sequence_length")});
      AppendValueInfo(
          *graph->add_value_info(), "_onx_range_init7_s_02::UnSq1x2x3__3", DataType::INT64,
          {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_02__3", DataType::INT64,
                      {DimSpec("batch_size")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0::UnSq0x1x2__2",
                      DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0::UnSq0x1x2__3",
                      DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0__2", DataType::INT64,
                      {DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0__3", DataType::INT64,
                      {DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_sin_mul_weights__1", DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "and_2", DataType::BOOL,
                      {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "dim1::Sq__1", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "dim2::Sq__1", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "embedding", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_normed", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "index", DataType::BOOL,
                      {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "input_ids::Shape1:2", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "le_3", DataType::BOOL,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "past_key_values_key_0::Shape2:3", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "past_key_values_value_2::Shape:1", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "to", DataType::BOOL,
                      {DimSpec("batch_size"), DimSpec("total_sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "to::RSh-1", DataType::BOOL,
                      {DimSpec("batch_size*total_sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "to::Shape-1:", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "unsqueeze_16", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(128))});
      AppendValueInfo(*graph->add_value_info(), "unsqueeze_17", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(128))});
      AppendValueInfo(*graph->add_value_info(), "uoutput_0", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "uoutput_1", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "uunsqueeze_16", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "uunsqueeze_17", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(64))});

      // Per-transformer-layer intermediate shapes (identical across the 4 layers).
      // Notation: B=batch_size, S=sequence_length, P=past_sequence_length.
      // hidden_size=1024; Q: 16 heads x 128; KV: 8 heads x 128; MLP=3072.
      for (int vi_layer = 0; vi_layer < 4; ++vi_layer) {
        const std::string vls = "layer_" + std::to_string(vi_layer) + "_";
        const auto vln = [&vls](const char *s) { return vls + s; };

        AppendValueInfo(
            *graph->add_value_info(), vln("attn_2d"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
        AppendValueInfo(*graph->add_value_info(), vln("attn_out"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("attn_out_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("attn_proj"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("down"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("gate"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(
            *graph->add_value_info(), vln("gate_act"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(*graph->add_value_info(), vln("k_4d"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_cos"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_half0"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(*graph->add_value_info(), vln("k_half1"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(
            *graph->add_value_info(), vln("k_mm"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("k_normed"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_rope"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_rot"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_sin"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("mlp_in"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("neg_k"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(*graph->add_value_info(), vln("neg_q"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(
            *graph->add_value_info(), vln("normed"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("out"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("q_4d"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_cos"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_half0"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(*graph->add_value_info(), vln("q_half1"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(
            *graph->add_value_info(), vln("q_mm"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
        AppendValueInfo(*graph->add_value_info(), vln("q_normed"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_rope"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_rot"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_sin"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("resid_attn"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("silu"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(
            *graph->add_value_info(), vln("swiglu"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(
            *graph->add_value_info(), vln("up"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(*graph->add_value_info(), vln("v_4d"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("v_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("v_mm"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      }

      // ---- Graph inputs -------------------------------------------------------
      AppendValueInfo(*graph->add_input(), "input_ids", DataType::INT64,
                      {DimSpec("batch_size"), DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_input(), "attention_mask", DataType::INT64,
                      {DimSpec("batch_size"), DimSpec("total_sequence_length")});
      for (int layer = 0; layer < 4; ++layer) {
        const std::string li = std::to_string(layer);
        AppendValueInfo(*graph->add_input(), "past_key_values_key_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_input(), "past_key_values_value_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length"), DimSpec(INT64_C(128))});
      }

      // ---- Graph outputs ------------------------------------------------------
      AppendValueInfo(*graph->add_output(), "output_0", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(32000))});
      for (int layer = 0; layer < 4; ++layer) {
        const std::string li = std::to_string(layer);
        AppendValueInfo(*graph->add_output(), "present_key_values_key_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_output(), "present_key_values_value_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
      }

    } else {
      // ---- Constant-node initializers ----------------------------------------
      // Constant nodes from the original model are emitted as graph initializers.
      AddInitializer<int64_t>(*graph, "init7_s_1__1", {}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s2_0_1__1", {INT64_C(2)}, {INT64_C(0), INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_-1_1__1", {INT64_C(3)},
                              {INT64_C(0), INT64_C(-1), INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s_0__2", {}, {INT64_C(0)});
      AddInitializer<int64_t>(*graph, "init7_s_1__2", {}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_1_2__2", {INT64_C(3)},
                              {INT64_C(0), INT64_C(1), INT64_C(2)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_1_3__2", {INT64_C(3)},
                              {INT64_C(0), INT64_C(1), INT64_C(3)});
      AddInitializer<int64_t>(*graph, "init7_s_0__3", {}, {INT64_C(0)});
      AddInitializer<int64_t>(*graph, "init7_s_1__3", {}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_1_2__3", {INT64_C(3)},
                              {INT64_C(0), INT64_C(1), INT64_C(2)});
      AddInitializer<int64_t>(*graph, "init7_s3_1_2_3__3", {INT64_C(3)},
                              {INT64_C(1), INT64_C(2), INT64_C(3)});

      // Per-layer attention constant initializers (one set per transformer layer).
      for (int layer = 0; layer < 4; ++layer) {
        const std::string layer_suffix = "layer_" + std::to_string(layer);
        const std::string index_init_name = "init7_s1_2__" + layer_suffix;
        const std::string neg_inf_init_name = "init10_s1___" + layer_suffix;
        const std::string zero_init_name = "init10_s1_2__" + layer_suffix;
        AddInitializer<int64_t>(*graph, index_init_name.c_str(), {INT64_C(1)}, {INT64_C(2)});
        AddInitializer<uint16_t>(*graph, neg_inf_init_name.c_str(), {INT64_C(1)},
                                 {static_cast<uint16_t>(64512u)});
        AddInitializer<uint16_t>(*graph, zero_init_name.c_str(), {INT64_C(1)},
                                 {static_cast<uint16_t>(0u)});
      }

      // ---- Graph initializers -------------------------------------------------
      AddInitializer<int64_t>(*graph, "init7_s1_1", {INT64_C(1)}, {INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s1_-1", {INT64_C(1)}, {INT64_C(-1)});
      AddInitializer<float>(*graph, "init1_s_", {}, {2.0f});
      AddInitializer<uint16_t>(*graph, "init10_s1_", {INT64_C(1)}, {static_cast<uint16_t>(13506u)});
      AddInitializer<float>(*graph, "init1_s_2::RSh1", {INT64_C(1)}, {1e-06f});

      // Per-layer projection weight initializers.
      for (int layer = 0; layer < 4; ++layer) {
        const std::string weight_prefix = "p_model_layers_" + std::to_string(layer);
        const std::string q_proj = weight_prefix + "_self_attn_q_proj_weight::T10";
        const std::string k_proj = weight_prefix + "_self_attn_k_proj_weight::T10";
        const std::string v_proj = weight_prefix + "_self_attn_v_proj_weight::T10";
        const std::string o_proj = weight_prefix + "_self_attn_o_proj_weight::T10";
        const std::string gate_proj = weight_prefix + "_mlp_gate_proj_weight::T10";
        const std::string up_proj = weight_prefix + "_mlp_up_proj_weight::T10";
        const std::string down_proj = weight_prefix + "_mlp_down_proj_weight::T10";
        AddInitializer<uint16_t>(*graph, q_proj.c_str(), {INT64_C(1024), INT64_C(2048)}, {});
        AddInitializer<uint16_t>(*graph, k_proj.c_str(), {INT64_C(1024), INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, v_proj.c_str(), {INT64_C(1024), INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, o_proj.c_str(), {INT64_C(2048), INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, gate_proj.c_str(), {INT64_C(1024), INT64_C(3072)}, {});
        AddInitializer<uint16_t>(*graph, up_proj.c_str(), {INT64_C(1024), INT64_C(3072)}, {});
        AddInitializer<uint16_t>(*graph, down_proj.c_str(), {INT64_C(3072), INT64_C(1024)}, {});
      }

      AddInitializer<uint16_t>(*graph, "p_lm_head_weight::T10", {INT64_C(1024), INT64_C(32000)},
                               {});
      AddInitializer<float>(
          *graph, "to_322", {INT64_C(1), INT64_C(1), INT64_C(64)},
          {1.0f,           0.80566406f,    0.64941406f,    0.5234375f,     0.42163086f,
           0.33984375f,    0.27392578f,    0.22070312f,    0.17785645f,    0.14331055f,
           0.11547852f,    0.093078613f,   0.075012207f,   0.060424805f,   0.048706055f,
           0.039245605f,   0.031616211f,   0.025482178f,   0.02053833f,    0.016555786f,
           0.013336182f,   0.010749817f,   0.0086593628f,  0.0069770813f,  0.0056228638f,
           0.0045318604f,  0.0036525726f,  0.0029430389f,  0.0023708344f,  0.0019111633f,
           0.001540184f,   0.0012407303f,  0.0010004044f,  0.0008058548f,  0.00064945221f,
           0.00052309036f, 0.00042176247f, 0.00033974648f, 0.00027394295f, 0.00022065639f,
           0.00017786026f, 0.00014328957f, 0.0001154542f,  9.304285e-05f,  7.4982643e-05f,
           6.043911e-05f,  4.8696995e-05f, 3.9219856e-05f, 3.1650066e-05f, 2.5510788e-05f,
           2.0563602e-05f, 1.6570091e-05f, 1.335144e-05f,  1.0728836e-05f, 8.6426735e-06f,
           6.9737434e-06f, 5.6028366e-06f, 4.529953e-06f,  3.6358833e-06f, 2.9206276e-06f,
           2.3841858e-06f, 1.9073486e-06f, 1.5497208e-06f, 1.2516975e-06f});
      AddInitializer<int64_t>(*graph, "init7_s5_1_1_2_1_1", {INT64_C(5)},
                              {INT64_C(1), INT64_C(1), INT64_C(2), INT64_C(1), INT64_C(1)});
      AddInitializer<int64_t>(*graph, "init7_s4_0_0_16_128", {INT64_C(4)},
                              {INT64_C(0), INT64_C(0), INT64_C(16), INT64_C(128)});
      AddInitializer<int64_t>(*graph, "init7_s4_0_0_8_128", {INT64_C(4)},
                              {INT64_C(0), INT64_C(0), INT64_C(8), INT64_C(128)});
      AddInitializer<int64_t>(*graph, "init7_s4_0_16_-1_128", {INT64_C(4)},
                              {INT64_C(0), INT64_C(16), INT64_C(-1), INT64_C(128)});
      AddInitializer<int64_t>(*graph, "init7_s3_0_0_2048", {INT64_C(3)},
                              {INT64_C(0), INT64_C(0), INT64_C(2048)});

      // Per-layer norm weight initializers.
      for (int layer = 0; layer < 4; ++layer) {
        const std::string norm_prefix = "model.layers." + std::to_string(layer);
        const std::string q_norm = norm_prefix + ".self_attn.q_norm.weight";
        const std::string k_norm = norm_prefix + ".self_attn.k_norm.weight";
        const std::string input_ln = norm_prefix + ".input_layernorm.weight";
        const std::string post_ln = norm_prefix + ".post_attention_layernorm.weight";
        // All-same FP16 initializer (value=0x3C00).
        AddInitializer<uint16_t>(*graph, q_norm.c_str(), {INT64_C(128)},
                                 std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
        // All-same FP16 initializer (value=0x3C00).
        AddInitializer<uint16_t>(*graph, k_norm.c_str(), {INT64_C(128)},
                                 std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
        AddInitializer<uint16_t>(*graph, input_ln.c_str(), {INT64_C(1024)}, {});
        AddInitializer<uint16_t>(*graph, post_ln.c_str(), {INT64_C(1024)}, {});
      }

      AddInitializer<uint16_t>(*graph, "model.norm.weight", {INT64_C(1024)}, {});
      AddInitializer<uint16_t>(*graph, "lm_head.weight", {INT64_C(32000), INT64_C(1024)}, {});

      // Golden shape-inference / in-place-reuse metadata, attached to each node as
      // it is created and verified by BigModelsInplaceInfo.
      const auto tag = [](NodeProto &n, const char *node_tag, const std::string &inplace,
                          const std::string &release_after,
                          const std::string &release_after_shape_tag) {
        n.add_metadata(core::compute::kNodeTagMetadataKey, node_tag);
        if (!inplace.empty()) {
          n.add_metadata(core::compute::kInPlaceReuseMetadataKey, inplace);
        }
        if (!release_after.empty()) {
          n.add_metadata(core::compute::kReleaseAfterMetadataKey, release_after);
        }
        if (!release_after_shape_tag.empty()) {
          n.add_metadata(core::compute::kReleaseAfterShapeTagMetadataKey, release_after_shape_tag);
        }
      };

      // ---- Nodes --------------------------------------------------------------
      // Constant nodes have been promoted to initializers above.
      {
        NodeProto &n = AddNode(*graph, "Shape", {"input_ids"}, {"input_ids::Shape1:2"});
        tag(n, "shape", "", "", "");
        AddAttribute<int64_t>(n, "end", INT64_C(2));
        AddAttribute<int64_t>(n, "start", INT64_C(1));
      }
      {
        NodeProto &n = AddNode(*graph, "Shape", {"past_key_values_key_0"},
                               {"past_key_values_key_0::Shape2:3"});
        tag(n, "shape", "", "", "");
        AddAttribute<int64_t>(n, "end", INT64_C(3));
        AddAttribute<int64_t>(n, "start", INT64_C(2));
      }
      {
        NodeProto &n = AddNode(*graph, "Shape", {"past_key_values_value_2"},
                               {"past_key_values_value_2::Shape:1"});
        tag(n, "shape", "", "", "");
        AddAttribute<int64_t>(n, "end", INT64_C(1));
        AddAttribute<int64_t>(n, "start", INT64_C(0));
      }
      tag(AddNode(*graph, "Add", {"input_ids::Shape1:2", "past_key_values_key_0::Shape2:3"},
                  {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"}),
          "shape", "0:0:equal", "input_ids::Shape1:2", "input_ids::Shape1:2");
      tag(AddNode(*graph, "Squeeze", {"past_key_values_key_0::Shape2:3"}, {"dim1::Sq__1"}), "shape",
          "", "", "");
      tag(AddNode(*graph, "Squeeze",
                  {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"},
                  {"dim2::Sq__1"}),
          "shape", "", "", "");
      tag(AddNode(*graph, "Range", {"dim1::Sq__1", "dim2::Sq__1", "init7_s_1__1"},
                  {"_onx_range_dim1::Sq__1"}),
          "shape", "", "dim1::Sq__1;dim2::Sq__1", "dim1::Sq__1;dim2::Sq__1");
      tag(AddNode(*graph, "Unsqueeze", {"_onx_range_dim1::Sq__1", "init7_s2_0_1__1"},
                  {"_onx_range_dim1::Sq::UnSq0x1__1"}),
          "shape", "0:0:equal", "_onx_range_dim1::Sq__1", "_onx_range_dim1::Sq__1");
      {
        NodeProto &n = AddNode(*graph, "Cast", {"_onx_range_dim1::Sq::UnSq0x1__1"},
                               {"_onx_range_dim1::Sq::UnSq0x1::C1__1"});
        tag(n, "shape", "", "_onx_range_dim1::Sq::UnSq0x1__1", "_onx_range_dim1::Sq::UnSq0x1__1");
        AddAttribute<int64_t>(n, "to", INT64_C(1));
      }
      tag(AddNode(*graph, "Reshape", {"_onx_range_dim1::Sq::UnSq0x1::C1__1", "init7_s3_0_-1_1__1"},
                  {"_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1"}),
          "shape", "0:0:equal", "_onx_range_dim1::Sq::UnSq0x1::C1__1",
          "_onx_range_dim1::Sq::UnSq0x1::C1__1");
      tag(AddNode(*graph, "Mul", {"to_322", "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1"},
                  {"_onx_mul_weights__1"}),
          "weight", "", "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1",
          "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1");
      tag(AddNode(*graph, "Cos", {"_onx_mul_weights__1"}, {"_onx_cos_mul_weights__1"}), "weight",
          "", "", "");
      tag(AddNode(*graph, "Sin", {"_onx_mul_weights__1"}, {"_onx_sin_mul_weights__1"}), "weight",
          "0:0:equal", "_onx_mul_weights__1", "");
      {
        NodeProto &n = AddNode(*graph, "Cast", {"_onx_cos_mul_weights__1"}, {"uoutput_0"});
        tag(n, "weight", "", "_onx_cos_mul_weights__1", "");
        AddAttribute<int64_t>(n, "to", INT64_C(10));
      }
      {
        NodeProto &n = AddNode(*graph, "Cast", {"_onx_sin_mul_weights__1"}, {"uoutput_1"});
        tag(n, "weight", "", "_onx_sin_mul_weights__1", "");
        AddAttribute<int64_t>(n, "to", INT64_C(10));
      }
      tag(AddNode(*graph, "Squeeze", {"past_key_values_key_0::Shape2:3"}, {"A::Sq__2"}), "shape",
          "0:0:equal", "past_key_values_key_0::Shape2:3", "past_key_values_key_0::Shape2:3");
      tag(AddNode(*graph, "Squeeze",
                  {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"}, {"B::Sq__2"}),
          "shape", "", "", "");
      tag(AddNode(*graph, "Range", {"init7_s_0__2", "B::Sq__2", "init7_s_1__2"},
                  {"_onx_range_init7_s_0__2"}),
          "weight", "", "", "");
      tag(AddNode(*graph, "Range", {"A::Sq__2", "B::Sq__2", "init7_s_1__2"},
                  {"_onx_range_A::Sq__2"}),
          "shape", "", "A::Sq__2;B::Sq__2", "A::Sq__2;B::Sq__2");
      tag(AddNode(*graph, "Unsqueeze", {"_onx_range_init7_s_0__2", "init7_s3_0_1_2__2"},
                  {"_onx_range_init7_s_0::UnSq0x1x2__2"}),
          "weight", "0:0:equal", "_onx_range_init7_s_0__2", "");
      tag(AddNode(*graph, "Unsqueeze", {"_onx_range_A::Sq__2", "init7_s3_0_1_3__2"},
                  {"_onx_range_A::Sq::UnSq0x1x3__2"}),
          "shape", "0:0:equal", "_onx_range_A::Sq__2", "_onx_range_A::Sq__2");
      tag(AddNode(*graph, "LessOrEqual",
                  {"_onx_range_init7_s_0::UnSq0x1x2__2", "_onx_range_A::Sq::UnSq0x1x3__2"},
                  {"le_3"}),
          "weight", "", "_onx_range_init7_s_0::UnSq0x1x2__2;_onx_range_A::Sq::UnSq0x1x3__2",
          "_onx_range_A::Sq::UnSq0x1x3__2");
      tag(AddNode(*graph, "Gather", {"lm_head.weight", "input_ids"}, {"embedding"}), "weight", "",
          "", "");
      {
        NodeProto &n = AddNode(*graph, "Cast", {"attention_mask"}, {"to"});
        tag(n, "weight", "", "", "");
        AddAttribute<int64_t>(n, "to", INT64_C(9));
      }
      {
        NodeProto &n = AddNode(*graph, "Shape", {"to"}, {"to::Shape-1:"});
        tag(n, "shape", "", "", "");
        AddAttribute<int64_t>(n, "start", INT64_C(-1));
      }
      tag(AddNode(*graph, "Squeeze",
                  {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25"}, {"A::Sq__3"}),
          "shape", "0:0:equal", "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25",
          "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25");
      tag(AddNode(*graph, "Squeeze", {"past_key_values_value_2::Shape:1"}, {"B::Sq__3"}), "shape",
          "0:0:equal", "past_key_values_value_2::Shape:1", "past_key_values_value_2::Shape:1");
      tag(AddNode(*graph, "Range", {"init7_s_0__3", "A::Sq__3", "init7_s_1__3"},
                  {"_onx_range_init7_s_0__3"}),
          "weight", "", "A::Sq__3", "A::Sq__3");
      tag(AddNode(*graph, "Range", {"init7_s_0__3", "B::Sq__3", "init7_s_1__3"},
                  {"_onx_range_init7_s_02__3"}),
          "weight", "", "B::Sq__3", "B::Sq__3");
      tag(AddNode(*graph, "Unsqueeze", {"_onx_range_init7_s_0__3", "init7_s3_0_1_2__3"},
                  {"_onx_range_init7_s_0::UnSq0x1x2__3"}),
          "weight", "0:0:equal", "_onx_range_init7_s_0__3", "");
      tag(AddNode(*graph, "Unsqueeze", {"_onx_range_init7_s_02__3", "init7_s3_1_2_3__3"},
                  {"_onx_range_init7_s_02::UnSq1x2x3__3"}),
          "weight", "0:0:equal", "_onx_range_init7_s_02__3", "");
      tag(AddNode(*graph, "Mul", {"_onx_range_init7_s_02::UnSq1x2x3__3", "to::Shape-1:"},
                  {"_onx_mul_range_init7_s_02::UnSq1x2x3__3"}),
          "weight", "0:0:equal",
          "_onx_range_init7_s_02::UnSq1x2x3__3;to::Shape-1:", "to::Shape-1:");
      tag(AddNode(*graph, "Add",
                  {"_onx_mul_range_init7_s_02::UnSq1x2x3__3", "_onx_range_init7_s_0::UnSq0x1x2__3"},
                  {"_onx_add_unsqueeze_12"}),
          "weight", "",
          "_onx_mul_range_init7_s_02::UnSq1x2x3__3;_onx_range_init7_s_0::UnSq0x1x2__3", "");
      tag(AddNode(*graph, "Shape", {"_onx_add_unsqueeze_12"}, {"_onx_add_unsqueeze_12::Shape:"}),
          "shape", "", "", "");
      tag(AddNode(*graph, "Reshape", {"to", "init7_s1_-1"}, {"to::RSh-1"}), "weight", "0:0:equal",
          "to", "");
      tag(AddNode(*graph, "Reshape", {"_onx_add_unsqueeze_12", "init7_s1_-1"},
                  {"_onx_add_unsqueeze_12::RSh-1"}),
          "weight", "0:0:equal", "_onx_add_unsqueeze_12", "");
      tag(AddNode(*graph, "Gather", {"to::RSh-1", "_onx_add_unsqueeze_12::RSh-1"},
                  {"_onx_gather_to::RSh-1"}),
          "weight", "", "to::RSh-1;_onx_add_unsqueeze_12::RSh-1", "");
      tag(AddNode(*graph, "Reshape", {"_onx_gather_to::RSh-1", "_onx_add_unsqueeze_12::Shape:"},
                  {"index"}),
          "weight", "0:0:equal",
          "_onx_gather_to::RSh-1;_onx_add_unsqueeze_12::Shape:", "_onx_add_unsqueeze_12::Shape:");
      tag(AddNode(*graph, "And", {"le_3", "index"}, {"and_2"}), "weight", "", "le_3;index", "");
      tag(AddNode(*graph, "Unsqueeze", {"uoutput_0", "init7_s1_1"}, {"uunsqueeze_16"}), "weight",
          "0:0:equal", "uoutput_0", "");
      {
        NodeProto &n =
            AddNode(*graph, "Concat", {"uunsqueeze_16", "uunsqueeze_16"}, {"unsqueeze_16"});
        tag(n, "weight", "", "uunsqueeze_16", "");
        AddAttribute<int64_t>(n, "axis", INT64_C(-1));
      }
      tag(AddNode(*graph, "Unsqueeze", {"uoutput_1", "init7_s1_1"}, {"uunsqueeze_17"}), "weight",
          "0:0:equal", "uoutput_1", "");
      {
        NodeProto &n =
            AddNode(*graph, "Concat", {"uunsqueeze_17", "uunsqueeze_17"}, {"unsqueeze_17"});
        tag(n, "weight", "", "uunsqueeze_17", "");
        AddAttribute<int64_t>(n, "axis", INT64_C(-1));
      }

      // ---- Transformer layers -------------------------------------------------
      // Each layer applies: pre-attention RMSNorm, GQA attention with RoPE and
      // KV-cache, post-attention RMSNorm, SwiGLU MLP.
      std::string layer_input = "embedding";
      for (int layer = 0; layer < 4; ++layer) {
        const std::string li = std::to_string(layer);
        const std::string weight_prefix = "p_model_layers_" + li; // weight name prefix
        const std::string norm_prefix = "model.layers." + li;     // norm weight name prefix
        const std::string layer_suffix = "layer_" + li;           // per-layer node/init suffix
        // Helper: generate a unique intermediate node name for this layer.
        auto make_layer_name = [&layer_suffix](const std::string &s) -> std::string {
          return layer_suffix + "_" + s;
        };

        // Pre-attention RMSNorm (input_layernorm).
        {
          NodeProto &n = AddNode(*graph, "Cast", {layer_input}, {make_layer_name("f32")});
          tag(n, "weight", "", "", "");
          AddAttribute<int64_t>(n, "to", INT64_C(1));
        }
        tag(AddNode(*graph, "Pow", {make_layer_name("f32"), "init1_s_"},
                    {make_layer_name("pow_pre")}),
            "weight", "", "", "");
        {
          NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_pre"), "init7_s1_-1"},
                                 {make_layer_name("mean_pre")});
          tag(n, "weight", "", "layer_" + li + "_pow_pre", "");
          AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
        }
        tag(AddNode(*graph, "Add", {make_layer_name("mean_pre"), "init1_s_2::RSh1"},
                    {make_layer_name("add_pre")}),
            "weight", "0:0:equal", "layer_" + li + "_mean_pre", "");
        tag(AddNode(*graph, "Sqrt", {make_layer_name("add_pre")}, {make_layer_name("sqrt_pre")}),
            "weight", "0:0:equal", "layer_" + li + "_add_pre", "");
        tag(AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_pre")},
                    {make_layer_name("rsqrt_pre")}),
            "weight", "0:0:equal", "layer_" + li + "_sqrt_pre", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("f32"), make_layer_name("rsqrt_pre")},
                    {make_layer_name("mul_pre")}),
            "weight", "0:0:equal", "layer_" + li + "_f32;layer_" + li + "_rsqrt_pre", "");
        {
          NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("mul_pre")},
                                 {make_layer_name("normed_half")});
          tag(n, "weight", "", "layer_" + li + "_mul_pre", "");
          AddAttribute<int64_t>(n, "to", INT64_C(10));
        }
        tag(AddNode(*graph, "Mul",
                    {norm_prefix + ".input_layernorm.weight", make_layer_name("normed_half")},
                    {make_layer_name("normed")}),
            "weight", "0:1:equal", "layer_" + li + "_normed_half", "");

        // Q projection + q_norm + transpose.
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("normed"), weight_prefix + "_self_attn_q_proj_weight::T10"},
                    {make_layer_name("q_mm")}),
            "weight", "", "", "");
        {
          NodeProto &n =
              AddNode(*graph, "Cast", {make_layer_name("q_mm")}, {make_layer_name("q_f32")});
          tag(n, "weight", "", "layer_" + li + "_q_mm", "");
          AddAttribute<int64_t>(n, "to", INT64_C(1));
        }
        tag(AddNode(*graph, "Reshape", {make_layer_name("q_f32"), "init7_s4_0_0_16_128"},
                    {make_layer_name("q_4d")}),
            "weight", "0:0:equal", "layer_" + li + "_q_f32", "");
        tag(AddNode(*graph, "Pow", {make_layer_name("q_4d"), "init1_s_"},
                    {make_layer_name("pow_q")}),
            "weight", "", "", "");
        {
          NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_q"), "init7_s1_-1"},
                                 {make_layer_name("mean_q")});
          tag(n, "weight", "", "layer_" + li + "_pow_q", "");
          AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
        }
        tag(AddNode(*graph, "Add", {make_layer_name("mean_q"), "init1_s_2::RSh1"},
                    {make_layer_name("add_q")}),
            "weight", "0:0:equal", "layer_" + li + "_mean_q", "");
        tag(AddNode(*graph, "Sqrt", {make_layer_name("add_q")}, {make_layer_name("sqrt_q")}),
            "weight", "0:0:equal", "layer_" + li + "_add_q", "");
        tag(AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_q")},
                    {make_layer_name("rsqrt_q")}),
            "weight", "0:0:equal", "layer_" + li + "_sqrt_q", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("q_4d"), make_layer_name("rsqrt_q")},
                    {make_layer_name("mul_q")}),
            "weight", "0:0:equal", "layer_" + li + "_q_4d;layer_" + li + "_rsqrt_q", "");
        {
          NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("mul_q")},
                                 {make_layer_name("q_normed_half")});
          tag(n, "weight", "", "layer_" + li + "_mul_q", "");
          AddAttribute<int64_t>(n, "to", INT64_C(10));
        }
        tag(AddNode(*graph, "Mul",
                    {norm_prefix + ".self_attn.q_norm.weight", make_layer_name("q_normed_half")},
                    {make_layer_name("q_normed")}),
            "weight", "0:1:equal", "layer_" + li + "_q_normed_half", "");
        {
          NodeProto &n =
              AddNode(*graph, "Transpose", {make_layer_name("q_normed")}, {make_layer_name("q_T")});
          tag(n, "weight", "0:0:equal", "layer_" + li + "_q_normed", "");
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }

        // K projection + k_norm + transpose.
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("normed"), weight_prefix + "_self_attn_k_proj_weight::T10"},
                    {make_layer_name("k_mm")}),
            "weight", "", "", "");
        {
          NodeProto &n =
              AddNode(*graph, "Cast", {make_layer_name("k_mm")}, {make_layer_name("k_f32")});
          tag(n, "weight", "", "layer_" + li + "_k_mm", "");
          AddAttribute<int64_t>(n, "to", INT64_C(1));
        }
        tag(AddNode(*graph, "Reshape", {make_layer_name("k_f32"), "init7_s4_0_0_8_128"},
                    {make_layer_name("k_4d")}),
            "weight", "0:0:equal", "layer_" + li + "_k_f32", "");
        tag(AddNode(*graph, "Pow", {make_layer_name("k_4d"), "init1_s_"},
                    {make_layer_name("pow_k")}),
            "weight", "", "", "");
        {
          NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_k"), "init7_s1_-1"},
                                 {make_layer_name("mean_k")});
          tag(n, "weight", "", "layer_" + li + "_pow_k", "");
          AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
        }
        tag(AddNode(*graph, "Add", {make_layer_name("mean_k"), "init1_s_2::RSh1"},
                    {make_layer_name("add_k")}),
            "weight", "0:0:equal", "layer_" + li + "_mean_k", "");
        tag(AddNode(*graph, "Sqrt", {make_layer_name("add_k")}, {make_layer_name("sqrt_k")}),
            "weight", "0:0:equal", "layer_" + li + "_add_k", "");
        tag(AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_k")},
                    {make_layer_name("rsqrt_k")}),
            "weight", "0:0:equal", "layer_" + li + "_sqrt_k", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("k_4d"), make_layer_name("rsqrt_k")},
                    {make_layer_name("mul_k")}),
            "weight", "0:0:equal", "layer_" + li + "_k_4d;layer_" + li + "_rsqrt_k", "");
        {
          NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("mul_k")},
                                 {make_layer_name("k_normed_half")});
          tag(n, "weight", "", "layer_" + li + "_mul_k", "");
          AddAttribute<int64_t>(n, "to", INT64_C(10));
        }
        tag(AddNode(*graph, "Mul",
                    {norm_prefix + ".self_attn.k_norm.weight", make_layer_name("k_normed_half")},
                    {make_layer_name("k_normed")}),
            "weight", "0:1:equal", "layer_" + li + "_k_normed_half", "");
        {
          NodeProto &n =
              AddNode(*graph, "Transpose", {make_layer_name("k_normed")}, {make_layer_name("k_T")});
          tag(n, "weight", "0:0:equal", "layer_" + li + "_k_normed", "");
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }

        // V projection + transpose.
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("normed"), weight_prefix + "_self_attn_v_proj_weight::T10"},
                    {make_layer_name("v_mm")}),
            "weight", "0:0:equal", "layer_" + li + "_normed", "");
        tag(AddNode(*graph, "Reshape", {make_layer_name("v_mm"), "init7_s4_0_0_8_128"},
                    {make_layer_name("v_4d")}),
            "weight", "0:0:equal", "layer_" + li + "_v_mm", "");
        {
          NodeProto &n =
              AddNode(*graph, "Transpose", {make_layer_name("v_4d")}, {make_layer_name("v_T")});
          tag(n, "weight", "0:0:equal", "layer_" + li + "_v_4d", "");
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }

        // RoPE for Q.
        {
          NodeProto &n = AddNode(*graph, "Split", {make_layer_name("q_T")},
                                 {make_layer_name("q_half0"), make_layer_name("q_half1")});
          tag(n, "weight", "", "", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
          AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
        }
        tag(AddNode(*graph, "Neg", {make_layer_name("q_half1")}, {make_layer_name("neg_q")}),
            "weight", "0:0:equal", "layer_" + li + "_q_half1", "");
        {
          NodeProto &n =
              AddNode(*graph, "Concat", {make_layer_name("neg_q"), make_layer_name("q_half0")},
                      {make_layer_name("q_rot")});
          tag(n, "weight", "", "layer_" + li + "_neg_q;layer_" + li + "_q_half0", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
        }
        tag(AddNode(*graph, "Mul", {make_layer_name("q_T"), "unsqueeze_16"},
                    {make_layer_name("q_cos")}),
            "weight", "0:0:equal", "layer_" + li + "_q_T", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("q_rot"), "unsqueeze_17"},
                    {make_layer_name("q_sin")}),
            "weight", "0:0:equal", "layer_" + li + "_q_rot", "");
        tag(AddNode(*graph, "Add", {make_layer_name("q_cos"), make_layer_name("q_sin")},
                    {make_layer_name("q_rope")}),
            "weight", "0:0:equal", "layer_" + li + "_q_cos;layer_" + li + "_q_sin", "");

        // RoPE for K.
        {
          NodeProto &n = AddNode(*graph, "Split", {make_layer_name("k_T")},
                                 {make_layer_name("k_half0"), make_layer_name("k_half1")});
          tag(n, "weight", "", "", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
          AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
        }
        tag(AddNode(*graph, "Neg", {make_layer_name("k_half1")}, {make_layer_name("neg_k")}),
            "weight", "0:0:equal", "layer_" + li + "_k_half1", "");
        {
          NodeProto &n =
              AddNode(*graph, "Concat", {make_layer_name("neg_k"), make_layer_name("k_half0")},
                      {make_layer_name("k_rot")});
          tag(n, "weight", "", "layer_" + li + "_neg_k;layer_" + li + "_k_half0", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
        }
        // In the last layer the shared RoPE cos/sin tables (unsqueeze_16/17) reach
        // their last use here, so they are additionally released after these nodes.
        tag(AddNode(*graph, "Mul", {make_layer_name("k_T"), "unsqueeze_16"},
                    {make_layer_name("k_cos")}),
            "weight", "0:0:equal", "layer_" + li + "_k_T" + (layer == 3 ? ";unsqueeze_16" : ""),
            "");
        tag(AddNode(*graph, "Mul", {make_layer_name("k_rot"), "unsqueeze_17"},
                    {make_layer_name("k_sin")}),
            "weight", "0:0:equal", "layer_" + li + "_k_rot" + (layer == 3 ? ";unsqueeze_17" : ""),
            "");
        tag(AddNode(*graph, "Add", {make_layer_name("k_cos"), make_layer_name("k_sin")},
                    {make_layer_name("k_rope")}),
            "weight", "0:0:equal", "layer_" + li + "_k_cos;layer_" + li + "_k_sin", "");

        // KV-cache concatenation (produces the layer's present key/value outputs).
        {
          NodeProto &n =
              AddNode(*graph, "Concat", {"past_key_values_key_" + li, make_layer_name("k_rope")},
                      {"present_key_values_key_" + li});
          tag(n, "weight", "", "layer_" + li + "_k_rope", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-2));
        }
        {
          NodeProto &n =
              AddNode(*graph, "Concat", {"past_key_values_value_" + li, make_layer_name("v_T")},
                      {"present_key_values_value_" + li});
          tag(n, "weight", "", "layer_" + li + "_v_T", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-2));
        }

        // GQA: expand KV to Q-head count, then compute scaled dot-product attention.
        tag(AddNode(*graph, "Mul", {"present_key_values_key_" + li, "init10_s1_"},
                    {make_layer_name("scaled_k")}),
            "weight", "", "", "");
        tag(AddNode(*graph, "Unsqueeze",
                    {make_layer_name("scaled_k"), "init7_s1_2__" + layer_suffix},
                    {make_layer_name("scaled_k_unsq")}),
            "weight", "0:0:equal", "layer_" + li + "_scaled_k", "");
        tag(AddNode(*graph, "Unsqueeze",
                    {"present_key_values_value_" + li, "init7_s1_2__" + layer_suffix},
                    {make_layer_name("v_unsq")}),
            "weight", "", "", "");
        tag(AddNode(*graph, "Expand", {make_layer_name("scaled_k_unsq"), "init7_s5_1_1_2_1_1"},
                    {make_layer_name("scaled_k_exp")}),
            "weight", "", "layer_" + li + "_scaled_k_unsq", "");
        tag(AddNode(*graph, "Expand", {make_layer_name("v_unsq"), "init7_s5_1_1_2_1_1"},
                    {make_layer_name("v_exp")}),
            "weight", "", "layer_" + li + "_v_unsq", "");
        tag(AddNode(*graph, "Reshape", {make_layer_name("scaled_k_exp"), "init7_s4_0_16_-1_128"},
                    {make_layer_name("k_gqa")}),
            "weight", "0:0:equal", "layer_" + li + "_scaled_k_exp", "");
        tag(AddNode(*graph, "Reshape", {make_layer_name("v_exp"), "init7_s4_0_16_-1_128"},
                    {make_layer_name("v_gqa")}),
            "weight", "0:0:equal", "layer_" + li + "_v_exp", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("q_rope"), "init10_s1_"},
                    {make_layer_name("scaled_q")}),
            "weight", "0:0:equal", "layer_" + li + "_q_rope", "");
        {
          NodeProto &n = AddNode(*graph, "Transpose", {make_layer_name("k_gqa")},
                                 {make_layer_name("k_gqa_T")});
          tag(n, "weight", "0:0:equal", "layer_" + li + "_k_gqa", "");
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(1), INT64_C(3), INT64_C(2)});
        }
        tag(AddNode(*graph, "MatMul", {make_layer_name("scaled_q"), make_layer_name("k_gqa_T")},
                    {make_layer_name("attn_scores")}),
            "weight", "", "layer_" + li + "_scaled_q;layer_" + li + "_k_gqa_T", "");
        // In the last layer the shared causal mask (and_2) reaches its last use
        // here, so it is additionally released after this node.
        tag(AddNode(*graph, "Where",
                    {"and_2", make_layer_name("attn_scores"), "init10_s1___" + layer_suffix},
                    {make_layer_name("masked")}),
            "weight", "0:1:equal",
            std::string(layer == 3 ? "and_2;" : "") + "layer_" + li + "_attn_scores", "");
        {
          NodeProto &n =
              AddNode(*graph, "Softmax", {make_layer_name("masked")}, {make_layer_name("softmax")});
          tag(n, "weight", "0:0:equal", "layer_" + li + "_masked", "");
          AddAttribute<int64_t>(n, "axis", INT64_C(-1));
        }
        tag(AddNode(*graph, "IsNaN", {make_layer_name("softmax")}, {make_layer_name("is_nan")}),
            "weight", "", "", "");
        tag(AddNode(*graph, "Where",
                    {make_layer_name("is_nan"), "init10_s1_2__" + layer_suffix,
                     make_layer_name("softmax")},
                    {make_layer_name("attn_w")}),
            "weight", "0:2:equal", "layer_" + li + "_is_nan;layer_" + li + "_softmax", "");
        tag(AddNode(*graph, "MatMul", {make_layer_name("attn_w"), make_layer_name("v_gqa")},
                    {make_layer_name("attn_out")}),
            "weight", "", "layer_" + li + "_attn_w;layer_" + li + "_v_gqa", "");
        {
          NodeProto &n = AddNode(*graph, "Transpose", {make_layer_name("attn_out")},
                                 {make_layer_name("attn_out_T")});
          tag(n, "weight", "0:0:equal", "layer_" + li + "_attn_out", "");
          AddAttribute<std::vector<int64_t>>(n, "perm",
                                             {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
        }
        tag(AddNode(*graph, "Reshape", {make_layer_name("attn_out_T"), "init7_s3_0_0_2048"},
                    {make_layer_name("attn_2d")}),
            "weight", "0:0:equal", "layer_" + li + "_attn_out_T", "");
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("attn_2d"), weight_prefix + "_self_attn_o_proj_weight::T10"},
                    {make_layer_name("attn_proj")}),
            "weight", "", "layer_" + li + "_attn_2d", "");
        tag(AddNode(*graph, "Add", {layer_input, make_layer_name("attn_proj")},
                    {make_layer_name("resid_attn")}),
            "weight", "0:0:equal", layer_input + ";" + "layer_" + li + "_attn_proj", "");

        // Post-attention RMSNorm (post_attention_layernorm).
        {
          NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("resid_attn")},
                                 {make_layer_name("post_f32")});
          tag(n, "weight", "", "", "");
          AddAttribute<int64_t>(n, "to", INT64_C(1));
        }
        tag(AddNode(*graph, "Pow", {make_layer_name("post_f32"), "init1_s_"},
                    {make_layer_name("pow_post")}),
            "weight", "", "", "");
        {
          NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_post"), "init7_s1_-1"},
                                 {make_layer_name("mean_post")});
          tag(n, "weight", "", "layer_" + li + "_pow_post", "");
          AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
        }
        tag(AddNode(*graph, "Add", {make_layer_name("mean_post"), "init1_s_2::RSh1"},
                    {make_layer_name("add_post")}),
            "weight", "0:0:equal", "layer_" + li + "_mean_post", "");
        tag(AddNode(*graph, "Sqrt", {make_layer_name("add_post")}, {make_layer_name("sqrt_post")}),
            "weight", "0:0:equal", "layer_" + li + "_add_post", "");
        tag(AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_post")},
                    {make_layer_name("rsqrt_post")}),
            "weight", "0:0:equal", "layer_" + li + "_sqrt_post", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("post_f32"), make_layer_name("rsqrt_post")},
                    {make_layer_name("mul_post")}),
            "weight", "0:0:equal", "layer_" + li + "_post_f32;layer_" + li + "_rsqrt_post", "");
        {
          NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("mul_post")},
                                 {make_layer_name("post_half")});
          tag(n, "weight", "", "layer_" + li + "_mul_post", "");
          AddAttribute<int64_t>(n, "to", INT64_C(10));
        }
        tag(AddNode(
                *graph, "Mul",
                {norm_prefix + ".post_attention_layernorm.weight", make_layer_name("post_half")},
                {make_layer_name("mlp_in")}),
            "weight", "0:1:equal", "layer_" + li + "_post_half", "");

        // SwiGLU MLP.
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("mlp_in"), weight_prefix + "_mlp_gate_proj_weight::T10"},
                    {make_layer_name("gate")}),
            "weight", "", "", "");
        tag(AddNode(*graph, "Sigmoid", {make_layer_name("gate")}, {make_layer_name("gate_act")}),
            "weight", "", "", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("gate"), make_layer_name("gate_act")},
                    {make_layer_name("silu")}),
            "weight", "0:0:equal", "layer_" + li + "_gate;layer_" + li + "_gate_act", "");
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("mlp_in"), weight_prefix + "_mlp_up_proj_weight::T10"},
                    {make_layer_name("up")}),
            "weight", "", "layer_" + li + "_mlp_in", "");
        tag(AddNode(*graph, "Mul", {make_layer_name("silu"), make_layer_name("up")},
                    {make_layer_name("swiglu")}),
            "weight", "0:0:equal", "layer_" + li + "_silu;layer_" + li + "_up", "");
        tag(AddNode(*graph, "MatMul",
                    {make_layer_name("swiglu"), weight_prefix + "_mlp_down_proj_weight::T10"},
                    {make_layer_name("down")}),
            "weight", "", "layer_" + li + "_swiglu", "");
        tag(AddNode(*graph, "Add", {make_layer_name("resid_attn"), make_layer_name("down")},
                    {make_layer_name("out")}),
            "weight", "0:0:equal", "layer_" + li + "_resid_attn;layer_" + li + "_down", "");

        layer_input = make_layer_name("out");
      }

      // ---- Final RMSNorm (model.norm) + language-model head -------------------
      {
        NodeProto &n = AddNode(*graph, "Cast", {layer_input}, {"final_f32"});
        tag(n, "weight", "", "layer_3_out", "");
        AddAttribute<int64_t>(n, "to", INT64_C(1));
      }
      tag(AddNode(*graph, "Pow", {"final_f32", "init1_s_"}, {"final_pow"}), "weight", "", "", "");
      {
        NodeProto &n = AddNode(*graph, "ReduceMean", {"final_pow", "init7_s1_-1"}, {"final_mean"});
        tag(n, "weight", "", "final_pow", "");
        AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
      }
      tag(AddNode(*graph, "Add", {"final_mean", "init1_s_2::RSh1"}, {"final_add"}), "weight",
          "0:0:equal", "final_mean", "");
      tag(AddNode(*graph, "Sqrt", {"final_add"}, {"final_sqrt"}), "weight", "0:0:equal",
          "final_add", "");
      tag(AddNode(*graph, "Reciprocal", {"final_sqrt"}, {"final_rsqrt"}), "weight", "0:0:equal",
          "final_sqrt", "");
      tag(AddNode(*graph, "Mul", {"final_f32", "final_rsqrt"}, {"final_mul"}), "weight",
          "0:0:equal", "final_f32;final_rsqrt", "");
      {
        NodeProto &n = AddNode(*graph, "Cast", {"final_mul"}, {"final_half"});
        tag(n, "weight", "", "final_mul", "");
        AddAttribute<int64_t>(n, "to", INT64_C(10));
      }
      tag(AddNode(*graph, "Mul", {"model.norm.weight", "final_half"}, {"final_normed"}), "weight",
          "0:1:equal", "final_half", "");
      tag(AddNode(*graph, "MatMul", {"final_normed", "p_lm_head_weight::T10"}, {"output_0"}),
          "weight", "", "final_normed", "");

      // ---- Intermediate value_info shapes -------------------------------------
      // Every intermediate result produced by the graph records its expected
      // shape so shape inference is validated on the full model: each entry is
      // stripped before inference runs and must be recovered with the same
      // elem_type, the same rank and identical concrete dims (symbolic dims such
      // as batch_size / sequence_length / past_sequence_length /
      // total_sequence_length, and arithmetic expressions thereof, are tolerated).
      // elem_type: FLOAT=float32 RMSNorm accumulators, FLOAT16=model activations,
      // INT64=shape / index math, BOOL=attention masks.
      //
      // Values computed once before the transformer stack (RoPE tables, causal
      // mask, token embedding) and the trailing final-RMSNorm / LM-head values.
      AppendValueInfo(*graph->add_value_info(), "A::Sq__2", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "A::Sq__3", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "B::Sq__2", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "B::Sq__3", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(),
                      "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25",
                      DataType::INT64, {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_add_unsqueeze_12", DataType::INT64,
                      {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_add_unsqueeze_12::RSh-1", DataType::INT64,
                      {DimSpec("batch_size*(past_sequence_length+sequence_length)")});
      AppendValueInfo(*graph->add_value_info(), "_onx_add_unsqueeze_12::Shape:", DataType::INT64,
                      {DimSpec(INT64_C(4))});
      AppendValueInfo(*graph->add_value_info(), "_onx_cos_mul_weights__1", DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "_onx_gather_to::RSh-1", DataType::BOOL,
                      {DimSpec("batch_size*(past_sequence_length+sequence_length)")});
      AppendValueInfo(
          *graph->add_value_info(), "_onx_mul_range_init7_s_02::UnSq1x2x3__3", DataType::INT64,
          {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_mul_weights__1", DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_A::Sq::UnSq0x1x3__2", DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_A::Sq__2", DataType::INT64,
                      {DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1",
                      DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1::C1__1",
                      DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1__1", DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq__1", DataType::INT64,
                      {DimSpec("sequence_length")});
      AppendValueInfo(
          *graph->add_value_info(), "_onx_range_init7_s_02::UnSq1x2x3__3", DataType::INT64,
          {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_02__3", DataType::INT64,
                      {DimSpec("batch_size")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0::UnSq0x1x2__2",
                      DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0::UnSq0x1x2__3",
                      DataType::INT64,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0__2", DataType::INT64,
                      {DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0__3", DataType::INT64,
                      {DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "_onx_sin_mul_weights__1", DataType::FLOAT,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "and_2", DataType::BOOL,
                      {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "dim1::Sq__1", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "dim2::Sq__1", DataType::INT64, {});
      AppendValueInfo(*graph->add_value_info(), "embedding", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_add", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "final_f32", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_half", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_mean", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "final_mul", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_normed", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_pow", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
      AppendValueInfo(*graph->add_value_info(), "final_rsqrt", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "final_sqrt", DataType::FLOAT,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "index", DataType::BOOL,
                      {DimSpec("batch_size"), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "input_ids::Shape1:2", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "le_3", DataType::BOOL,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec("past_sequence_length+sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "past_key_values_key_0::Shape2:3", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "past_key_values_value_2::Shape:1", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "to", DataType::BOOL,
                      {DimSpec("batch_size"), DimSpec("total_sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "to::RSh-1", DataType::BOOL,
                      {DimSpec("batch_size*total_sequence_length")});
      AppendValueInfo(*graph->add_value_info(), "to::Shape-1:", DataType::INT64,
                      {DimSpec(INT64_C(1))});
      AppendValueInfo(*graph->add_value_info(), "unsqueeze_16", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(128))});
      AppendValueInfo(*graph->add_value_info(), "unsqueeze_17", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(128))});
      AppendValueInfo(*graph->add_value_info(), "uoutput_0", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "uoutput_1", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "uunsqueeze_16", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(64))});
      AppendValueInfo(*graph->add_value_info(), "uunsqueeze_17", DataType::FLOAT16,
                      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"),
                       DimSpec(INT64_C(64))});

      // Per-transformer-layer intermediate shapes (identical across the 4 layers).
      // Notation: B=batch_size, S=sequence_length, P=past_sequence_length.
      // hidden_size=1024; Q: 16 heads x 128; KV: 8 heads x 128; MLP=3072.
      for (int vi_layer = 0; vi_layer < 4; ++vi_layer) {
        const std::string vls = "layer_" + std::to_string(vi_layer) + "_";
        const auto vln = [&vls](const char *s) { return vls + s; };

        AppendValueInfo(*graph->add_value_info(), vln("add_k"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("add_post"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("add_pre"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("add_q"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(
            *graph->add_value_info(), vln("attn_2d"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
        AppendValueInfo(*graph->add_value_info(), vln("attn_out"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("attn_out_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("attn_proj"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("attn_scores"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec("past_sequence_length+sequence_length")});
        AppendValueInfo(*graph->add_value_info(), vln("attn_w"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec("past_sequence_length+sequence_length")});
        AppendValueInfo(
            *graph->add_value_info(), vln("down"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("f32"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("gate"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(
            *graph->add_value_info(), vln("gate_act"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(*graph->add_value_info(), vln("is_nan"), DataType::BOOL,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec("past_sequence_length+sequence_length")});
        AppendValueInfo(*graph->add_value_info(), vln("k_4d"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_cos"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("k_f32"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("k_gqa"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_gqa_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec(INT64_C(128)),
                         DimSpec("past_sequence_length+sequence_length")});
        AppendValueInfo(*graph->add_value_info(), vln("k_half0"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(*graph->add_value_info(), vln("k_half1"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(
            *graph->add_value_info(), vln("k_mm"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("k_normed"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_normed_half"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_rope"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_rot"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("k_sin"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("masked"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec("past_sequence_length+sequence_length")});
        AppendValueInfo(*graph->add_value_info(), vln("mean_k"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("mean_post"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("mean_pre"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("mean_q"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(
            *graph->add_value_info(), vln("mlp_in"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("mul_k"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("mul_post"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("mul_pre"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("mul_q"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("neg_k"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(*graph->add_value_info(), vln("neg_q"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(
            *graph->add_value_info(), vln("normed"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("normed_half"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("out"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("post_f32"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("post_half"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("pow_k"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("pow_post"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(
            *graph->add_value_info(), vln("pow_pre"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("pow_q"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_4d"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_cos"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("q_f32"), DataType::FLOAT,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
        AppendValueInfo(*graph->add_value_info(), vln("q_half0"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(*graph->add_value_info(), vln("q_half1"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(64))});
        AppendValueInfo(
            *graph->add_value_info(), vln("q_mm"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
        AppendValueInfo(*graph->add_value_info(), vln("q_normed"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_normed_half"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_rope"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_rot"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("q_sin"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("resid_attn"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("rsqrt_k"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("rsqrt_post"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("rsqrt_pre"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("rsqrt_q"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("scaled_k"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("scaled_k_exp"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec(INT64_C(2)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("scaled_k_unsq"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec(INT64_C(1)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("scaled_q"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("silu"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(*graph->add_value_info(), vln("softmax"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                         DimSpec("past_sequence_length+sequence_length")});
        AppendValueInfo(*graph->add_value_info(), vln("sqrt_k"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("sqrt_post"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("sqrt_pre"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
        AppendValueInfo(*graph->add_value_info(), vln("sqrt_q"), DataType::FLOAT,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                         DimSpec(INT64_C(1))});
        AppendValueInfo(
            *graph->add_value_info(), vln("swiglu"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(
            *graph->add_value_info(), vln("up"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
        AppendValueInfo(*graph->add_value_info(), vln("v_4d"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("v_T"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("sequence_length"),
                         DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("v_exp"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec(INT64_C(2)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_value_info(), vln("v_gqa"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(16)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(
            *graph->add_value_info(), vln("v_mm"), DataType::FLOAT16,
            {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
        AppendValueInfo(*graph->add_value_info(), vln("v_unsq"), DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec(INT64_C(1)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
      }

      // ---- Graph inputs -------------------------------------------------------
      AppendValueInfo(*graph->add_input(), "input_ids", DataType::INT64,
                      {DimSpec("batch_size"), DimSpec("sequence_length")});
      AppendValueInfo(*graph->add_input(), "attention_mask", DataType::INT64,
                      {DimSpec("batch_size"), DimSpec("total_sequence_length")});
      for (int layer = 0; layer < 4; ++layer) {
        const std::string li = std::to_string(layer);
        AppendValueInfo(*graph->add_input(), "past_key_values_key_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_input(), "past_key_values_value_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length"), DimSpec(INT64_C(128))});
      }

      // ---- Graph outputs ------------------------------------------------------
      AppendValueInfo(*graph->add_output(), "output_0", DataType::FLOAT16,
                      {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(32000))});
      for (int layer = 0; layer < 4; ++layer) {
        const std::string li = std::to_string(layer);
        AppendValueInfo(*graph->add_output(), "present_key_values_key_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
        AppendValueInfo(*graph->add_output(), "present_key_values_value_" + li, DataType::FLOAT16,
                        {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                         DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
      }

      // ---- Per-value value-tag metadata (onnx_light.value_tag) ---------------
      // Encode each tensor's shape/weight/axes/ambiguous classification directly on
      // its ValueInfoProto (graph inputs, value_info, outputs) and on initializer
      // TensorProtos, mirroring the runtime WriteValueAndNodeTagsToMetadata output
      // and the cases_tiny_llm.cc convention. The shape/weight/axes/ambiguous tag
      // lives on each value's ValueInfoProto (no graph-level value_tags JSON is
      // embedded). Every tensor here is tagged
      // "weight" except the 24 "shape", 10 "axes" and 1 "ambiguous" entries listed
      // below (the exact output of the value-tag inference pass for this model).
      {
        const std::unordered_map<std::string, const char *> non_weight_tags = {
            // value_info tagged "shape"
            {"A::Sq__2", "shape"},
            {"A::Sq__3", "shape"},
            {"B::Sq__2", "shape"},
            {"B::Sq__3", "shape"},
            {"SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25", "shape"},
            {"_onx_add_unsqueeze_12::Shape:", "shape"},
            {"_onx_range_A::Sq::UnSq0x1x3__2", "shape"},
            {"_onx_range_A::Sq__2", "shape"},
            {"_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1", "shape"},
            {"_onx_range_dim1::Sq::UnSq0x1::C1__1", "shape"},
            {"_onx_range_dim1::Sq::UnSq0x1__1", "shape"},
            {"_onx_range_dim1::Sq__1", "shape"},
            {"dim1::Sq__1", "shape"},
            {"dim2::Sq__1", "shape"},
            {"input_ids::Shape1:2", "shape"},
            {"past_key_values_key_0::Shape2:3", "shape"},
            {"past_key_values_value_2::Shape:1", "shape"},
            {"to::Shape-1:", "shape"},
            // initializers tagged "shape"
            {"init7_s3_0_-1_1__1", "shape"},
            {"init7_s3_0_0_2048", "shape"},
            {"init7_s4_0_0_16_128", "shape"},
            {"init7_s4_0_0_8_128", "shape"},
            {"init7_s4_0_16_-1_128", "shape"},
            {"init7_s5_1_1_2_1_1", "shape"},
            // initializers tagged "axes"
            {"init7_s1_1", "axes"},
            {"init7_s1_2__layer_0", "axes"},
            {"init7_s1_2__layer_1", "axes"},
            {"init7_s1_2__layer_2", "axes"},
            {"init7_s1_2__layer_3", "axes"},
            {"init7_s2_0_1__1", "axes"},
            {"init7_s3_0_1_2__2", "axes"},
            {"init7_s3_0_1_2__3", "axes"},
            {"init7_s3_0_1_3__2", "axes"},
            {"init7_s3_1_2_3__3", "axes"},
            // initializer tagged "ambiguous"
            {"init7_s1_-1", "ambiguous"},
        };
        const auto tag_for = [&non_weight_tags](const std::string &name) -> const char * {
          auto it = non_weight_tags.find(name);
          return it == non_weight_tags.end() ? "weight" : it->second;
        };
        const auto set_value_tag = [&tag_for](StringStringEntryProto *entry,
                                              const std::string &name) {
          entry->set_key(core::compute::kValueTagMetadataKey);
          entry->set_value(tag_for(name));
        };
        for (std::size_t i = 0; i < graph->input().size(); ++i) {
          ValueInfoProto *vi = graph->mutable_input(static_cast<std::size_t>(i));
          set_value_tag(vi->add_metadata_props(), vi->name());
        }
        for (std::size_t i = 0; i < graph->value_info().size(); ++i) {
          ValueInfoProto *vi = graph->mutable_value_info(static_cast<std::size_t>(i));
          set_value_tag(vi->add_metadata_props(), vi->name());
        }
        for (std::size_t i = 0; i < graph->output().size(); ++i) {
          ValueInfoProto *vi = graph->mutable_output(static_cast<std::size_t>(i));
          set_value_tag(vi->add_metadata_props(), vi->name());
        }
        for (std::size_t i = 0; i < graph->initializer().size(); ++i) {
          TensorProto *init = graph->mutable_initializer(static_cast<std::size_t>(i));
          set_value_tag(init->add_metadata_props(), init->name());
        }
      }

      // Constant information: mark every build-time-constant node and value
      // (initializers plus any node whose inputs are all constant). In this graph
      // every intermediate depends on the runtime inputs, so only the initializers
      // are constant, but computing it programmatically keeps the golden metadata
      // correct regardless of the graph's complexity.
      core::compute::WriteConstantInfoToMetadata(model);
    }

    registry.emplace_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
