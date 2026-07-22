// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"

#include "onnx_core/annotations/inplace_reuse.h"
#include "onnx_core/annotations/value_tags.h"
#include "onnx_core/backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// ``qwen3_4_layers_like`` — a 4-layer Qwen3-style causal language model.
//
// Faithfully reproduces the structure of a Qwen3 model with 4 transformer
// layers exported from PyTorch via onnxscript (opset 21, IR version 10).
// External weight initializers carry shape and dtype metadata only
// (no payload data); doc_strings are omitted.
//
// Graph signature:
//
//   Inputs:
//     input_ids               INT64[batch_size, sequence_length]
//     attention_mask          INT64[batch_size, total_sequence_length]
//     past_key_values_key_N   FP16[batch_size, 8, past_sequence_length, 128]
//     past_key_values_value_N FP16[batch_size, 8, past_sequence_length, 128]
//
//   Outputs:
//     output_0                FP16[batch_size, sequence_length, 151936]
//     present_key_values_key_N    FP16[batch_size, 8, past_seq+seq, 128]
//     present_key_values_value_N  FP16[batch_size, 8, past_seq+seq, 128]
//
// Architecture per transformer block:
//   RMSNorm (manual: Pow+ReduceMean+Add+Sqrt+Reciprocal+Mul+Cast)
//   GQA-style attention (16 Q heads, 8 KV heads, head_dim=128)
//   RoPE applied to Q and K
//   Causal masking with attention softmax
//   SwiGLU MLP (gate+up projections, Sigmoid, Mul, down projection)
// ---------------------------------------------------------------------------
void RegisterQwen3_4LayersLikeShapeInferenceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(21);

  const std::string name = "test_cc_shape_inference_big_qwen3_4_layers_like";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3f;
  tc.atol = 1e-5f;

  ModelProto &model = tc.emplace_model();
  InitModel(model, /*ir_version=*/10, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

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

  AddInitializer<uint16_t>(*graph, "p_lm_head_weight::T10", {INT64_C(1024), INT64_C(151936)}, {});
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
  AddInitializer<uint16_t>(*graph, "lm_head.weight", {INT64_C(151936), INT64_C(1024)}, {});

  // ---- Nodes --------------------------------------------------------------
  // Constant nodes have been promoted to initializers above.
  {
    NodeProto &n = AddNode(*graph, "Shape", {"input_ids"}, {"input_ids::Shape1:2"});
    AddAttribute<int64_t>(n, "end", INT64_C(2));
    AddAttribute<int64_t>(n, "start", INT64_C(1));
  }
  {
    NodeProto &n =
        AddNode(*graph, "Shape", {"past_key_values_key_0"}, {"past_key_values_key_0::Shape2:3"});
    AddAttribute<int64_t>(n, "end", INT64_C(3));
    AddAttribute<int64_t>(n, "start", INT64_C(2));
  }
  {
    NodeProto &n =
        AddNode(*graph, "Shape", {"past_key_values_value_2"}, {"past_key_values_value_2::Shape:1"});
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
  AddNode(*graph, "Reshape", {"_onx_gather_to::RSh-1", "_onx_add_unsqueeze_12::Shape:"}, {"index"});
  AddNode(*graph, "And", {"le_3", "index"}, {"and_2"});
  AddNode(*graph, "Unsqueeze", {"uoutput_0", "init7_s1_1"}, {"uunsqueeze_16"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"uunsqueeze_16", "uunsqueeze_16"}, {"unsqueeze_16"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Unsqueeze", {"uoutput_1", "init7_s1_1"}, {"uunsqueeze_17"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"uunsqueeze_17", "uunsqueeze_17"}, {"unsqueeze_17"});
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
      AddAttribute<int64_t>(n, "to", INT64_C(1));
    }
    AddNode(*graph, "Pow", {make_layer_name("f32"), "init1_s_"}, {make_layer_name("pow_pre")});
    {
      NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_pre"), "init7_s1_-1"},
                             {make_layer_name("mean_pre")});
      AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
    }
    AddNode(*graph, "Add", {make_layer_name("mean_pre"), "init1_s_2::RSh1"},
            {make_layer_name("add_pre")});
    AddNode(*graph, "Sqrt", {make_layer_name("add_pre")}, {make_layer_name("sqrt_pre")});
    AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_pre")}, {make_layer_name("rsqrt_pre")});
    AddNode(*graph, "Mul", {make_layer_name("f32"), make_layer_name("rsqrt_pre")},
            {make_layer_name("mul_pre")});
    {
      NodeProto &n =
          AddNode(*graph, "Cast", {make_layer_name("mul_pre")}, {make_layer_name("normed_half")});
      AddAttribute<int64_t>(n, "to", INT64_C(10));
    }
    AddNode(*graph, "Mul",
            {norm_prefix + ".input_layernorm.weight", make_layer_name("normed_half")},
            {make_layer_name("normed")});

    // Q projection + q_norm + transpose.
    AddNode(*graph, "MatMul",
            {make_layer_name("normed"), weight_prefix + "_self_attn_q_proj_weight::T10"},
            {make_layer_name("q_mm")});
    {
      NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("q_mm")}, {make_layer_name("q_f32")});
      AddAttribute<int64_t>(n, "to", INT64_C(1));
    }
    AddNode(*graph, "Reshape", {make_layer_name("q_f32"), "init7_s4_0_0_16_128"},
            {make_layer_name("q_4d")});
    AddNode(*graph, "Pow", {make_layer_name("q_4d"), "init1_s_"}, {make_layer_name("pow_q")});
    {
      NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_q"), "init7_s1_-1"},
                             {make_layer_name("mean_q")});
      AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
    }
    AddNode(*graph, "Add", {make_layer_name("mean_q"), "init1_s_2::RSh1"},
            {make_layer_name("add_q")});
    AddNode(*graph, "Sqrt", {make_layer_name("add_q")}, {make_layer_name("sqrt_q")});
    AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_q")}, {make_layer_name("rsqrt_q")});
    AddNode(*graph, "Mul", {make_layer_name("q_4d"), make_layer_name("rsqrt_q")},
            {make_layer_name("mul_q")});
    {
      NodeProto &n =
          AddNode(*graph, "Cast", {make_layer_name("mul_q")}, {make_layer_name("q_normed_half")});
      AddAttribute<int64_t>(n, "to", INT64_C(10));
    }
    AddNode(*graph, "Mul",
            {norm_prefix + ".self_attn.q_norm.weight", make_layer_name("q_normed_half")},
            {make_layer_name("q_normed")});
    {
      NodeProto &n =
          AddNode(*graph, "Transpose", {make_layer_name("q_normed")}, {make_layer_name("q_T")});
      AddAttribute<std::vector<int64_t>>(n, "perm",
                                         {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
    }

    // K projection + k_norm + transpose.
    AddNode(*graph, "MatMul",
            {make_layer_name("normed"), weight_prefix + "_self_attn_k_proj_weight::T10"},
            {make_layer_name("k_mm")});
    {
      NodeProto &n = AddNode(*graph, "Cast", {make_layer_name("k_mm")}, {make_layer_name("k_f32")});
      AddAttribute<int64_t>(n, "to", INT64_C(1));
    }
    AddNode(*graph, "Reshape", {make_layer_name("k_f32"), "init7_s4_0_0_8_128"},
            {make_layer_name("k_4d")});
    AddNode(*graph, "Pow", {make_layer_name("k_4d"), "init1_s_"}, {make_layer_name("pow_k")});
    {
      NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_k"), "init7_s1_-1"},
                             {make_layer_name("mean_k")});
      AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
    }
    AddNode(*graph, "Add", {make_layer_name("mean_k"), "init1_s_2::RSh1"},
            {make_layer_name("add_k")});
    AddNode(*graph, "Sqrt", {make_layer_name("add_k")}, {make_layer_name("sqrt_k")});
    AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_k")}, {make_layer_name("rsqrt_k")});
    AddNode(*graph, "Mul", {make_layer_name("k_4d"), make_layer_name("rsqrt_k")},
            {make_layer_name("mul_k")});
    {
      NodeProto &n =
          AddNode(*graph, "Cast", {make_layer_name("mul_k")}, {make_layer_name("k_normed_half")});
      AddAttribute<int64_t>(n, "to", INT64_C(10));
    }
    AddNode(*graph, "Mul",
            {norm_prefix + ".self_attn.k_norm.weight", make_layer_name("k_normed_half")},
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
    AddNode(*graph, "Mul", {make_layer_name("q_T"), "unsqueeze_16"}, {make_layer_name("q_cos")});
    AddNode(*graph, "Mul", {make_layer_name("q_rot"), "unsqueeze_17"}, {make_layer_name("q_sin")});
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
    AddNode(*graph, "Mul", {make_layer_name("k_T"), "unsqueeze_16"}, {make_layer_name("k_cos")});
    AddNode(*graph, "Mul", {make_layer_name("k_rot"), "unsqueeze_17"}, {make_layer_name("k_sin")});
    AddNode(*graph, "Add", {make_layer_name("k_cos"), make_layer_name("k_sin")},
            {make_layer_name("k_rope")});

    // KV-cache concatenation (produces the layer's present key/value outputs).
    {
      NodeProto &n =
          AddNode(*graph, "Concat", {"past_key_values_key_" + li, make_layer_name("k_rope")},
                  {"present_key_values_key_" + li});
      AddAttribute<int64_t>(n, "axis", INT64_C(-2));
    }
    {
      NodeProto &n =
          AddNode(*graph, "Concat", {"past_key_values_value_" + li, make_layer_name("v_T")},
                  {"present_key_values_value_" + li});
      AddAttribute<int64_t>(n, "axis", INT64_C(-2));
    }

    // GQA: expand KV to Q-head count, then compute scaled dot-product attention.
    AddNode(*graph, "Mul", {"present_key_values_key_" + li, "init10_s1_"},
            {make_layer_name("scaled_k")});
    AddNode(*graph, "Unsqueeze", {make_layer_name("scaled_k"), "init7_s1_2__" + layer_suffix},
            {make_layer_name("scaled_k_unsq")});
    AddNode(*graph, "Unsqueeze", {"present_key_values_value_" + li, "init7_s1_2__" + layer_suffix},
            {make_layer_name("v_unsq")});
    AddNode(*graph, "Expand", {make_layer_name("scaled_k_unsq"), "init7_s5_1_1_2_1_1"},
            {make_layer_name("scaled_k_exp")});
    AddNode(*graph, "Expand", {make_layer_name("v_unsq"), "init7_s5_1_1_2_1_1"},
            {make_layer_name("v_exp")});
    AddNode(*graph, "Reshape", {make_layer_name("scaled_k_exp"), "init7_s4_0_16_-1_128"},
            {make_layer_name("k_gqa")});
    AddNode(*graph, "Reshape", {make_layer_name("v_exp"), "init7_s4_0_16_-1_128"},
            {make_layer_name("v_gqa")});
    AddNode(*graph, "Mul", {make_layer_name("q_rope"), "init10_s1_"},
            {make_layer_name("scaled_q")});
    {
      NodeProto &n =
          AddNode(*graph, "Transpose", {make_layer_name("k_gqa")}, {make_layer_name("k_gqa_T")});
      AddAttribute<std::vector<int64_t>>(n, "perm",
                                         {INT64_C(0), INT64_C(1), INT64_C(3), INT64_C(2)});
    }
    AddNode(*graph, "MatMul", {make_layer_name("scaled_q"), make_layer_name("k_gqa_T")},
            {make_layer_name("attn_scores")});
    AddNode(*graph, "Where",
            {"and_2", make_layer_name("attn_scores"), "init10_s1___" + layer_suffix},
            {make_layer_name("masked")});
    {
      NodeProto &n =
          AddNode(*graph, "Softmax", {make_layer_name("masked")}, {make_layer_name("softmax")});
      AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    }
    AddNode(*graph, "IsNaN", {make_layer_name("softmax")}, {make_layer_name("is_nan")});
    AddNode(*graph, "Where",
            {make_layer_name("is_nan"), "init10_s1_2__" + layer_suffix, make_layer_name("softmax")},
            {make_layer_name("attn_w")});
    AddNode(*graph, "MatMul", {make_layer_name("attn_w"), make_layer_name("v_gqa")},
            {make_layer_name("attn_out")});
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
    {
      NodeProto &n =
          AddNode(*graph, "Cast", {make_layer_name("resid_attn")}, {make_layer_name("post_f32")});
      AddAttribute<int64_t>(n, "to", INT64_C(1));
    }
    AddNode(*graph, "Pow", {make_layer_name("post_f32"), "init1_s_"},
            {make_layer_name("pow_post")});
    {
      NodeProto &n = AddNode(*graph, "ReduceMean", {make_layer_name("pow_post"), "init7_s1_-1"},
                             {make_layer_name("mean_post")});
      AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
    }
    AddNode(*graph, "Add", {make_layer_name("mean_post"), "init1_s_2::RSh1"},
            {make_layer_name("add_post")});
    AddNode(*graph, "Sqrt", {make_layer_name("add_post")}, {make_layer_name("sqrt_post")});
    AddNode(*graph, "Reciprocal", {make_layer_name("sqrt_post")}, {make_layer_name("rsqrt_post")});
    AddNode(*graph, "Mul", {make_layer_name("post_f32"), make_layer_name("rsqrt_post")},
            {make_layer_name("mul_post")});
    {
      NodeProto &n =
          AddNode(*graph, "Cast", {make_layer_name("mul_post")}, {make_layer_name("post_half")});
      AddAttribute<int64_t>(n, "to", INT64_C(10));
    }
    AddNode(*graph, "Mul",
            {norm_prefix + ".post_attention_layernorm.weight", make_layer_name("post_half")},
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
  {
    NodeProto &n = AddNode(*graph, "Cast", {layer_input}, {"final_f32"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"final_f32", "init1_s_"}, {"final_pow"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"final_pow", "init7_s1_-1"}, {"final_mean"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"final_mean", "init1_s_2::RSh1"}, {"final_add"});
  AddNode(*graph, "Sqrt", {"final_add"}, {"final_sqrt"});
  AddNode(*graph, "Reciprocal", {"final_sqrt"}, {"final_rsqrt"});
  AddNode(*graph, "Mul", {"final_f32", "final_rsqrt"}, {"final_mul"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"final_mul"}, {"final_half"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.norm.weight", "final_half"}, {"final_normed"});
  AddNode(*graph, "MatMul", {"final_normed", "p_lm_head_weight::T10"}, {"output_0"});

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
                  "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25", DataType::INT64,
                  {DimSpec(INT64_C(1))});
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
  AppendValueInfo(
      *graph->add_value_info(), "_onx_range_A::Sq::UnSq0x1x3__2", DataType::INT64,
      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
  AppendValueInfo(*graph->add_value_info(), "_onx_range_A::Sq__2", DataType::INT64,
                  {DimSpec("sequence_length")});
  AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1",
                  DataType::FLOAT,
                  {DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(1))});
  AppendValueInfo(*graph->add_value_info(), "_onx_range_dim1::Sq::UnSq0x1::C1__1", DataType::FLOAT,
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
  AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0::UnSq0x1x2__2", DataType::INT64,
                  {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec(INT64_C(1)),
                   DimSpec("past_sequence_length+sequence_length")});
  AppendValueInfo(*graph->add_value_info(), "_onx_range_init7_s_0::UnSq0x1x2__3", DataType::INT64,
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
  AppendValueInfo(*graph->add_value_info(), "to::Shape-1:", DataType::INT64, {DimSpec(INT64_C(1))});
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
  AppendValueInfo(
      *graph->add_value_info(), "uunsqueeze_16", DataType::FLOAT16,
      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});
  AppendValueInfo(
      *graph->add_value_info(), "uunsqueeze_17", DataType::FLOAT16,
      {DimSpec(INT64_C(1)), DimSpec(INT64_C(1)), DimSpec("sequence_length"), DimSpec(INT64_C(64))});

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
    AppendValueInfo(*graph->add_value_info(), vln("attn_2d"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
    AppendValueInfo(*graph->add_value_info(), vln("attn_out"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                     DimSpec(INT64_C(128))});
    AppendValueInfo(*graph->add_value_info(), vln("attn_out_T"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(16)),
                     DimSpec(INT64_C(128))});
    AppendValueInfo(*graph->add_value_info(), vln("attn_proj"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("attn_scores"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                     DimSpec("past_sequence_length+sequence_length")});
    AppendValueInfo(*graph->add_value_info(), vln("attn_w"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                     DimSpec("past_sequence_length+sequence_length")});
    AppendValueInfo(*graph->add_value_info(), vln("down"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("f32"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("gate"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
    AppendValueInfo(*graph->add_value_info(), vln("gate_act"), DataType::FLOAT16,
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
    AppendValueInfo(*graph->add_value_info(), vln("k_f32"), DataType::FLOAT,
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
    AppendValueInfo(*graph->add_value_info(), vln("k_mm"), DataType::FLOAT16,
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
    AppendValueInfo(*graph->add_value_info(), vln("mlp_in"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("mul_k"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                     DimSpec(INT64_C(128))});
    AppendValueInfo(*graph->add_value_info(), vln("mul_post"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("mul_pre"), DataType::FLOAT,
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
    AppendValueInfo(*graph->add_value_info(), vln("normed"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("normed_half"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("out"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("post_f32"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("post_half"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("pow_k"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(8)),
                     DimSpec(INT64_C(128))});
    AppendValueInfo(*graph->add_value_info(), vln("pow_post"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(1024))});
    AppendValueInfo(*graph->add_value_info(), vln("pow_pre"), DataType::FLOAT,
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
    AppendValueInfo(*graph->add_value_info(), vln("q_f32"), DataType::FLOAT,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(2048))});
    AppendValueInfo(*graph->add_value_info(), vln("q_half0"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                     DimSpec(INT64_C(64))});
    AppendValueInfo(*graph->add_value_info(), vln("q_half1"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(16)), DimSpec("sequence_length"),
                     DimSpec(INT64_C(64))});
    AppendValueInfo(*graph->add_value_info(), vln("q_mm"), DataType::FLOAT16,
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
    AppendValueInfo(*graph->add_value_info(), vln("resid_attn"), DataType::FLOAT16,
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
    AppendValueInfo(*graph->add_value_info(), vln("silu"), DataType::FLOAT16,
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
    AppendValueInfo(*graph->add_value_info(), vln("swiglu"), DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(3072))});
    AppendValueInfo(*graph->add_value_info(), vln("up"), DataType::FLOAT16,
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
    AppendValueInfo(*graph->add_value_info(), vln("v_mm"), DataType::FLOAT16,
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
                    {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                     DimSpec(INT64_C(128))});
    AppendValueInfo(*graph->add_input(), "past_key_values_value_" + li, DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                     DimSpec(INT64_C(128))});
  }

  // ---- Graph outputs ------------------------------------------------------
  AppendValueInfo(*graph->add_output(), "output_0", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(151936))});
  for (int layer = 0; layer < 4; ++layer) {
    const std::string li = std::to_string(layer);
    AppendValueInfo(*graph->add_output(), "present_key_values_key_" + li, DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                     DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
    AppendValueInfo(*graph->add_output(), "present_key_values_value_" + li, DataType::FLOAT16,
                    {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                     DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  }

  // ---- Pre-embedded metadata (value-tags, node-tags, inplace-reuse) -------
  // Golden expected outputs of WriteValueAndNodeTagsToMetadata and
  // WriteInPlaceReuseToMetadata for this model, verified by
  // TestBackendMetadataCoverage.
  {
    namespace ann = core::annotations;

    // Graph-level value-tags JSON (DumpValueTagsAsJson, strict ASCII order).
    graph->add_metadata(ann::kValueTagsMetadataKey,
                        "{\"A::Sq__2\":\"shape\",\"A::Sq__3\":\"shape\",\"B::Sq__2\":\"shape"
                        "\",\"B::Sq__3\":\"shape\",\"SqueezeAddPattern_SwapRangeAddScalarPatt"
                        "ern--sym_size_int_25\":\"shape\",\"_onx_add_unsqueeze_12\":\"weight"
                        "\",\"_onx_add_unsqueeze_12::RSh-1\":\"weight\",\"_onx_add_unsqueeze_"
                        "12::Shape:\":\"shape\",\"_onx_cos_mul_weights__1\":\"weight\",\"_onx"
                        "_gather_to::RSh-1\":\"weight\",\"_onx_mul_range_init7_s_02::UnSq1x2x"
                        "3__3\":\"weight\",\"_onx_mul_weights__1\":\"weight\",\"_onx_range_A:"
                        ":Sq::UnSq0x1x3__2\":\"shape\",\"_onx_range_A::Sq__2\":\"shape\",\"_o"
                        "nx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1\":\"shape\",\"_onx_rang"
                        "e_dim1::Sq::UnSq0x1::C1__1\":\"shape\",\"_onx_range_dim1::Sq::UnSq0x"
                        "1__1\":\"shape\",\"_onx_range_dim1::Sq__1\":\"shape\",\"_onx_range_i"
                        "nit7_s_02::UnSq1x2x3__3\":\"weight\",\"_onx_range_init7_s_02__3\":\""
                        "weight\",\"_onx_range_init7_s_0::UnSq0x1x2__2\":\"weight\",\"_onx_ra"
                        "nge_init7_s_0::UnSq0x1x2__3\":\"weight\",\"_onx_range_init7_s_0__2\""
                        ":\"weight\",\"_onx_range_init7_s_0__3\":\"weight\",\"_onx_sin_mul_we"
                        "ights__1\":\"weight\",\"and_2\":\"weight\",\"attention_mask\":\"weig"
                        "ht\",\"dim1::Sq__1\":\"shape\",\"dim2::Sq__1\":\"shape\",\"embedding"
                        "\":\"weight\",\"final_add\":\"weight\",\"final_f32\":\"weight\",\"fi"
                        "nal_half\":\"weight\",\"final_mean\":\"weight\",\"final_mul\":\"weig"
                        "ht\",\"final_normed\":\"weight\",\"final_pow\":\"weight\",\"final_rs"
                        "qrt\":\"weight\",\"final_sqrt\":\"weight\",\"index\":\"weight\",\"in"
                        "it10_s1_\":\"weight\",\"init10_s1_2__layer_0\":\"weight\",\"init10_s"
                        "1_2__layer_1\":\"weight\",\"init10_s1_2__layer_2\":\"weight\",\"init"
                        "10_s1_2__layer_3\":\"weight\",\"init10_s1___layer_0\":\"weight\",\"i"
                        "nit10_s1___layer_1\":\"weight\",\"init10_s1___layer_2\":\"weight\","
                        "\"init10_s1___layer_3\":\"weight\",\"init1_s_\":\"weight\",\"init1_s"
                        "_2::RSh1\":\"weight\",\"init7_s1_-1\":\"ambiguous\",\"init7_s1_1\":"
                        "\"axes\",\"init7_s1_2__layer_0\":\"axes\",\"init7_s1_2__layer_1\":\""
                        "axes\",\"init7_s1_2__layer_2\":\"axes\",\"init7_s1_2__layer_3\":\"ax"
                        "es\",\"init7_s2_0_1__1\":\"axes\",\"init7_s3_0_-1_1__1\":\"shape\","
                        "\"init7_s3_0_0_2048\":\"shape\",\"init7_s3_0_1_2__2\":\"axes\",\"ini"
                        "t7_s3_0_1_2__3\":\"axes\",\"init7_s3_0_1_3__2\":\"axes\",\"init7_s3_"
                        "1_2_3__3\":\"axes\",\"init7_s4_0_0_16_128\":\"shape\",\"init7_s4_0_0"
                        "_8_128\":\"shape\",\"init7_s4_0_16_-1_128\":\"shape\",\"init7_s5_1_1"
                        "_2_1_1\":\"shape\",\"init7_s_0__2\":\"weight\",\"init7_s_0__3\":\"we"
                        "ight\",\"init7_s_1__1\":\"weight\",\"init7_s_1__2\":\"weight\",\"ini"
                        "t7_s_1__3\":\"weight\",\"input_ids\":\"weight\",\"input_ids::Shape1:"
                        "2\":\"shape\",\"layer_0_add_k\":\"weight\",\"layer_0_add_post\":\"we"
                        "ight\",\"layer_0_add_pre\":\"weight\",\"layer_0_add_q\":\"weight\","
                        "\"layer_0_attn_2d\":\"weight\",\"layer_0_attn_out\":\"weight\",\"lay"
                        "er_0_attn_out_T\":\"weight\",\"layer_0_attn_proj\":\"weight\",\"laye"
                        "r_0_attn_scores\":\"weight\",\"layer_0_attn_w\":\"weight\",\"layer_0"
                        "_down\":\"weight\",\"layer_0_f32\":\"weight\",\"layer_0_gate\":\"wei"
                        "ght\",\"layer_0_gate_act\":\"weight\",\"layer_0_is_nan\":\"weight\","
                        "\"layer_0_k_4d\":\"weight\",\"layer_0_k_T\":\"weight\",\"layer_0_k_c"
                        "os\":\"weight\",\"layer_0_k_f32\":\"weight\",\"layer_0_k_gqa\":\"wei"
                        "ght\",\"layer_0_k_gqa_T\":\"weight\",\"layer_0_k_half0\":\"weight\","
                        "\"layer_0_k_half1\":\"weight\",\"layer_0_k_mm\":\"weight\",\"layer_0"
                        "_k_normed\":\"weight\",\"layer_0_k_normed_half\":\"weight\",\"layer_"
                        "0_k_rope\":\"weight\",\"layer_0_k_rot\":\"weight\",\"layer_0_k_sin\""
                        ":\"weight\",\"layer_0_masked\":\"weight\",\"layer_0_mean_k\":\"weigh"
                        "t\",\"layer_0_mean_post\":\"weight\",\"layer_0_mean_pre\":\"weight\""
                        ",\"layer_0_mean_q\":\"weight\",\"layer_0_mlp_in\":\"weight\",\"layer"
                        "_0_mul_k\":\"weight\",\"layer_0_mul_post\":\"weight\",\"layer_0_mul_"
                        "pre\":\"weight\",\"layer_0_mul_q\":\"weight\",\"layer_0_neg_k\":\"we"
                        "ight\",\"layer_0_neg_q\":\"weight\",\"layer_0_normed\":\"weight\",\""
                        "layer_0_normed_half\":\"weight\",\"layer_0_out\":\"weight\",\"layer_"
                        "0_post_f32\":\"weight\",\"layer_0_post_half\":\"weight\",\"layer_0_p"
                        "ow_k\":\"weight\",\"layer_0_pow_post\":\"weight\",\"layer_0_pow_pre"
                        "\":\"weight\",\"layer_0_pow_q\":\"weight\",\"layer_0_q_4d\":\"weight"
                        "\",\"layer_0_q_T\":\"weight\",\"layer_0_q_cos\":\"weight\",\"layer_0"
                        "_q_f32\":\"weight\",\"layer_0_q_half0\":\"weight\",\"layer_0_q_half1"
                        "\":\"weight\",\"layer_0_q_mm\":\"weight\",\"layer_0_q_normed\":\"wei"
                        "ght\",\"layer_0_q_normed_half\":\"weight\",\"layer_0_q_rope\":\"weig"
                        "ht\",\"layer_0_q_rot\":\"weight\",\"layer_0_q_sin\":\"weight\",\"lay"
                        "er_0_resid_attn\":\"weight\",\"layer_0_rsqrt_k\":\"weight\",\"layer_"
                        "0_rsqrt_post\":\"weight\",\"layer_0_rsqrt_pre\":\"weight\",\"layer_0"
                        "_rsqrt_q\":\"weight\",\"layer_0_scaled_k\":\"weight\",\"layer_0_scal"
                        "ed_k_exp\":\"weight\",\"layer_0_scaled_k_unsq\":\"weight\",\"layer_0"
                        "_scaled_q\":\"weight\",\"layer_0_silu\":\"weight\",\"layer_0_softmax"
                        "\":\"weight\",\"layer_0_sqrt_k\":\"weight\",\"layer_0_sqrt_post\":\""
                        "weight\",\"layer_0_sqrt_pre\":\"weight\",\"layer_0_sqrt_q\":\"weight"
                        "\",\"layer_0_swiglu\":\"weight\",\"layer_0_up\":\"weight\",\"layer_0"
                        "_v_4d\":\"weight\",\"layer_0_v_T\":\"weight\",\"layer_0_v_exp\":\"we"
                        "ight\",\"layer_0_v_gqa\":\"weight\",\"layer_0_v_mm\":\"weight\",\"la"
                        "yer_0_v_unsq\":\"weight\",\"layer_1_add_k\":\"weight\",\"layer_1_add"
                        "_post\":\"weight\",\"layer_1_add_pre\":\"weight\",\"layer_1_add_q\":"
                        "\"weight\",\"layer_1_attn_2d\":\"weight\",\"layer_1_attn_out\":\"wei"
                        "ght\",\"layer_1_attn_out_T\":\"weight\",\"layer_1_attn_proj\":\"weig"
                        "ht\",\"layer_1_attn_scores\":\"weight\",\"layer_1_attn_w\":\"weight"
                        "\",\"layer_1_down\":\"weight\",\"layer_1_f32\":\"weight\",\"layer_1_"
                        "gate\":\"weight\",\"layer_1_gate_act\":\"weight\",\"layer_1_is_nan\""
                        ":\"weight\",\"layer_1_k_4d\":\"weight\",\"layer_1_k_T\":\"weight\","
                        "\"layer_1_k_cos\":\"weight\",\"layer_1_k_f32\":\"weight\",\"layer_1_"
                        "k_gqa\":\"weight\",\"layer_1_k_gqa_T\":\"weight\",\"layer_1_k_half0"
                        "\":\"weight\",\"layer_1_k_half1\":\"weight\",\"layer_1_k_mm\":\"weig"
                        "ht\",\"layer_1_k_normed\":\"weight\",\"layer_1_k_normed_half\":\"wei"
                        "ght\",\"layer_1_k_rope\":\"weight\",\"layer_1_k_rot\":\"weight\",\"l"
                        "ayer_1_k_sin\":\"weight\",\"layer_1_masked\":\"weight\",\"layer_1_me"
                        "an_k\":\"weight\",\"layer_1_mean_post\":\"weight\",\"layer_1_mean_pr"
                        "e\":\"weight\",\"layer_1_mean_q\":\"weight\",\"layer_1_mlp_in\":\"we"
                        "ight\",\"layer_1_mul_k\":\"weight\",\"layer_1_mul_post\":\"weight\","
                        "\"layer_1_mul_pre\":\"weight\",\"layer_1_mul_q\":\"weight\",\"layer_"
                        "1_neg_k\":\"weight\",\"layer_1_neg_q\":\"weight\",\"layer_1_normed\""
                        ":\"weight\",\"layer_1_normed_half\":\"weight\",\"layer_1_out\":\"wei"
                        "ght\",\"layer_1_post_f32\":\"weight\",\"layer_1_post_half\":\"weight"
                        "\",\"layer_1_pow_k\":\"weight\",\"layer_1_pow_post\":\"weight\",\"la"
                        "yer_1_pow_pre\":\"weight\",\"layer_1_pow_q\":\"weight\",\"layer_1_q_"
                        "4d\":\"weight\",\"layer_1_q_T\":\"weight\",\"layer_1_q_cos\":\"weigh"
                        "t\",\"layer_1_q_f32\":\"weight\",\"layer_1_q_half0\":\"weight\",\"la"
                        "yer_1_q_half1\":\"weight\",\"layer_1_q_mm\":\"weight\",\"layer_1_q_n"
                        "ormed\":\"weight\",\"layer_1_q_normed_half\":\"weight\",\"layer_1_q_"
                        "rope\":\"weight\",\"layer_1_q_rot\":\"weight\",\"layer_1_q_sin\":\"w"
                        "eight\",\"layer_1_resid_attn\":\"weight\",\"layer_1_rsqrt_k\":\"weig"
                        "ht\",\"layer_1_rsqrt_post\":\"weight\",\"layer_1_rsqrt_pre\":\"weigh"
                        "t\",\"layer_1_rsqrt_q\":\"weight\",\"layer_1_scaled_k\":\"weight\","
                        "\"layer_1_scaled_k_exp\":\"weight\",\"layer_1_scaled_k_unsq\":\"weig"
                        "ht\",\"layer_1_scaled_q\":\"weight\",\"layer_1_silu\":\"weight\",\"l"
                        "ayer_1_softmax\":\"weight\",\"layer_1_sqrt_k\":\"weight\",\"layer_1_"
                        "sqrt_post\":\"weight\",\"layer_1_sqrt_pre\":\"weight\",\"layer_1_sqr"
                        "t_q\":\"weight\",\"layer_1_swiglu\":\"weight\",\"layer_1_up\":\"weig"
                        "ht\",\"layer_1_v_4d\":\"weight\",\"layer_1_v_T\":\"weight\",\"layer_"
                        "1_v_exp\":\"weight\",\"layer_1_v_gqa\":\"weight\",\"layer_1_v_mm\":"
                        "\"weight\",\"layer_1_v_unsq\":\"weight\",\"layer_2_add_k\":\"weight"
                        "\",\"layer_2_add_post\":\"weight\",\"layer_2_add_pre\":\"weight\",\""
                        "layer_2_add_q\":\"weight\",\"layer_2_attn_2d\":\"weight\",\"layer_2_"
                        "attn_out\":\"weight\",\"layer_2_attn_out_T\":\"weight\",\"layer_2_at"
                        "tn_proj\":\"weight\",\"layer_2_attn_scores\":\"weight\",\"layer_2_at"
                        "tn_w\":\"weight\",\"layer_2_down\":\"weight\",\"layer_2_f32\":\"weig"
                        "ht\",\"layer_2_gate\":\"weight\",\"layer_2_gate_act\":\"weight\",\"l"
                        "ayer_2_is_nan\":\"weight\",\"layer_2_k_4d\":\"weight\",\"layer_2_k_T"
                        "\":\"weight\",\"layer_2_k_cos\":\"weight\",\"layer_2_k_f32\":\"weigh"
                        "t\",\"layer_2_k_gqa\":\"weight\",\"layer_2_k_gqa_T\":\"weight\",\"la"
                        "yer_2_k_half0\":\"weight\",\"layer_2_k_half1\":\"weight\",\"layer_2_"
                        "k_mm\":\"weight\",\"layer_2_k_normed\":\"weight\",\"layer_2_k_normed"
                        "_half\":\"weight\",\"layer_2_k_rope\":\"weight\",\"layer_2_k_rot\":"
                        "\"weight\",\"layer_2_k_sin\":\"weight\",\"layer_2_masked\":\"weight"
                        "\",\"layer_2_mean_k\":\"weight\",\"layer_2_mean_post\":\"weight\",\""
                        "layer_2_mean_pre\":\"weight\",\"layer_2_mean_q\":\"weight\",\"layer_"
                        "2_mlp_in\":\"weight\",\"layer_2_mul_k\":\"weight\",\"layer_2_mul_pos"
                        "t\":\"weight\",\"layer_2_mul_pre\":\"weight\",\"layer_2_mul_q\":\"we"
                        "ight\",\"layer_2_neg_k\":\"weight\",\"layer_2_neg_q\":\"weight\",\"l"
                        "ayer_2_normed\":\"weight\",\"layer_2_normed_half\":\"weight\",\"laye"
                        "r_2_out\":\"weight\",\"layer_2_post_f32\":\"weight\",\"layer_2_post_"
                        "half\":\"weight\",\"layer_2_pow_k\":\"weight\",\"layer_2_pow_post\":"
                        "\"weight\",\"layer_2_pow_pre\":\"weight\",\"layer_2_pow_q\":\"weight"
                        "\",\"layer_2_q_4d\":\"weight\",\"layer_2_q_T\":\"weight\",\"layer_2_"
                        "q_cos\":\"weight\",\"layer_2_q_f32\":\"weight\",\"layer_2_q_half0\":"
                        "\"weight\",\"layer_2_q_half1\":\"weight\",\"layer_2_q_mm\":\"weight"
                        "\",\"layer_2_q_normed\":\"weight\",\"layer_2_q_normed_half\":\"weigh"
                        "t\",\"layer_2_q_rope\":\"weight\",\"layer_2_q_rot\":\"weight\",\"lay"
                        "er_2_q_sin\":\"weight\",\"layer_2_resid_attn\":\"weight\",\"layer_2_"
                        "rsqrt_k\":\"weight\",\"layer_2_rsqrt_post\":\"weight\",\"layer_2_rsq"
                        "rt_pre\":\"weight\",\"layer_2_rsqrt_q\":\"weight\",\"layer_2_scaled_"
                        "k\":\"weight\",\"layer_2_scaled_k_exp\":\"weight\",\"layer_2_scaled_"
                        "k_unsq\":\"weight\",\"layer_2_scaled_q\":\"weight\",\"layer_2_silu\""
                        ":\"weight\",\"layer_2_softmax\":\"weight\",\"layer_2_sqrt_k\":\"weig"
                        "ht\",\"layer_2_sqrt_post\":\"weight\",\"layer_2_sqrt_pre\":\"weight"
                        "\",\"layer_2_sqrt_q\":\"weight\",\"layer_2_swiglu\":\"weight\",\"lay"
                        "er_2_up\":\"weight\",\"layer_2_v_4d\":\"weight\",\"layer_2_v_T\":\"w"
                        "eight\",\"layer_2_v_exp\":\"weight\",\"layer_2_v_gqa\":\"weight\",\""
                        "layer_2_v_mm\":\"weight\",\"layer_2_v_unsq\":\"weight\",\"layer_3_ad"
                        "d_k\":\"weight\",\"layer_3_add_post\":\"weight\",\"layer_3_add_pre\""
                        ":\"weight\",\"layer_3_add_q\":\"weight\",\"layer_3_attn_2d\":\"weigh"
                        "t\",\"layer_3_attn_out\":\"weight\",\"layer_3_attn_out_T\":\"weight"
                        "\",\"layer_3_attn_proj\":\"weight\",\"layer_3_attn_scores\":\"weight"
                        "\",\"layer_3_attn_w\":\"weight\",\"layer_3_down\":\"weight\",\"layer"
                        "_3_f32\":\"weight\",\"layer_3_gate\":\"weight\",\"layer_3_gate_act\""
                        ":\"weight\",\"layer_3_is_nan\":\"weight\",\"layer_3_k_4d\":\"weight"
                        "\",\"layer_3_k_T\":\"weight\",\"layer_3_k_cos\":\"weight\",\"layer_3"
                        "_k_f32\":\"weight\",\"layer_3_k_gqa\":\"weight\",\"layer_3_k_gqa_T\""
                        ":\"weight\",\"layer_3_k_half0\":\"weight\",\"layer_3_k_half1\":\"wei"
                        "ght\",\"layer_3_k_mm\":\"weight\",\"layer_3_k_normed\":\"weight\",\""
                        "layer_3_k_normed_half\":\"weight\",\"layer_3_k_rope\":\"weight\",\"l"
                        "ayer_3_k_rot\":\"weight\",\"layer_3_k_sin\":\"weight\",\"layer_3_mas"
                        "ked\":\"weight\",\"layer_3_mean_k\":\"weight\",\"layer_3_mean_post\""
                        ":\"weight\",\"layer_3_mean_pre\":\"weight\",\"layer_3_mean_q\":\"wei"
                        "ght\",\"layer_3_mlp_in\":\"weight\",\"layer_3_mul_k\":\"weight\",\"l"
                        "ayer_3_mul_post\":\"weight\",\"layer_3_mul_pre\":\"weight\",\"layer_"
                        "3_mul_q\":\"weight\",\"layer_3_neg_k\":\"weight\",\"layer_3_neg_q\":"
                        "\"weight\",\"layer_3_normed\":\"weight\",\"layer_3_normed_half\":\"w"
                        "eight\",\"layer_3_out\":\"weight\",\"layer_3_post_f32\":\"weight\","
                        "\"layer_3_post_half\":\"weight\",\"layer_3_pow_k\":\"weight\",\"laye"
                        "r_3_pow_post\":\"weight\",\"layer_3_pow_pre\":\"weight\",\"layer_3_p"
                        "ow_q\":\"weight\",\"layer_3_q_4d\":\"weight\",\"layer_3_q_T\":\"weig"
                        "ht\",\"layer_3_q_cos\":\"weight\",\"layer_3_q_f32\":\"weight\",\"lay"
                        "er_3_q_half0\":\"weight\",\"layer_3_q_half1\":\"weight\",\"layer_3_q"
                        "_mm\":\"weight\",\"layer_3_q_normed\":\"weight\",\"layer_3_q_normed_"
                        "half\":\"weight\",\"layer_3_q_rope\":\"weight\",\"layer_3_q_rot\":\""
                        "weight\",\"layer_3_q_sin\":\"weight\",\"layer_3_resid_attn\":\"weigh"
                        "t\",\"layer_3_rsqrt_k\":\"weight\",\"layer_3_rsqrt_post\":\"weight\""
                        ",\"layer_3_rsqrt_pre\":\"weight\",\"layer_3_rsqrt_q\":\"weight\",\"l"
                        "ayer_3_scaled_k\":\"weight\",\"layer_3_scaled_k_exp\":\"weight\",\"l"
                        "ayer_3_scaled_k_unsq\":\"weight\",\"layer_3_scaled_q\":\"weight\",\""
                        "layer_3_silu\":\"weight\",\"layer_3_softmax\":\"weight\",\"layer_3_s"
                        "qrt_k\":\"weight\",\"layer_3_sqrt_post\":\"weight\",\"layer_3_sqrt_p"
                        "re\":\"weight\",\"layer_3_sqrt_q\":\"weight\",\"layer_3_swiglu\":\"w"
                        "eight\",\"layer_3_up\":\"weight\",\"layer_3_v_4d\":\"weight\",\"laye"
                        "r_3_v_T\":\"weight\",\"layer_3_v_exp\":\"weight\",\"layer_3_v_gqa\":"
                        "\"weight\",\"layer_3_v_mm\":\"weight\",\"layer_3_v_unsq\":\"weight\""
                        ",\"le_3\":\"weight\",\"lm_head.weight\":\"weight\",\"model.layers.0."
                        "input_layernorm.weight\":\"weight\",\"model.layers.0.post_attention_"
                        "layernorm.weight\":\"weight\",\"model.layers.0.self_attn.k_norm.weig"
                        "ht\":\"weight\",\"model.layers.0.self_attn.q_norm.weight\":\"weight"
                        "\",\"model.layers.1.input_layernorm.weight\":\"weight\",\"model.laye"
                        "rs.1.post_attention_layernorm.weight\":\"weight\",\"model.layers.1.s"
                        "elf_attn.k_norm.weight\":\"weight\",\"model.layers.1.self_attn.q_nor"
                        "m.weight\":\"weight\",\"model.layers.2.input_layernorm.weight\":\"we"
                        "ight\",\"model.layers.2.post_attention_layernorm.weight\":\"weight\""
                        ",\"model.layers.2.self_attn.k_norm.weight\":\"weight\",\"model.layer"
                        "s.2.self_attn.q_norm.weight\":\"weight\",\"model.layers.3.input_laye"
                        "rnorm.weight\":\"weight\",\"model.layers.3.post_attention_layernorm."
                        "weight\":\"weight\",\"model.layers.3.self_attn.k_norm.weight\":\"wei"
                        "ght\",\"model.layers.3.self_attn.q_norm.weight\":\"weight\",\"model."
                        "norm.weight\":\"weight\",\"output_0\":\"weight\",\"p_lm_head_weight:"
                        ":T10\":\"weight\",\"p_model_layers_0_mlp_down_proj_weight::T10\":\"w"
                        "eight\",\"p_model_layers_0_mlp_gate_proj_weight::T10\":\"weight\",\""
                        "p_model_layers_0_mlp_up_proj_weight::T10\":\"weight\",\"p_model_laye"
                        "rs_0_self_attn_k_proj_weight::T10\":\"weight\",\"p_model_layers_0_se"
                        "lf_attn_o_proj_weight::T10\":\"weight\",\"p_model_layers_0_self_attn"
                        "_q_proj_weight::T10\":\"weight\",\"p_model_layers_0_self_attn_v_proj"
                        "_weight::T10\":\"weight\",\"p_model_layers_1_mlp_down_proj_weight::T"
                        "10\":\"weight\",\"p_model_layers_1_mlp_gate_proj_weight::T10\":\"wei"
                        "ght\",\"p_model_layers_1_mlp_up_proj_weight::T10\":\"weight\",\"p_mo"
                        "del_layers_1_self_attn_k_proj_weight::T10\":\"weight\",\"p_model_lay"
                        "ers_1_self_attn_o_proj_weight::T10\":\"weight\",\"p_model_layers_1_s"
                        "elf_attn_q_proj_weight::T10\":\"weight\",\"p_model_layers_1_self_att"
                        "n_v_proj_weight::T10\":\"weight\",\"p_model_layers_2_mlp_down_proj_w"
                        "eight::T10\":\"weight\",\"p_model_layers_2_mlp_gate_proj_weight::T10"
                        "\":\"weight\",\"p_model_layers_2_mlp_up_proj_weight::T10\":\"weight"
                        "\",\"p_model_layers_2_self_attn_k_proj_weight::T10\":\"weight\",\"p_"
                        "model_layers_2_self_attn_o_proj_weight::T10\":\"weight\",\"p_model_l"
                        "ayers_2_self_attn_q_proj_weight::T10\":\"weight\",\"p_model_layers_2"
                        "_self_attn_v_proj_weight::T10\":\"weight\",\"p_model_layers_3_mlp_do"
                        "wn_proj_weight::T10\":\"weight\",\"p_model_layers_3_mlp_gate_proj_we"
                        "ight::T10\":\"weight\",\"p_model_layers_3_mlp_up_proj_weight::T10\":"
                        "\"weight\",\"p_model_layers_3_self_attn_k_proj_weight::T10\":\"weigh"
                        "t\",\"p_model_layers_3_self_attn_o_proj_weight::T10\":\"weight\",\"p"
                        "_model_layers_3_self_attn_q_proj_weight::T10\":\"weight\",\"p_model_"
                        "layers_3_self_attn_v_proj_weight::T10\":\"weight\",\"past_key_values"
                        "_key_0\":\"weight\",\"past_key_values_key_0::Shape2:3\":\"shape\",\""
                        "past_key_values_key_1\":\"weight\",\"past_key_values_key_2\":\"weigh"
                        "t\",\"past_key_values_key_3\":\"weight\",\"past_key_values_value_0\""
                        ":\"weight\",\"past_key_values_value_1\":\"weight\",\"past_key_values"
                        "_value_2\":\"weight\",\"past_key_values_value_2::Shape:1\":\"shape\""
                        ",\"past_key_values_value_3\":\"weight\",\"present_key_values_key_0\""
                        ":\"weight\",\"present_key_values_key_1\":\"weight\",\"present_key_va"
                        "lues_key_2\":\"weight\",\"present_key_values_key_3\":\"weight\",\"pr"
                        "esent_key_values_value_0\":\"weight\",\"present_key_values_value_1\""
                        ":\"weight\",\"present_key_values_value_2\":\"weight\",\"present_key_"
                        "values_value_3\":\"weight\",\"to\":\"weight\",\"to::RSh-1\":\"weight"
                        "\",\"to::Shape-1:\":\"shape\",\"to_322\":\"weight\",\"unsqueeze_16\""
                        ":\"weight\",\"unsqueeze_17\":\"weight\",\"uoutput_0\":\"weight\",\"u"
                        "output_1\":\"weight\",\"uunsqueeze_16\":\"weight\",\"uunsqueeze_17\""
                        ":\"weight\"}");

    // Helper: add a metadata entry to the i-th node.
    const auto node_meta = [&](int i, const char *key, const std::string &value) {
      (*graph->mutable_node())[static_cast<std::size_t>(i)].add_metadata(key, value);
    };

    node_meta(0, ann::kNodeTagMetadataKey, "shape");
    node_meta(1, ann::kNodeTagMetadataKey, "shape");
    node_meta(2, ann::kNodeTagMetadataKey, "shape");
    node_meta(3, ann::kNodeTagMetadataKey, "shape");
    node_meta(3, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(3, ann::kReleaseAfterMetadataKey, "input_ids::Shape1:2");
    node_meta(3, ann::kReleaseAfterShapeTagMetadataKey, "input_ids::Shape1:2");
    node_meta(4, ann::kNodeTagMetadataKey, "shape");
    node_meta(5, ann::kNodeTagMetadataKey, "shape");
    node_meta(6, ann::kNodeTagMetadataKey, "shape");
    node_meta(6, ann::kReleaseAfterMetadataKey, "dim1::Sq__1;dim2::Sq__1");
    node_meta(6, ann::kReleaseAfterShapeTagMetadataKey, "dim1::Sq__1;dim2::Sq__1");
    node_meta(7, ann::kNodeTagMetadataKey, "shape");
    node_meta(7, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(7, ann::kReleaseAfterMetadataKey, "_onx_range_dim1::Sq__1");
    node_meta(7, ann::kReleaseAfterShapeTagMetadataKey, "_onx_range_dim1::Sq__1");
    node_meta(8, ann::kNodeTagMetadataKey, "shape");
    node_meta(8, ann::kReleaseAfterMetadataKey, "_onx_range_dim1::Sq::UnSq0x1__1");
    node_meta(8, ann::kReleaseAfterShapeTagMetadataKey, "_onx_range_dim1::Sq::UnSq0x1__1");
    node_meta(9, ann::kNodeTagMetadataKey, "shape");
    node_meta(9, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(9, ann::kReleaseAfterMetadataKey, "_onx_range_dim1::Sq::UnSq0x1::C1__1");
    node_meta(9, ann::kReleaseAfterShapeTagMetadataKey, "_onx_range_dim1::Sq::UnSq0x1::C1__1");
    node_meta(10, ann::kNodeTagMetadataKey, "weight");
    node_meta(10, ann::kReleaseAfterMetadataKey, "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1");
    node_meta(10, ann::kReleaseAfterShapeTagMetadataKey,
              "_onx_range_dim1::Sq::UnSq0x1::C1::RSh0x-1x1__1");
    node_meta(11, ann::kNodeTagMetadataKey, "weight");
    node_meta(12, ann::kNodeTagMetadataKey, "weight");
    node_meta(12, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(12, ann::kReleaseAfterMetadataKey, "_onx_mul_weights__1");
    node_meta(13, ann::kNodeTagMetadataKey, "weight");
    node_meta(13, ann::kReleaseAfterMetadataKey, "_onx_cos_mul_weights__1");
    node_meta(14, ann::kNodeTagMetadataKey, "weight");
    node_meta(14, ann::kReleaseAfterMetadataKey, "_onx_sin_mul_weights__1");
    node_meta(15, ann::kNodeTagMetadataKey, "shape");
    node_meta(15, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(15, ann::kReleaseAfterMetadataKey, "past_key_values_key_0::Shape2:3");
    node_meta(15, ann::kReleaseAfterShapeTagMetadataKey, "past_key_values_key_0::Shape2:3");
    node_meta(16, ann::kNodeTagMetadataKey, "shape");
    node_meta(17, ann::kNodeTagMetadataKey, "weight");
    node_meta(18, ann::kNodeTagMetadataKey, "shape");
    node_meta(18, ann::kReleaseAfterMetadataKey, "A::Sq__2;B::Sq__2");
    node_meta(18, ann::kReleaseAfterShapeTagMetadataKey, "A::Sq__2;B::Sq__2");
    node_meta(19, ann::kNodeTagMetadataKey, "weight");
    node_meta(19, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(19, ann::kReleaseAfterMetadataKey, "_onx_range_init7_s_0__2");
    node_meta(20, ann::kNodeTagMetadataKey, "shape");
    node_meta(20, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(20, ann::kReleaseAfterMetadataKey, "_onx_range_A::Sq__2");
    node_meta(20, ann::kReleaseAfterShapeTagMetadataKey, "_onx_range_A::Sq__2");
    node_meta(21, ann::kNodeTagMetadataKey, "weight");
    node_meta(21, ann::kReleaseAfterMetadataKey,
              "_onx_range_init7_s_0::UnSq0x1x2__2;_onx_range_A::Sq::UnSq0x1x3__2");
    node_meta(21, ann::kReleaseAfterShapeTagMetadataKey, "_onx_range_A::Sq::UnSq0x1x3__2");
    node_meta(22, ann::kNodeTagMetadataKey, "weight");
    node_meta(23, ann::kNodeTagMetadataKey, "weight");
    node_meta(24, ann::kNodeTagMetadataKey, "shape");
    node_meta(25, ann::kNodeTagMetadataKey, "shape");
    node_meta(25, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(25, ann::kReleaseAfterMetadataKey,
              "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25");
    node_meta(25, ann::kReleaseAfterShapeTagMetadataKey,
              "SqueezeAddPattern_SwapRangeAddScalarPattern--sym_size_int_25");
    node_meta(26, ann::kNodeTagMetadataKey, "shape");
    node_meta(26, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(26, ann::kReleaseAfterMetadataKey, "past_key_values_value_2::Shape:1");
    node_meta(26, ann::kReleaseAfterShapeTagMetadataKey, "past_key_values_value_2::Shape:1");
    node_meta(27, ann::kNodeTagMetadataKey, "weight");
    node_meta(27, ann::kReleaseAfterMetadataKey, "A::Sq__3");
    node_meta(27, ann::kReleaseAfterShapeTagMetadataKey, "A::Sq__3");
    node_meta(28, ann::kNodeTagMetadataKey, "weight");
    node_meta(28, ann::kReleaseAfterMetadataKey, "B::Sq__3");
    node_meta(28, ann::kReleaseAfterShapeTagMetadataKey, "B::Sq__3");
    node_meta(29, ann::kNodeTagMetadataKey, "weight");
    node_meta(29, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(29, ann::kReleaseAfterMetadataKey, "_onx_range_init7_s_0__3");
    node_meta(30, ann::kNodeTagMetadataKey, "weight");
    node_meta(30, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(30, ann::kReleaseAfterMetadataKey, "_onx_range_init7_s_02__3");
    node_meta(31, ann::kNodeTagMetadataKey, "weight");
    node_meta(31, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(31, ann::kReleaseAfterMetadataKey,
              "_onx_range_init7_s_02::UnSq1x2x3__3;to::Shape-1:");
    node_meta(31, ann::kReleaseAfterShapeTagMetadataKey, "to::Shape-1:");
    node_meta(32, ann::kNodeTagMetadataKey, "weight");
    node_meta(32, ann::kReleaseAfterMetadataKey,
              "_onx_mul_range_init7_s_02::UnSq1x2x3__3;_onx_range_init7_s_0::UnSq0x1x2__3");
    node_meta(33, ann::kNodeTagMetadataKey, "shape");
    node_meta(34, ann::kNodeTagMetadataKey, "weight");
    node_meta(34, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(34, ann::kReleaseAfterMetadataKey, "to");
    node_meta(35, ann::kNodeTagMetadataKey, "weight");
    node_meta(35, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(35, ann::kReleaseAfterMetadataKey, "_onx_add_unsqueeze_12");
    node_meta(36, ann::kNodeTagMetadataKey, "weight");
    node_meta(36, ann::kReleaseAfterMetadataKey, "to::RSh-1;_onx_add_unsqueeze_12::RSh-1");
    node_meta(37, ann::kNodeTagMetadataKey, "weight");
    node_meta(37, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(37, ann::kReleaseAfterMetadataKey,
              "_onx_gather_to::RSh-1;_onx_add_unsqueeze_12::Shape:");
    node_meta(37, ann::kReleaseAfterShapeTagMetadataKey, "_onx_add_unsqueeze_12::Shape:");
    node_meta(38, ann::kNodeTagMetadataKey, "weight");
    node_meta(38, ann::kReleaseAfterMetadataKey, "le_3;index");
    node_meta(39, ann::kNodeTagMetadataKey, "weight");
    node_meta(39, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(39, ann::kReleaseAfterMetadataKey, "uoutput_0");
    node_meta(40, ann::kNodeTagMetadataKey, "weight");
    node_meta(40, ann::kReleaseAfterMetadataKey, "uunsqueeze_16");
    node_meta(41, ann::kNodeTagMetadataKey, "weight");
    node_meta(41, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(41, ann::kReleaseAfterMetadataKey, "uoutput_1");
    node_meta(42, ann::kNodeTagMetadataKey, "weight");
    node_meta(42, ann::kReleaseAfterMetadataKey, "uunsqueeze_17");
    node_meta(43, ann::kNodeTagMetadataKey, "weight");
    node_meta(44, ann::kNodeTagMetadataKey, "weight");
    node_meta(45, ann::kNodeTagMetadataKey, "weight");
    node_meta(45, ann::kReleaseAfterMetadataKey, "layer_0_pow_pre");
    node_meta(46, ann::kNodeTagMetadataKey, "weight");
    node_meta(46, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(46, ann::kReleaseAfterMetadataKey, "layer_0_mean_pre");
    node_meta(47, ann::kNodeTagMetadataKey, "weight");
    node_meta(47, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(47, ann::kReleaseAfterMetadataKey, "layer_0_add_pre");
    node_meta(48, ann::kNodeTagMetadataKey, "weight");
    node_meta(48, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(48, ann::kReleaseAfterMetadataKey, "layer_0_sqrt_pre");
    node_meta(49, ann::kNodeTagMetadataKey, "weight");
    node_meta(49, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(49, ann::kReleaseAfterMetadataKey, "layer_0_f32;layer_0_rsqrt_pre");
    node_meta(50, ann::kNodeTagMetadataKey, "weight");
    node_meta(50, ann::kReleaseAfterMetadataKey, "layer_0_mul_pre");
    node_meta(51, ann::kNodeTagMetadataKey, "weight");
    node_meta(51, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(51, ann::kReleaseAfterMetadataKey, "layer_0_normed_half");
    node_meta(52, ann::kNodeTagMetadataKey, "weight");
    node_meta(53, ann::kNodeTagMetadataKey, "weight");
    node_meta(53, ann::kReleaseAfterMetadataKey, "layer_0_q_mm");
    node_meta(54, ann::kNodeTagMetadataKey, "weight");
    node_meta(54, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(54, ann::kReleaseAfterMetadataKey, "layer_0_q_f32");
    node_meta(55, ann::kNodeTagMetadataKey, "weight");
    node_meta(56, ann::kNodeTagMetadataKey, "weight");
    node_meta(56, ann::kReleaseAfterMetadataKey, "layer_0_pow_q");
    node_meta(57, ann::kNodeTagMetadataKey, "weight");
    node_meta(57, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(57, ann::kReleaseAfterMetadataKey, "layer_0_mean_q");
    node_meta(58, ann::kNodeTagMetadataKey, "weight");
    node_meta(58, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(58, ann::kReleaseAfterMetadataKey, "layer_0_add_q");
    node_meta(59, ann::kNodeTagMetadataKey, "weight");
    node_meta(59, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(59, ann::kReleaseAfterMetadataKey, "layer_0_sqrt_q");
    node_meta(60, ann::kNodeTagMetadataKey, "weight");
    node_meta(60, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(60, ann::kReleaseAfterMetadataKey, "layer_0_q_4d;layer_0_rsqrt_q");
    node_meta(61, ann::kNodeTagMetadataKey, "weight");
    node_meta(61, ann::kReleaseAfterMetadataKey, "layer_0_mul_q");
    node_meta(62, ann::kNodeTagMetadataKey, "weight");
    node_meta(62, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(62, ann::kReleaseAfterMetadataKey, "layer_0_q_normed_half");
    node_meta(63, ann::kNodeTagMetadataKey, "weight");
    node_meta(63, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(63, ann::kReleaseAfterMetadataKey, "layer_0_q_normed");
    node_meta(64, ann::kNodeTagMetadataKey, "weight");
    node_meta(65, ann::kNodeTagMetadataKey, "weight");
    node_meta(65, ann::kReleaseAfterMetadataKey, "layer_0_k_mm");
    node_meta(66, ann::kNodeTagMetadataKey, "weight");
    node_meta(66, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(66, ann::kReleaseAfterMetadataKey, "layer_0_k_f32");
    node_meta(67, ann::kNodeTagMetadataKey, "weight");
    node_meta(68, ann::kNodeTagMetadataKey, "weight");
    node_meta(68, ann::kReleaseAfterMetadataKey, "layer_0_pow_k");
    node_meta(69, ann::kNodeTagMetadataKey, "weight");
    node_meta(69, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(69, ann::kReleaseAfterMetadataKey, "layer_0_mean_k");
    node_meta(70, ann::kNodeTagMetadataKey, "weight");
    node_meta(70, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(70, ann::kReleaseAfterMetadataKey, "layer_0_add_k");
    node_meta(71, ann::kNodeTagMetadataKey, "weight");
    node_meta(71, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(71, ann::kReleaseAfterMetadataKey, "layer_0_sqrt_k");
    node_meta(72, ann::kNodeTagMetadataKey, "weight");
    node_meta(72, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(72, ann::kReleaseAfterMetadataKey, "layer_0_k_4d;layer_0_rsqrt_k");
    node_meta(73, ann::kNodeTagMetadataKey, "weight");
    node_meta(73, ann::kReleaseAfterMetadataKey, "layer_0_mul_k");
    node_meta(74, ann::kNodeTagMetadataKey, "weight");
    node_meta(74, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(74, ann::kReleaseAfterMetadataKey, "layer_0_k_normed_half");
    node_meta(75, ann::kNodeTagMetadataKey, "weight");
    node_meta(75, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(75, ann::kReleaseAfterMetadataKey, "layer_0_k_normed");
    node_meta(76, ann::kNodeTagMetadataKey, "weight");
    node_meta(76, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(76, ann::kReleaseAfterMetadataKey, "layer_0_normed");
    node_meta(77, ann::kNodeTagMetadataKey, "weight");
    node_meta(77, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(77, ann::kReleaseAfterMetadataKey, "layer_0_v_mm");
    node_meta(78, ann::kNodeTagMetadataKey, "weight");
    node_meta(78, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(78, ann::kReleaseAfterMetadataKey, "layer_0_v_4d");
    node_meta(79, ann::kNodeTagMetadataKey, "weight");
    node_meta(80, ann::kNodeTagMetadataKey, "weight");
    node_meta(80, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(80, ann::kReleaseAfterMetadataKey, "layer_0_q_half1");
    node_meta(81, ann::kNodeTagMetadataKey, "weight");
    node_meta(81, ann::kReleaseAfterMetadataKey, "layer_0_neg_q;layer_0_q_half0");
    node_meta(82, ann::kNodeTagMetadataKey, "weight");
    node_meta(82, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(82, ann::kReleaseAfterMetadataKey, "layer_0_q_T");
    node_meta(83, ann::kNodeTagMetadataKey, "weight");
    node_meta(83, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(83, ann::kReleaseAfterMetadataKey, "layer_0_q_rot");
    node_meta(84, ann::kNodeTagMetadataKey, "weight");
    node_meta(84, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(84, ann::kReleaseAfterMetadataKey, "layer_0_q_cos;layer_0_q_sin");
    node_meta(85, ann::kNodeTagMetadataKey, "weight");
    node_meta(86, ann::kNodeTagMetadataKey, "weight");
    node_meta(86, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(86, ann::kReleaseAfterMetadataKey, "layer_0_k_half1");
    node_meta(87, ann::kNodeTagMetadataKey, "weight");
    node_meta(87, ann::kReleaseAfterMetadataKey, "layer_0_neg_k;layer_0_k_half0");
    node_meta(88, ann::kNodeTagMetadataKey, "weight");
    node_meta(88, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(88, ann::kReleaseAfterMetadataKey, "layer_0_k_T");
    node_meta(89, ann::kNodeTagMetadataKey, "weight");
    node_meta(89, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(89, ann::kReleaseAfterMetadataKey, "layer_0_k_rot");
    node_meta(90, ann::kNodeTagMetadataKey, "weight");
    node_meta(90, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(90, ann::kReleaseAfterMetadataKey, "layer_0_k_cos;layer_0_k_sin");
    node_meta(91, ann::kNodeTagMetadataKey, "weight");
    node_meta(91, ann::kReleaseAfterMetadataKey, "layer_0_k_rope");
    node_meta(92, ann::kNodeTagMetadataKey, "weight");
    node_meta(92, ann::kReleaseAfterMetadataKey, "layer_0_v_T");
    node_meta(93, ann::kNodeTagMetadataKey, "weight");
    node_meta(94, ann::kNodeTagMetadataKey, "weight");
    node_meta(94, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(94, ann::kReleaseAfterMetadataKey, "layer_0_scaled_k");
    node_meta(95, ann::kNodeTagMetadataKey, "weight");
    node_meta(96, ann::kNodeTagMetadataKey, "weight");
    node_meta(96, ann::kReleaseAfterMetadataKey, "layer_0_scaled_k_unsq");
    node_meta(97, ann::kNodeTagMetadataKey, "weight");
    node_meta(97, ann::kReleaseAfterMetadataKey, "layer_0_v_unsq");
    node_meta(98, ann::kNodeTagMetadataKey, "weight");
    node_meta(98, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(98, ann::kReleaseAfterMetadataKey, "layer_0_scaled_k_exp");
    node_meta(99, ann::kNodeTagMetadataKey, "weight");
    node_meta(99, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(99, ann::kReleaseAfterMetadataKey, "layer_0_v_exp");
    node_meta(100, ann::kNodeTagMetadataKey, "weight");
    node_meta(100, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(100, ann::kReleaseAfterMetadataKey, "layer_0_q_rope");
    node_meta(101, ann::kNodeTagMetadataKey, "weight");
    node_meta(101, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(101, ann::kReleaseAfterMetadataKey, "layer_0_k_gqa");
    node_meta(102, ann::kNodeTagMetadataKey, "weight");
    node_meta(102, ann::kReleaseAfterMetadataKey, "layer_0_scaled_q;layer_0_k_gqa_T");
    node_meta(103, ann::kNodeTagMetadataKey, "weight");
    node_meta(103, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(103, ann::kReleaseAfterMetadataKey, "layer_0_attn_scores");
    node_meta(104, ann::kNodeTagMetadataKey, "weight");
    node_meta(104, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(104, ann::kReleaseAfterMetadataKey, "layer_0_masked");
    node_meta(105, ann::kNodeTagMetadataKey, "weight");
    node_meta(106, ann::kNodeTagMetadataKey, "weight");
    node_meta(106, ann::kInPlaceReuseMetadataKey, "0:2:equal");
    node_meta(106, ann::kReleaseAfterMetadataKey, "layer_0_is_nan;layer_0_softmax");
    node_meta(107, ann::kNodeTagMetadataKey, "weight");
    node_meta(107, ann::kReleaseAfterMetadataKey, "layer_0_attn_w;layer_0_v_gqa");
    node_meta(108, ann::kNodeTagMetadataKey, "weight");
    node_meta(108, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(108, ann::kReleaseAfterMetadataKey, "layer_0_attn_out");
    node_meta(109, ann::kNodeTagMetadataKey, "weight");
    node_meta(109, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(109, ann::kReleaseAfterMetadataKey, "layer_0_attn_out_T");
    node_meta(110, ann::kNodeTagMetadataKey, "weight");
    node_meta(110, ann::kReleaseAfterMetadataKey, "layer_0_attn_2d");
    node_meta(111, ann::kNodeTagMetadataKey, "weight");
    node_meta(111, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(111, ann::kReleaseAfterMetadataKey, "embedding;layer_0_attn_proj");
    node_meta(112, ann::kNodeTagMetadataKey, "weight");
    node_meta(113, ann::kNodeTagMetadataKey, "weight");
    node_meta(114, ann::kNodeTagMetadataKey, "weight");
    node_meta(114, ann::kReleaseAfterMetadataKey, "layer_0_pow_post");
    node_meta(115, ann::kNodeTagMetadataKey, "weight");
    node_meta(115, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(115, ann::kReleaseAfterMetadataKey, "layer_0_mean_post");
    node_meta(116, ann::kNodeTagMetadataKey, "weight");
    node_meta(116, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(116, ann::kReleaseAfterMetadataKey, "layer_0_add_post");
    node_meta(117, ann::kNodeTagMetadataKey, "weight");
    node_meta(117, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(117, ann::kReleaseAfterMetadataKey, "layer_0_sqrt_post");
    node_meta(118, ann::kNodeTagMetadataKey, "weight");
    node_meta(118, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(118, ann::kReleaseAfterMetadataKey, "layer_0_post_f32;layer_0_rsqrt_post");
    node_meta(119, ann::kNodeTagMetadataKey, "weight");
    node_meta(119, ann::kReleaseAfterMetadataKey, "layer_0_mul_post");
    node_meta(120, ann::kNodeTagMetadataKey, "weight");
    node_meta(120, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(120, ann::kReleaseAfterMetadataKey, "layer_0_post_half");
    node_meta(121, ann::kNodeTagMetadataKey, "weight");
    node_meta(122, ann::kNodeTagMetadataKey, "weight");
    node_meta(123, ann::kNodeTagMetadataKey, "weight");
    node_meta(123, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(123, ann::kReleaseAfterMetadataKey, "layer_0_gate;layer_0_gate_act");
    node_meta(124, ann::kNodeTagMetadataKey, "weight");
    node_meta(124, ann::kReleaseAfterMetadataKey, "layer_0_mlp_in");
    node_meta(125, ann::kNodeTagMetadataKey, "weight");
    node_meta(125, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(125, ann::kReleaseAfterMetadataKey, "layer_0_silu;layer_0_up");
    node_meta(126, ann::kNodeTagMetadataKey, "weight");
    node_meta(126, ann::kReleaseAfterMetadataKey, "layer_0_swiglu");
    node_meta(127, ann::kNodeTagMetadataKey, "weight");
    node_meta(127, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(127, ann::kReleaseAfterMetadataKey, "layer_0_resid_attn;layer_0_down");
    node_meta(128, ann::kNodeTagMetadataKey, "weight");
    node_meta(129, ann::kNodeTagMetadataKey, "weight");
    node_meta(130, ann::kNodeTagMetadataKey, "weight");
    node_meta(130, ann::kReleaseAfterMetadataKey, "layer_1_pow_pre");
    node_meta(131, ann::kNodeTagMetadataKey, "weight");
    node_meta(131, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(131, ann::kReleaseAfterMetadataKey, "layer_1_mean_pre");
    node_meta(132, ann::kNodeTagMetadataKey, "weight");
    node_meta(132, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(132, ann::kReleaseAfterMetadataKey, "layer_1_add_pre");
    node_meta(133, ann::kNodeTagMetadataKey, "weight");
    node_meta(133, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(133, ann::kReleaseAfterMetadataKey, "layer_1_sqrt_pre");
    node_meta(134, ann::kNodeTagMetadataKey, "weight");
    node_meta(134, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(134, ann::kReleaseAfterMetadataKey, "layer_1_f32;layer_1_rsqrt_pre");
    node_meta(135, ann::kNodeTagMetadataKey, "weight");
    node_meta(135, ann::kReleaseAfterMetadataKey, "layer_1_mul_pre");
    node_meta(136, ann::kNodeTagMetadataKey, "weight");
    node_meta(136, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(136, ann::kReleaseAfterMetadataKey, "layer_1_normed_half");
    node_meta(137, ann::kNodeTagMetadataKey, "weight");
    node_meta(138, ann::kNodeTagMetadataKey, "weight");
    node_meta(138, ann::kReleaseAfterMetadataKey, "layer_1_q_mm");
    node_meta(139, ann::kNodeTagMetadataKey, "weight");
    node_meta(139, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(139, ann::kReleaseAfterMetadataKey, "layer_1_q_f32");
    node_meta(140, ann::kNodeTagMetadataKey, "weight");
    node_meta(141, ann::kNodeTagMetadataKey, "weight");
    node_meta(141, ann::kReleaseAfterMetadataKey, "layer_1_pow_q");
    node_meta(142, ann::kNodeTagMetadataKey, "weight");
    node_meta(142, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(142, ann::kReleaseAfterMetadataKey, "layer_1_mean_q");
    node_meta(143, ann::kNodeTagMetadataKey, "weight");
    node_meta(143, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(143, ann::kReleaseAfterMetadataKey, "layer_1_add_q");
    node_meta(144, ann::kNodeTagMetadataKey, "weight");
    node_meta(144, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(144, ann::kReleaseAfterMetadataKey, "layer_1_sqrt_q");
    node_meta(145, ann::kNodeTagMetadataKey, "weight");
    node_meta(145, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(145, ann::kReleaseAfterMetadataKey, "layer_1_q_4d;layer_1_rsqrt_q");
    node_meta(146, ann::kNodeTagMetadataKey, "weight");
    node_meta(146, ann::kReleaseAfterMetadataKey, "layer_1_mul_q");
    node_meta(147, ann::kNodeTagMetadataKey, "weight");
    node_meta(147, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(147, ann::kReleaseAfterMetadataKey, "layer_1_q_normed_half");
    node_meta(148, ann::kNodeTagMetadataKey, "weight");
    node_meta(148, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(148, ann::kReleaseAfterMetadataKey, "layer_1_q_normed");
    node_meta(149, ann::kNodeTagMetadataKey, "weight");
    node_meta(150, ann::kNodeTagMetadataKey, "weight");
    node_meta(150, ann::kReleaseAfterMetadataKey, "layer_1_k_mm");
    node_meta(151, ann::kNodeTagMetadataKey, "weight");
    node_meta(151, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(151, ann::kReleaseAfterMetadataKey, "layer_1_k_f32");
    node_meta(152, ann::kNodeTagMetadataKey, "weight");
    node_meta(153, ann::kNodeTagMetadataKey, "weight");
    node_meta(153, ann::kReleaseAfterMetadataKey, "layer_1_pow_k");
    node_meta(154, ann::kNodeTagMetadataKey, "weight");
    node_meta(154, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(154, ann::kReleaseAfterMetadataKey, "layer_1_mean_k");
    node_meta(155, ann::kNodeTagMetadataKey, "weight");
    node_meta(155, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(155, ann::kReleaseAfterMetadataKey, "layer_1_add_k");
    node_meta(156, ann::kNodeTagMetadataKey, "weight");
    node_meta(156, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(156, ann::kReleaseAfterMetadataKey, "layer_1_sqrt_k");
    node_meta(157, ann::kNodeTagMetadataKey, "weight");
    node_meta(157, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(157, ann::kReleaseAfterMetadataKey, "layer_1_k_4d;layer_1_rsqrt_k");
    node_meta(158, ann::kNodeTagMetadataKey, "weight");
    node_meta(158, ann::kReleaseAfterMetadataKey, "layer_1_mul_k");
    node_meta(159, ann::kNodeTagMetadataKey, "weight");
    node_meta(159, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(159, ann::kReleaseAfterMetadataKey, "layer_1_k_normed_half");
    node_meta(160, ann::kNodeTagMetadataKey, "weight");
    node_meta(160, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(160, ann::kReleaseAfterMetadataKey, "layer_1_k_normed");
    node_meta(161, ann::kNodeTagMetadataKey, "weight");
    node_meta(161, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(161, ann::kReleaseAfterMetadataKey, "layer_1_normed");
    node_meta(162, ann::kNodeTagMetadataKey, "weight");
    node_meta(162, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(162, ann::kReleaseAfterMetadataKey, "layer_1_v_mm");
    node_meta(163, ann::kNodeTagMetadataKey, "weight");
    node_meta(163, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(163, ann::kReleaseAfterMetadataKey, "layer_1_v_4d");
    node_meta(164, ann::kNodeTagMetadataKey, "weight");
    node_meta(165, ann::kNodeTagMetadataKey, "weight");
    node_meta(165, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(165, ann::kReleaseAfterMetadataKey, "layer_1_q_half1");
    node_meta(166, ann::kNodeTagMetadataKey, "weight");
    node_meta(166, ann::kReleaseAfterMetadataKey, "layer_1_neg_q;layer_1_q_half0");
    node_meta(167, ann::kNodeTagMetadataKey, "weight");
    node_meta(167, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(167, ann::kReleaseAfterMetadataKey, "layer_1_q_T");
    node_meta(168, ann::kNodeTagMetadataKey, "weight");
    node_meta(168, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(168, ann::kReleaseAfterMetadataKey, "layer_1_q_rot");
    node_meta(169, ann::kNodeTagMetadataKey, "weight");
    node_meta(169, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(169, ann::kReleaseAfterMetadataKey, "layer_1_q_cos;layer_1_q_sin");
    node_meta(170, ann::kNodeTagMetadataKey, "weight");
    node_meta(171, ann::kNodeTagMetadataKey, "weight");
    node_meta(171, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(171, ann::kReleaseAfterMetadataKey, "layer_1_k_half1");
    node_meta(172, ann::kNodeTagMetadataKey, "weight");
    node_meta(172, ann::kReleaseAfterMetadataKey, "layer_1_neg_k;layer_1_k_half0");
    node_meta(173, ann::kNodeTagMetadataKey, "weight");
    node_meta(173, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(173, ann::kReleaseAfterMetadataKey, "layer_1_k_T");
    node_meta(174, ann::kNodeTagMetadataKey, "weight");
    node_meta(174, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(174, ann::kReleaseAfterMetadataKey, "layer_1_k_rot");
    node_meta(175, ann::kNodeTagMetadataKey, "weight");
    node_meta(175, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(175, ann::kReleaseAfterMetadataKey, "layer_1_k_cos;layer_1_k_sin");
    node_meta(176, ann::kNodeTagMetadataKey, "weight");
    node_meta(176, ann::kReleaseAfterMetadataKey, "layer_1_k_rope");
    node_meta(177, ann::kNodeTagMetadataKey, "weight");
    node_meta(177, ann::kReleaseAfterMetadataKey, "layer_1_v_T");
    node_meta(178, ann::kNodeTagMetadataKey, "weight");
    node_meta(179, ann::kNodeTagMetadataKey, "weight");
    node_meta(179, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(179, ann::kReleaseAfterMetadataKey, "layer_1_scaled_k");
    node_meta(180, ann::kNodeTagMetadataKey, "weight");
    node_meta(181, ann::kNodeTagMetadataKey, "weight");
    node_meta(181, ann::kReleaseAfterMetadataKey, "layer_1_scaled_k_unsq");
    node_meta(182, ann::kNodeTagMetadataKey, "weight");
    node_meta(182, ann::kReleaseAfterMetadataKey, "layer_1_v_unsq");
    node_meta(183, ann::kNodeTagMetadataKey, "weight");
    node_meta(183, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(183, ann::kReleaseAfterMetadataKey, "layer_1_scaled_k_exp");
    node_meta(184, ann::kNodeTagMetadataKey, "weight");
    node_meta(184, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(184, ann::kReleaseAfterMetadataKey, "layer_1_v_exp");
    node_meta(185, ann::kNodeTagMetadataKey, "weight");
    node_meta(185, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(185, ann::kReleaseAfterMetadataKey, "layer_1_q_rope");
    node_meta(186, ann::kNodeTagMetadataKey, "weight");
    node_meta(186, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(186, ann::kReleaseAfterMetadataKey, "layer_1_k_gqa");
    node_meta(187, ann::kNodeTagMetadataKey, "weight");
    node_meta(187, ann::kReleaseAfterMetadataKey, "layer_1_scaled_q;layer_1_k_gqa_T");
    node_meta(188, ann::kNodeTagMetadataKey, "weight");
    node_meta(188, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(188, ann::kReleaseAfterMetadataKey, "layer_1_attn_scores");
    node_meta(189, ann::kNodeTagMetadataKey, "weight");
    node_meta(189, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(189, ann::kReleaseAfterMetadataKey, "layer_1_masked");
    node_meta(190, ann::kNodeTagMetadataKey, "weight");
    node_meta(191, ann::kNodeTagMetadataKey, "weight");
    node_meta(191, ann::kInPlaceReuseMetadataKey, "0:2:equal");
    node_meta(191, ann::kReleaseAfterMetadataKey, "layer_1_is_nan;layer_1_softmax");
    node_meta(192, ann::kNodeTagMetadataKey, "weight");
    node_meta(192, ann::kReleaseAfterMetadataKey, "layer_1_attn_w;layer_1_v_gqa");
    node_meta(193, ann::kNodeTagMetadataKey, "weight");
    node_meta(193, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(193, ann::kReleaseAfterMetadataKey, "layer_1_attn_out");
    node_meta(194, ann::kNodeTagMetadataKey, "weight");
    node_meta(194, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(194, ann::kReleaseAfterMetadataKey, "layer_1_attn_out_T");
    node_meta(195, ann::kNodeTagMetadataKey, "weight");
    node_meta(195, ann::kReleaseAfterMetadataKey, "layer_1_attn_2d");
    node_meta(196, ann::kNodeTagMetadataKey, "weight");
    node_meta(196, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(196, ann::kReleaseAfterMetadataKey, "layer_0_out;layer_1_attn_proj");
    node_meta(197, ann::kNodeTagMetadataKey, "weight");
    node_meta(198, ann::kNodeTagMetadataKey, "weight");
    node_meta(199, ann::kNodeTagMetadataKey, "weight");
    node_meta(199, ann::kReleaseAfterMetadataKey, "layer_1_pow_post");
    node_meta(200, ann::kNodeTagMetadataKey, "weight");
    node_meta(200, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(200, ann::kReleaseAfterMetadataKey, "layer_1_mean_post");
    node_meta(201, ann::kNodeTagMetadataKey, "weight");
    node_meta(201, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(201, ann::kReleaseAfterMetadataKey, "layer_1_add_post");
    node_meta(202, ann::kNodeTagMetadataKey, "weight");
    node_meta(202, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(202, ann::kReleaseAfterMetadataKey, "layer_1_sqrt_post");
    node_meta(203, ann::kNodeTagMetadataKey, "weight");
    node_meta(203, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(203, ann::kReleaseAfterMetadataKey, "layer_1_post_f32;layer_1_rsqrt_post");
    node_meta(204, ann::kNodeTagMetadataKey, "weight");
    node_meta(204, ann::kReleaseAfterMetadataKey, "layer_1_mul_post");
    node_meta(205, ann::kNodeTagMetadataKey, "weight");
    node_meta(205, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(205, ann::kReleaseAfterMetadataKey, "layer_1_post_half");
    node_meta(206, ann::kNodeTagMetadataKey, "weight");
    node_meta(207, ann::kNodeTagMetadataKey, "weight");
    node_meta(208, ann::kNodeTagMetadataKey, "weight");
    node_meta(208, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(208, ann::kReleaseAfterMetadataKey, "layer_1_gate;layer_1_gate_act");
    node_meta(209, ann::kNodeTagMetadataKey, "weight");
    node_meta(209, ann::kReleaseAfterMetadataKey, "layer_1_mlp_in");
    node_meta(210, ann::kNodeTagMetadataKey, "weight");
    node_meta(210, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(210, ann::kReleaseAfterMetadataKey, "layer_1_silu;layer_1_up");
    node_meta(211, ann::kNodeTagMetadataKey, "weight");
    node_meta(211, ann::kReleaseAfterMetadataKey, "layer_1_swiglu");
    node_meta(212, ann::kNodeTagMetadataKey, "weight");
    node_meta(212, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(212, ann::kReleaseAfterMetadataKey, "layer_1_resid_attn;layer_1_down");
    node_meta(213, ann::kNodeTagMetadataKey, "weight");
    node_meta(214, ann::kNodeTagMetadataKey, "weight");
    node_meta(215, ann::kNodeTagMetadataKey, "weight");
    node_meta(215, ann::kReleaseAfterMetadataKey, "layer_2_pow_pre");
    node_meta(216, ann::kNodeTagMetadataKey, "weight");
    node_meta(216, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(216, ann::kReleaseAfterMetadataKey, "layer_2_mean_pre");
    node_meta(217, ann::kNodeTagMetadataKey, "weight");
    node_meta(217, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(217, ann::kReleaseAfterMetadataKey, "layer_2_add_pre");
    node_meta(218, ann::kNodeTagMetadataKey, "weight");
    node_meta(218, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(218, ann::kReleaseAfterMetadataKey, "layer_2_sqrt_pre");
    node_meta(219, ann::kNodeTagMetadataKey, "weight");
    node_meta(219, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(219, ann::kReleaseAfterMetadataKey, "layer_2_f32;layer_2_rsqrt_pre");
    node_meta(220, ann::kNodeTagMetadataKey, "weight");
    node_meta(220, ann::kReleaseAfterMetadataKey, "layer_2_mul_pre");
    node_meta(221, ann::kNodeTagMetadataKey, "weight");
    node_meta(221, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(221, ann::kReleaseAfterMetadataKey, "layer_2_normed_half");
    node_meta(222, ann::kNodeTagMetadataKey, "weight");
    node_meta(223, ann::kNodeTagMetadataKey, "weight");
    node_meta(223, ann::kReleaseAfterMetadataKey, "layer_2_q_mm");
    node_meta(224, ann::kNodeTagMetadataKey, "weight");
    node_meta(224, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(224, ann::kReleaseAfterMetadataKey, "layer_2_q_f32");
    node_meta(225, ann::kNodeTagMetadataKey, "weight");
    node_meta(226, ann::kNodeTagMetadataKey, "weight");
    node_meta(226, ann::kReleaseAfterMetadataKey, "layer_2_pow_q");
    node_meta(227, ann::kNodeTagMetadataKey, "weight");
    node_meta(227, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(227, ann::kReleaseAfterMetadataKey, "layer_2_mean_q");
    node_meta(228, ann::kNodeTagMetadataKey, "weight");
    node_meta(228, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(228, ann::kReleaseAfterMetadataKey, "layer_2_add_q");
    node_meta(229, ann::kNodeTagMetadataKey, "weight");
    node_meta(229, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(229, ann::kReleaseAfterMetadataKey, "layer_2_sqrt_q");
    node_meta(230, ann::kNodeTagMetadataKey, "weight");
    node_meta(230, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(230, ann::kReleaseAfterMetadataKey, "layer_2_q_4d;layer_2_rsqrt_q");
    node_meta(231, ann::kNodeTagMetadataKey, "weight");
    node_meta(231, ann::kReleaseAfterMetadataKey, "layer_2_mul_q");
    node_meta(232, ann::kNodeTagMetadataKey, "weight");
    node_meta(232, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(232, ann::kReleaseAfterMetadataKey, "layer_2_q_normed_half");
    node_meta(233, ann::kNodeTagMetadataKey, "weight");
    node_meta(233, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(233, ann::kReleaseAfterMetadataKey, "layer_2_q_normed");
    node_meta(234, ann::kNodeTagMetadataKey, "weight");
    node_meta(235, ann::kNodeTagMetadataKey, "weight");
    node_meta(235, ann::kReleaseAfterMetadataKey, "layer_2_k_mm");
    node_meta(236, ann::kNodeTagMetadataKey, "weight");
    node_meta(236, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(236, ann::kReleaseAfterMetadataKey, "layer_2_k_f32");
    node_meta(237, ann::kNodeTagMetadataKey, "weight");
    node_meta(238, ann::kNodeTagMetadataKey, "weight");
    node_meta(238, ann::kReleaseAfterMetadataKey, "layer_2_pow_k");
    node_meta(239, ann::kNodeTagMetadataKey, "weight");
    node_meta(239, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(239, ann::kReleaseAfterMetadataKey, "layer_2_mean_k");
    node_meta(240, ann::kNodeTagMetadataKey, "weight");
    node_meta(240, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(240, ann::kReleaseAfterMetadataKey, "layer_2_add_k");
    node_meta(241, ann::kNodeTagMetadataKey, "weight");
    node_meta(241, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(241, ann::kReleaseAfterMetadataKey, "layer_2_sqrt_k");
    node_meta(242, ann::kNodeTagMetadataKey, "weight");
    node_meta(242, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(242, ann::kReleaseAfterMetadataKey, "layer_2_k_4d;layer_2_rsqrt_k");
    node_meta(243, ann::kNodeTagMetadataKey, "weight");
    node_meta(243, ann::kReleaseAfterMetadataKey, "layer_2_mul_k");
    node_meta(244, ann::kNodeTagMetadataKey, "weight");
    node_meta(244, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(244, ann::kReleaseAfterMetadataKey, "layer_2_k_normed_half");
    node_meta(245, ann::kNodeTagMetadataKey, "weight");
    node_meta(245, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(245, ann::kReleaseAfterMetadataKey, "layer_2_k_normed");
    node_meta(246, ann::kNodeTagMetadataKey, "weight");
    node_meta(246, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(246, ann::kReleaseAfterMetadataKey, "layer_2_normed");
    node_meta(247, ann::kNodeTagMetadataKey, "weight");
    node_meta(247, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(247, ann::kReleaseAfterMetadataKey, "layer_2_v_mm");
    node_meta(248, ann::kNodeTagMetadataKey, "weight");
    node_meta(248, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(248, ann::kReleaseAfterMetadataKey, "layer_2_v_4d");
    node_meta(249, ann::kNodeTagMetadataKey, "weight");
    node_meta(250, ann::kNodeTagMetadataKey, "weight");
    node_meta(250, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(250, ann::kReleaseAfterMetadataKey, "layer_2_q_half1");
    node_meta(251, ann::kNodeTagMetadataKey, "weight");
    node_meta(251, ann::kReleaseAfterMetadataKey, "layer_2_neg_q;layer_2_q_half0");
    node_meta(252, ann::kNodeTagMetadataKey, "weight");
    node_meta(252, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(252, ann::kReleaseAfterMetadataKey, "layer_2_q_T");
    node_meta(253, ann::kNodeTagMetadataKey, "weight");
    node_meta(253, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(253, ann::kReleaseAfterMetadataKey, "layer_2_q_rot");
    node_meta(254, ann::kNodeTagMetadataKey, "weight");
    node_meta(254, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(254, ann::kReleaseAfterMetadataKey, "layer_2_q_cos;layer_2_q_sin");
    node_meta(255, ann::kNodeTagMetadataKey, "weight");
    node_meta(256, ann::kNodeTagMetadataKey, "weight");
    node_meta(256, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(256, ann::kReleaseAfterMetadataKey, "layer_2_k_half1");
    node_meta(257, ann::kNodeTagMetadataKey, "weight");
    node_meta(257, ann::kReleaseAfterMetadataKey, "layer_2_neg_k;layer_2_k_half0");
    node_meta(258, ann::kNodeTagMetadataKey, "weight");
    node_meta(258, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(258, ann::kReleaseAfterMetadataKey, "layer_2_k_T");
    node_meta(259, ann::kNodeTagMetadataKey, "weight");
    node_meta(259, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(259, ann::kReleaseAfterMetadataKey, "layer_2_k_rot");
    node_meta(260, ann::kNodeTagMetadataKey, "weight");
    node_meta(260, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(260, ann::kReleaseAfterMetadataKey, "layer_2_k_cos;layer_2_k_sin");
    node_meta(261, ann::kNodeTagMetadataKey, "weight");
    node_meta(261, ann::kReleaseAfterMetadataKey, "layer_2_k_rope");
    node_meta(262, ann::kNodeTagMetadataKey, "weight");
    node_meta(262, ann::kReleaseAfterMetadataKey, "layer_2_v_T");
    node_meta(263, ann::kNodeTagMetadataKey, "weight");
    node_meta(264, ann::kNodeTagMetadataKey, "weight");
    node_meta(264, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(264, ann::kReleaseAfterMetadataKey, "layer_2_scaled_k");
    node_meta(265, ann::kNodeTagMetadataKey, "weight");
    node_meta(266, ann::kNodeTagMetadataKey, "weight");
    node_meta(266, ann::kReleaseAfterMetadataKey, "layer_2_scaled_k_unsq");
    node_meta(267, ann::kNodeTagMetadataKey, "weight");
    node_meta(267, ann::kReleaseAfterMetadataKey, "layer_2_v_unsq");
    node_meta(268, ann::kNodeTagMetadataKey, "weight");
    node_meta(268, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(268, ann::kReleaseAfterMetadataKey, "layer_2_scaled_k_exp");
    node_meta(269, ann::kNodeTagMetadataKey, "weight");
    node_meta(269, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(269, ann::kReleaseAfterMetadataKey, "layer_2_v_exp");
    node_meta(270, ann::kNodeTagMetadataKey, "weight");
    node_meta(270, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(270, ann::kReleaseAfterMetadataKey, "layer_2_q_rope");
    node_meta(271, ann::kNodeTagMetadataKey, "weight");
    node_meta(271, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(271, ann::kReleaseAfterMetadataKey, "layer_2_k_gqa");
    node_meta(272, ann::kNodeTagMetadataKey, "weight");
    node_meta(272, ann::kReleaseAfterMetadataKey, "layer_2_scaled_q;layer_2_k_gqa_T");
    node_meta(273, ann::kNodeTagMetadataKey, "weight");
    node_meta(273, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(273, ann::kReleaseAfterMetadataKey, "layer_2_attn_scores");
    node_meta(274, ann::kNodeTagMetadataKey, "weight");
    node_meta(274, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(274, ann::kReleaseAfterMetadataKey, "layer_2_masked");
    node_meta(275, ann::kNodeTagMetadataKey, "weight");
    node_meta(276, ann::kNodeTagMetadataKey, "weight");
    node_meta(276, ann::kInPlaceReuseMetadataKey, "0:2:equal");
    node_meta(276, ann::kReleaseAfterMetadataKey, "layer_2_is_nan;layer_2_softmax");
    node_meta(277, ann::kNodeTagMetadataKey, "weight");
    node_meta(277, ann::kReleaseAfterMetadataKey, "layer_2_attn_w;layer_2_v_gqa");
    node_meta(278, ann::kNodeTagMetadataKey, "weight");
    node_meta(278, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(278, ann::kReleaseAfterMetadataKey, "layer_2_attn_out");
    node_meta(279, ann::kNodeTagMetadataKey, "weight");
    node_meta(279, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(279, ann::kReleaseAfterMetadataKey, "layer_2_attn_out_T");
    node_meta(280, ann::kNodeTagMetadataKey, "weight");
    node_meta(280, ann::kReleaseAfterMetadataKey, "layer_2_attn_2d");
    node_meta(281, ann::kNodeTagMetadataKey, "weight");
    node_meta(281, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(281, ann::kReleaseAfterMetadataKey, "layer_1_out;layer_2_attn_proj");
    node_meta(282, ann::kNodeTagMetadataKey, "weight");
    node_meta(283, ann::kNodeTagMetadataKey, "weight");
    node_meta(284, ann::kNodeTagMetadataKey, "weight");
    node_meta(284, ann::kReleaseAfterMetadataKey, "layer_2_pow_post");
    node_meta(285, ann::kNodeTagMetadataKey, "weight");
    node_meta(285, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(285, ann::kReleaseAfterMetadataKey, "layer_2_mean_post");
    node_meta(286, ann::kNodeTagMetadataKey, "weight");
    node_meta(286, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(286, ann::kReleaseAfterMetadataKey, "layer_2_add_post");
    node_meta(287, ann::kNodeTagMetadataKey, "weight");
    node_meta(287, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(287, ann::kReleaseAfterMetadataKey, "layer_2_sqrt_post");
    node_meta(288, ann::kNodeTagMetadataKey, "weight");
    node_meta(288, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(288, ann::kReleaseAfterMetadataKey, "layer_2_post_f32;layer_2_rsqrt_post");
    node_meta(289, ann::kNodeTagMetadataKey, "weight");
    node_meta(289, ann::kReleaseAfterMetadataKey, "layer_2_mul_post");
    node_meta(290, ann::kNodeTagMetadataKey, "weight");
    node_meta(290, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(290, ann::kReleaseAfterMetadataKey, "layer_2_post_half");
    node_meta(291, ann::kNodeTagMetadataKey, "weight");
    node_meta(292, ann::kNodeTagMetadataKey, "weight");
    node_meta(293, ann::kNodeTagMetadataKey, "weight");
    node_meta(293, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(293, ann::kReleaseAfterMetadataKey, "layer_2_gate;layer_2_gate_act");
    node_meta(294, ann::kNodeTagMetadataKey, "weight");
    node_meta(294, ann::kReleaseAfterMetadataKey, "layer_2_mlp_in");
    node_meta(295, ann::kNodeTagMetadataKey, "weight");
    node_meta(295, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(295, ann::kReleaseAfterMetadataKey, "layer_2_silu;layer_2_up");
    node_meta(296, ann::kNodeTagMetadataKey, "weight");
    node_meta(296, ann::kReleaseAfterMetadataKey, "layer_2_swiglu");
    node_meta(297, ann::kNodeTagMetadataKey, "weight");
    node_meta(297, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(297, ann::kReleaseAfterMetadataKey, "layer_2_resid_attn;layer_2_down");
    node_meta(298, ann::kNodeTagMetadataKey, "weight");
    node_meta(299, ann::kNodeTagMetadataKey, "weight");
    node_meta(300, ann::kNodeTagMetadataKey, "weight");
    node_meta(300, ann::kReleaseAfterMetadataKey, "layer_3_pow_pre");
    node_meta(301, ann::kNodeTagMetadataKey, "weight");
    node_meta(301, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(301, ann::kReleaseAfterMetadataKey, "layer_3_mean_pre");
    node_meta(302, ann::kNodeTagMetadataKey, "weight");
    node_meta(302, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(302, ann::kReleaseAfterMetadataKey, "layer_3_add_pre");
    node_meta(303, ann::kNodeTagMetadataKey, "weight");
    node_meta(303, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(303, ann::kReleaseAfterMetadataKey, "layer_3_sqrt_pre");
    node_meta(304, ann::kNodeTagMetadataKey, "weight");
    node_meta(304, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(304, ann::kReleaseAfterMetadataKey, "layer_3_f32;layer_3_rsqrt_pre");
    node_meta(305, ann::kNodeTagMetadataKey, "weight");
    node_meta(305, ann::kReleaseAfterMetadataKey, "layer_3_mul_pre");
    node_meta(306, ann::kNodeTagMetadataKey, "weight");
    node_meta(306, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(306, ann::kReleaseAfterMetadataKey, "layer_3_normed_half");
    node_meta(307, ann::kNodeTagMetadataKey, "weight");
    node_meta(308, ann::kNodeTagMetadataKey, "weight");
    node_meta(308, ann::kReleaseAfterMetadataKey, "layer_3_q_mm");
    node_meta(309, ann::kNodeTagMetadataKey, "weight");
    node_meta(309, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(309, ann::kReleaseAfterMetadataKey, "layer_3_q_f32");
    node_meta(310, ann::kNodeTagMetadataKey, "weight");
    node_meta(311, ann::kNodeTagMetadataKey, "weight");
    node_meta(311, ann::kReleaseAfterMetadataKey, "layer_3_pow_q");
    node_meta(312, ann::kNodeTagMetadataKey, "weight");
    node_meta(312, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(312, ann::kReleaseAfterMetadataKey, "layer_3_mean_q");
    node_meta(313, ann::kNodeTagMetadataKey, "weight");
    node_meta(313, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(313, ann::kReleaseAfterMetadataKey, "layer_3_add_q");
    node_meta(314, ann::kNodeTagMetadataKey, "weight");
    node_meta(314, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(314, ann::kReleaseAfterMetadataKey, "layer_3_sqrt_q");
    node_meta(315, ann::kNodeTagMetadataKey, "weight");
    node_meta(315, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(315, ann::kReleaseAfterMetadataKey, "layer_3_q_4d;layer_3_rsqrt_q");
    node_meta(316, ann::kNodeTagMetadataKey, "weight");
    node_meta(316, ann::kReleaseAfterMetadataKey, "layer_3_mul_q");
    node_meta(317, ann::kNodeTagMetadataKey, "weight");
    node_meta(317, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(317, ann::kReleaseAfterMetadataKey, "layer_3_q_normed_half");
    node_meta(318, ann::kNodeTagMetadataKey, "weight");
    node_meta(318, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(318, ann::kReleaseAfterMetadataKey, "layer_3_q_normed");
    node_meta(319, ann::kNodeTagMetadataKey, "weight");
    node_meta(320, ann::kNodeTagMetadataKey, "weight");
    node_meta(320, ann::kReleaseAfterMetadataKey, "layer_3_k_mm");
    node_meta(321, ann::kNodeTagMetadataKey, "weight");
    node_meta(321, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(321, ann::kReleaseAfterMetadataKey, "layer_3_k_f32");
    node_meta(322, ann::kNodeTagMetadataKey, "weight");
    node_meta(323, ann::kNodeTagMetadataKey, "weight");
    node_meta(323, ann::kReleaseAfterMetadataKey, "layer_3_pow_k");
    node_meta(324, ann::kNodeTagMetadataKey, "weight");
    node_meta(324, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(324, ann::kReleaseAfterMetadataKey, "layer_3_mean_k");
    node_meta(325, ann::kNodeTagMetadataKey, "weight");
    node_meta(325, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(325, ann::kReleaseAfterMetadataKey, "layer_3_add_k");
    node_meta(326, ann::kNodeTagMetadataKey, "weight");
    node_meta(326, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(326, ann::kReleaseAfterMetadataKey, "layer_3_sqrt_k");
    node_meta(327, ann::kNodeTagMetadataKey, "weight");
    node_meta(327, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(327, ann::kReleaseAfterMetadataKey, "layer_3_k_4d;layer_3_rsqrt_k");
    node_meta(328, ann::kNodeTagMetadataKey, "weight");
    node_meta(328, ann::kReleaseAfterMetadataKey, "layer_3_mul_k");
    node_meta(329, ann::kNodeTagMetadataKey, "weight");
    node_meta(329, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(329, ann::kReleaseAfterMetadataKey, "layer_3_k_normed_half");
    node_meta(330, ann::kNodeTagMetadataKey, "weight");
    node_meta(330, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(330, ann::kReleaseAfterMetadataKey, "layer_3_k_normed");
    node_meta(331, ann::kNodeTagMetadataKey, "weight");
    node_meta(331, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(331, ann::kReleaseAfterMetadataKey, "layer_3_normed");
    node_meta(332, ann::kNodeTagMetadataKey, "weight");
    node_meta(332, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(332, ann::kReleaseAfterMetadataKey, "layer_3_v_mm");
    node_meta(333, ann::kNodeTagMetadataKey, "weight");
    node_meta(333, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(333, ann::kReleaseAfterMetadataKey, "layer_3_v_4d");
    node_meta(334, ann::kNodeTagMetadataKey, "weight");
    node_meta(335, ann::kNodeTagMetadataKey, "weight");
    node_meta(335, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(335, ann::kReleaseAfterMetadataKey, "layer_3_q_half1");
    node_meta(336, ann::kNodeTagMetadataKey, "weight");
    node_meta(336, ann::kReleaseAfterMetadataKey, "layer_3_neg_q;layer_3_q_half0");
    node_meta(337, ann::kNodeTagMetadataKey, "weight");
    node_meta(337, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(337, ann::kReleaseAfterMetadataKey, "layer_3_q_T");
    node_meta(338, ann::kNodeTagMetadataKey, "weight");
    node_meta(338, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(338, ann::kReleaseAfterMetadataKey, "layer_3_q_rot");
    node_meta(339, ann::kNodeTagMetadataKey, "weight");
    node_meta(339, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(339, ann::kReleaseAfterMetadataKey, "layer_3_q_cos;layer_3_q_sin");
    node_meta(340, ann::kNodeTagMetadataKey, "weight");
    node_meta(341, ann::kNodeTagMetadataKey, "weight");
    node_meta(341, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(341, ann::kReleaseAfterMetadataKey, "layer_3_k_half1");
    node_meta(342, ann::kNodeTagMetadataKey, "weight");
    node_meta(342, ann::kReleaseAfterMetadataKey, "layer_3_neg_k;layer_3_k_half0");
    node_meta(343, ann::kNodeTagMetadataKey, "weight");
    node_meta(343, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(343, ann::kReleaseAfterMetadataKey, "layer_3_k_T;unsqueeze_16");
    node_meta(344, ann::kNodeTagMetadataKey, "weight");
    node_meta(344, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(344, ann::kReleaseAfterMetadataKey, "layer_3_k_rot;unsqueeze_17");
    node_meta(345, ann::kNodeTagMetadataKey, "weight");
    node_meta(345, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(345, ann::kReleaseAfterMetadataKey, "layer_3_k_cos;layer_3_k_sin");
    node_meta(346, ann::kNodeTagMetadataKey, "weight");
    node_meta(346, ann::kReleaseAfterMetadataKey, "layer_3_k_rope");
    node_meta(347, ann::kNodeTagMetadataKey, "weight");
    node_meta(347, ann::kReleaseAfterMetadataKey, "layer_3_v_T");
    node_meta(348, ann::kNodeTagMetadataKey, "weight");
    node_meta(349, ann::kNodeTagMetadataKey, "weight");
    node_meta(349, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(349, ann::kReleaseAfterMetadataKey, "layer_3_scaled_k");
    node_meta(350, ann::kNodeTagMetadataKey, "weight");
    node_meta(351, ann::kNodeTagMetadataKey, "weight");
    node_meta(351, ann::kReleaseAfterMetadataKey, "layer_3_scaled_k_unsq");
    node_meta(352, ann::kNodeTagMetadataKey, "weight");
    node_meta(352, ann::kReleaseAfterMetadataKey, "layer_3_v_unsq");
    node_meta(353, ann::kNodeTagMetadataKey, "weight");
    node_meta(353, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(353, ann::kReleaseAfterMetadataKey, "layer_3_scaled_k_exp");
    node_meta(354, ann::kNodeTagMetadataKey, "weight");
    node_meta(354, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(354, ann::kReleaseAfterMetadataKey, "layer_3_v_exp");
    node_meta(355, ann::kNodeTagMetadataKey, "weight");
    node_meta(355, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(355, ann::kReleaseAfterMetadataKey, "layer_3_q_rope");
    node_meta(356, ann::kNodeTagMetadataKey, "weight");
    node_meta(356, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(356, ann::kReleaseAfterMetadataKey, "layer_3_k_gqa");
    node_meta(357, ann::kNodeTagMetadataKey, "weight");
    node_meta(357, ann::kReleaseAfterMetadataKey, "layer_3_scaled_q;layer_3_k_gqa_T");
    node_meta(358, ann::kNodeTagMetadataKey, "weight");
    node_meta(358, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(358, ann::kReleaseAfterMetadataKey, "and_2;layer_3_attn_scores");
    node_meta(359, ann::kNodeTagMetadataKey, "weight");
    node_meta(359, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(359, ann::kReleaseAfterMetadataKey, "layer_3_masked");
    node_meta(360, ann::kNodeTagMetadataKey, "weight");
    node_meta(361, ann::kNodeTagMetadataKey, "weight");
    node_meta(361, ann::kInPlaceReuseMetadataKey, "0:2:equal");
    node_meta(361, ann::kReleaseAfterMetadataKey, "layer_3_is_nan;layer_3_softmax");
    node_meta(362, ann::kNodeTagMetadataKey, "weight");
    node_meta(362, ann::kReleaseAfterMetadataKey, "layer_3_attn_w;layer_3_v_gqa");
    node_meta(363, ann::kNodeTagMetadataKey, "weight");
    node_meta(363, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(363, ann::kReleaseAfterMetadataKey, "layer_3_attn_out");
    node_meta(364, ann::kNodeTagMetadataKey, "weight");
    node_meta(364, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(364, ann::kReleaseAfterMetadataKey, "layer_3_attn_out_T");
    node_meta(365, ann::kNodeTagMetadataKey, "weight");
    node_meta(365, ann::kReleaseAfterMetadataKey, "layer_3_attn_2d");
    node_meta(366, ann::kNodeTagMetadataKey, "weight");
    node_meta(366, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(366, ann::kReleaseAfterMetadataKey, "layer_2_out;layer_3_attn_proj");
    node_meta(367, ann::kNodeTagMetadataKey, "weight");
    node_meta(368, ann::kNodeTagMetadataKey, "weight");
    node_meta(369, ann::kNodeTagMetadataKey, "weight");
    node_meta(369, ann::kReleaseAfterMetadataKey, "layer_3_pow_post");
    node_meta(370, ann::kNodeTagMetadataKey, "weight");
    node_meta(370, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(370, ann::kReleaseAfterMetadataKey, "layer_3_mean_post");
    node_meta(371, ann::kNodeTagMetadataKey, "weight");
    node_meta(371, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(371, ann::kReleaseAfterMetadataKey, "layer_3_add_post");
    node_meta(372, ann::kNodeTagMetadataKey, "weight");
    node_meta(372, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(372, ann::kReleaseAfterMetadataKey, "layer_3_sqrt_post");
    node_meta(373, ann::kNodeTagMetadataKey, "weight");
    node_meta(373, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(373, ann::kReleaseAfterMetadataKey, "layer_3_post_f32;layer_3_rsqrt_post");
    node_meta(374, ann::kNodeTagMetadataKey, "weight");
    node_meta(374, ann::kReleaseAfterMetadataKey, "layer_3_mul_post");
    node_meta(375, ann::kNodeTagMetadataKey, "weight");
    node_meta(375, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(375, ann::kReleaseAfterMetadataKey, "layer_3_post_half");
    node_meta(376, ann::kNodeTagMetadataKey, "weight");
    node_meta(377, ann::kNodeTagMetadataKey, "weight");
    node_meta(378, ann::kNodeTagMetadataKey, "weight");
    node_meta(378, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(378, ann::kReleaseAfterMetadataKey, "layer_3_gate;layer_3_gate_act");
    node_meta(379, ann::kNodeTagMetadataKey, "weight");
    node_meta(379, ann::kReleaseAfterMetadataKey, "layer_3_mlp_in");
    node_meta(380, ann::kNodeTagMetadataKey, "weight");
    node_meta(380, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(380, ann::kReleaseAfterMetadataKey, "layer_3_silu;layer_3_up");
    node_meta(381, ann::kNodeTagMetadataKey, "weight");
    node_meta(381, ann::kReleaseAfterMetadataKey, "layer_3_swiglu");
    node_meta(382, ann::kNodeTagMetadataKey, "weight");
    node_meta(382, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(382, ann::kReleaseAfterMetadataKey, "layer_3_resid_attn;layer_3_down");
    node_meta(383, ann::kNodeTagMetadataKey, "weight");
    node_meta(383, ann::kReleaseAfterMetadataKey, "layer_3_out");
    node_meta(384, ann::kNodeTagMetadataKey, "weight");
    node_meta(385, ann::kNodeTagMetadataKey, "weight");
    node_meta(385, ann::kReleaseAfterMetadataKey, "final_pow");
    node_meta(386, ann::kNodeTagMetadataKey, "weight");
    node_meta(386, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(386, ann::kReleaseAfterMetadataKey, "final_mean");
    node_meta(387, ann::kNodeTagMetadataKey, "weight");
    node_meta(387, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(387, ann::kReleaseAfterMetadataKey, "final_add");
    node_meta(388, ann::kNodeTagMetadataKey, "weight");
    node_meta(388, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(388, ann::kReleaseAfterMetadataKey, "final_sqrt");
    node_meta(389, ann::kNodeTagMetadataKey, "weight");
    node_meta(389, ann::kInPlaceReuseMetadataKey, "0:0:equal");
    node_meta(389, ann::kReleaseAfterMetadataKey, "final_f32;final_rsqrt");
    node_meta(390, ann::kNodeTagMetadataKey, "weight");
    node_meta(390, ann::kReleaseAfterMetadataKey, "final_mul");
    node_meta(391, ann::kNodeTagMetadataKey, "weight");
    node_meta(391, ann::kInPlaceReuseMetadataKey, "0:1:equal");
    node_meta(391, ann::kReleaseAfterMetadataKey, "final_half");
    node_meta(392, ann::kNodeTagMetadataKey, "weight");
    node_meta(392, ann::kReleaseAfterMetadataKey, "final_normed");
  }

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
