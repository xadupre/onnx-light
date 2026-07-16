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

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
