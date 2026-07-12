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

// Builds a deterministic pseudo-random FP16 weight vector using an LCG.
// All external weights from the original model are replaced with random values.
std::vector<uint16_t> RandomWeightsF16(size_t count, uint32_t seed) {
  std::vector<uint16_t> values(count);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    values[i] = static_cast<uint16_t>(0x3400u | (s & 0x03FFu));
  }
  return values;
}

} // namespace

// ---------------------------------------------------------------------------
// ``qwen3_4_layers_like`` — a 4-layer Qwen3-style causal language model.
//
// Faithfully reproduces the structure of a Qwen3 model with 4 transformer
// layers exported from PyTorch via onnxscript (opset 21, IR version 10).
// External weight initializers are replaced with deterministic random FP16
// values; doc_strings are omitted.
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
void RegisterQwen3_4LayersLikeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(21);

  const std::string name = "test_cc_shape_inference_qwen3_4_layers_like";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3f;
  tc.atol = 1e-5f;

  ModelProto &model = tc.model;
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
  AddInitializer<int64_t>(*graph, "init7_s1_2__6", {INT64_C(1)}, {INT64_C(2)});
  AddInitializer<uint16_t>(*graph, "init10_s1___6", {INT64_C(1)}, {static_cast<uint16_t>(64512u)});
  AddInitializer<uint16_t>(*graph, "init10_s1_2__6", {INT64_C(1)}, {static_cast<uint16_t>(0u)});
  AddInitializer<int64_t>(*graph, "init7_s1_2__9", {INT64_C(1)}, {INT64_C(2)});
  AddInitializer<uint16_t>(*graph, "init10_s1___9", {INT64_C(1)}, {static_cast<uint16_t>(64512u)});
  AddInitializer<uint16_t>(*graph, "init10_s1_2__9", {INT64_C(1)}, {static_cast<uint16_t>(0u)});
  AddInitializer<int64_t>(*graph, "init7_s1_2__12", {INT64_C(1)}, {INT64_C(2)});
  AddInitializer<uint16_t>(*graph, "init10_s1___12", {INT64_C(1)}, {static_cast<uint16_t>(64512u)});
  AddInitializer<uint16_t>(*graph, "init10_s1_2__12", {INT64_C(1)}, {static_cast<uint16_t>(0u)});
  AddInitializer<int64_t>(*graph, "init7_s1_2__15", {INT64_C(1)}, {INT64_C(2)});
  AddInitializer<uint16_t>(*graph, "init10_s1___15", {INT64_C(1)}, {static_cast<uint16_t>(64512u)});
  AddInitializer<uint16_t>(*graph, "init10_s1_2__15", {INT64_C(1)}, {static_cast<uint16_t>(0u)});

  // ---- Graph initializers -------------------------------------------------
  AddInitializer<int64_t>(*graph, "init7_s1_1", {INT64_C(1)}, {INT64_C(1)});
  AddInitializer<int64_t>(*graph, "init7_s1_-1", {INT64_C(1)}, {INT64_C(-1)});
  AddInitializer<float>(*graph, "init1_s_", {}, {2.0f});
  AddInitializer<uint16_t>(*graph, "init10_s1_", {INT64_C(1)}, {static_cast<uint16_t>(13506u)});
  AddInitializer<float>(*graph, "init1_s_2::RSh1", {INT64_C(1)}, {1e-06f});
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_self_attn_q_proj_weight::T10",
                           {INT64_C(1024), INT64_C(2048)}, RandomWeightsF16(2097152u, 100u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_self_attn_k_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 101u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_self_attn_v_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 102u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_self_attn_o_proj_weight::T10",
                           {INT64_C(2048), INT64_C(1024)}, RandomWeightsF16(2097152u, 103u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_mlp_gate_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 104u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_mlp_up_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 105u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_0_mlp_down_proj_weight::T10",
                           {INT64_C(3072), INT64_C(1024)}, RandomWeightsF16(3145728u, 106u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_self_attn_q_proj_weight::T10",
                           {INT64_C(1024), INT64_C(2048)}, RandomWeightsF16(2097152u, 107u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_self_attn_k_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 108u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_self_attn_v_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 109u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_self_attn_o_proj_weight::T10",
                           {INT64_C(2048), INT64_C(1024)}, RandomWeightsF16(2097152u, 110u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_mlp_gate_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 111u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_mlp_up_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 112u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_1_mlp_down_proj_weight::T10",
                           {INT64_C(3072), INT64_C(1024)}, RandomWeightsF16(3145728u, 113u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_self_attn_q_proj_weight::T10",
                           {INT64_C(1024), INT64_C(2048)}, RandomWeightsF16(2097152u, 114u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_self_attn_k_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 115u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_self_attn_v_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 116u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_self_attn_o_proj_weight::T10",
                           {INT64_C(2048), INT64_C(1024)}, RandomWeightsF16(2097152u, 117u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_mlp_gate_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 118u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_mlp_up_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 119u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_2_mlp_down_proj_weight::T10",
                           {INT64_C(3072), INT64_C(1024)}, RandomWeightsF16(3145728u, 120u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_self_attn_q_proj_weight::T10",
                           {INT64_C(1024), INT64_C(2048)}, RandomWeightsF16(2097152u, 121u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_self_attn_k_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 122u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_self_attn_v_proj_weight::T10",
                           {INT64_C(1024), INT64_C(1024)}, RandomWeightsF16(1048576u, 123u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_self_attn_o_proj_weight::T10",
                           {INT64_C(2048), INT64_C(1024)}, RandomWeightsF16(2097152u, 124u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_mlp_gate_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 125u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_mlp_up_proj_weight::T10",
                           {INT64_C(1024), INT64_C(3072)}, RandomWeightsF16(3145728u, 126u));
  AddInitializer<uint16_t>(*graph, "p_model_layers_3_mlp_down_proj_weight::T10",
                           {INT64_C(3072), INT64_C(1024)}, RandomWeightsF16(3145728u, 127u));
  AddInitializer<uint16_t>(*graph, "p_lm_head_weight::T10", {INT64_C(1024), INT64_C(151936)},
                           RandomWeightsF16(155582464u, 128u));
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
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.0.self_attn.q_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.0.self_attn.k_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  AddInitializer<uint16_t>(*graph, "model.layers.0.input_layernorm.weight", {INT64_C(1024)},
                           RandomWeightsF16(1024u, 129u));
  AddInitializer<uint16_t>(*graph, "model.layers.0.post_attention_layernorm.weight",
                           {INT64_C(1024)}, RandomWeightsF16(1024u, 130u));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.1.self_attn.q_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.1.self_attn.k_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  AddInitializer<uint16_t>(*graph, "model.layers.1.input_layernorm.weight", {INT64_C(1024)},
                           RandomWeightsF16(1024u, 131u));
  AddInitializer<uint16_t>(*graph, "model.layers.1.post_attention_layernorm.weight",
                           {INT64_C(1024)}, RandomWeightsF16(1024u, 132u));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.2.self_attn.q_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.2.self_attn.k_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  AddInitializer<uint16_t>(*graph, "model.layers.2.input_layernorm.weight", {INT64_C(1024)},
                           RandomWeightsF16(1024u, 133u));
  AddInitializer<uint16_t>(*graph, "model.layers.2.post_attention_layernorm.weight",
                           {INT64_C(1024)}, RandomWeightsF16(1024u, 134u));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.3.self_attn.q_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  // All-same FP16 initializer (value=0x3C00).
  AddInitializer<uint16_t>(*graph, "model.layers.3.self_attn.k_norm.weight", {INT64_C(128)},
                           std::vector<uint16_t>(128u, static_cast<uint16_t>(15360u)));
  AddInitializer<uint16_t>(*graph, "model.layers.3.input_layernorm.weight", {INT64_C(1024)},
                           RandomWeightsF16(1024u, 135u));
  AddInitializer<uint16_t>(*graph, "model.layers.3.post_attention_layernorm.weight",
                           {INT64_C(1024)}, RandomWeightsF16(1024u, 136u));
  AddInitializer<uint16_t>(*graph, "model.norm.weight", {INT64_C(1024)},
                           RandomWeightsF16(1024u, 137u));
  AddInitializer<uint16_t>(*graph, "lm_head.weight", {INT64_C(151936), INT64_C(1024)},
                           RandomWeightsF16(155582464u, 138u));

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
  {
    NodeProto &n = AddNode(*graph, "Cast", {"embedding"}, {"to_10"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_10", "init1_s_"}, {"pow_1"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_1", "init7_s1_-1"}, {"mean"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean", "init1_s_2::RSh1"}, {"add_5"});
  AddNode(*graph, "Sqrt", {"add_5"}, {"_onx_sqrt_add_5"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_5"}, {"rsqrt"});
  AddNode(*graph, "Mul", {"to_10", "rsqrt"}, {"mul_2"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_2"}, {"to_11"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.0.input_layernorm.weight", "to_11"}, {"mul_3"});
  AddNode(*graph, "MatMul", {"mul_3", "p_model_layers_0_self_attn_q_proj_weight::T10"},
          {"_onx_matmul_mul_3"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_3"}, {"SwapUnaryPattern--view"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view", "init7_s4_0_0_16_128"}, {"to_12"});
  AddNode(*graph, "Pow", {"to_12", "init1_s_"}, {"pow_2"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_2", "init7_s1_-1"}, {"mean_1"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_1", "init1_s_2::RSh1"}, {"add_6"});
  AddNode(*graph, "Sqrt", {"add_6"}, {"_onx_sqrt_add_6"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_6"}, {"rsqrt_1"});
  AddNode(*graph, "Mul", {"to_12", "rsqrt_1"}, {"mul_4"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_4"}, {"to_13"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.0.self_attn.q_norm.weight", "to_13"}, {"mul_5"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_5"}, {"transpose_1"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_3", "p_model_layers_0_self_attn_k_proj_weight::T10"},
          {"_onx_matmul_mul_32"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_32"}, {"SwapUnaryPattern--view_1"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_1", "init7_s4_0_0_8_128"}, {"to_14"});
  AddNode(*graph, "Pow", {"to_14", "init1_s_"}, {"pow_3"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_3", "init7_s1_-1"}, {"mean_2"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_2", "init1_s_2::RSh1"}, {"add_7"});
  AddNode(*graph, "Sqrt", {"add_7"}, {"_onx_sqrt_add_7"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_7"}, {"rsqrt_2"});
  AddNode(*graph, "Mul", {"to_14", "rsqrt_2"}, {"mul_6"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_6"}, {"to_15"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.0.self_attn.k_norm.weight", "to_15"}, {"mul_7"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_7"}, {"transpose_2"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_3", "p_model_layers_0_self_attn_v_proj_weight::T10"},
          {"_onx_matmul_mul_33"});
  AddNode(*graph, "Reshape", {"_onx_matmul_mul_33", "init7_s4_0_0_8_128"}, {"view_2"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"view_2"}, {"transpose_3"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_1"}, {"_onx_split_X_0__4", "_onx_split_X_1__4"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__4"}, {"_onx_neg_split_X_1__4"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__4", "_onx_split_X_0__4"},
                           {"_onx_concat_neg_split_X_1__4"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_1", "unsqueeze_16"}, {"_onx_mul_X__4"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__4", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__4"});
  AddNode(*graph, "Add", {"_onx_mul_X__4", "_onx_mul_concat_neg_split_X_1__4"}, {"add_8"});
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_2"}, {"_onx_split_X_0__5", "_onx_split_X_1__5"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__5"}, {"_onx_neg_split_X_1__5"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__5", "_onx_split_X_0__5"},
                           {"_onx_concat_neg_split_X_1__5"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_2", "unsqueeze_16"}, {"_onx_mul_X__5"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__5", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__5"});
  AddNode(*graph, "Add", {"_onx_mul_X__5", "_onx_mul_concat_neg_split_X_1__5"}, {"add_9"});
  {
    NodeProto &n =
        AddNode(*graph, "Concat", {"past_key_values_key_0", "add_9"}, {"present_key_values_key_0"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_value_0", "transpose_3"},
                           {"present_key_values_value_0"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  AddNode(*graph, "Mul", {"present_key_values_key_0", "init10_s1_"}, {"_onx_mul_keys__6"});
  AddNode(*graph, "Unsqueeze", {"_onx_mul_keys__6", "init7_s1_2__6"}, {"_onx_mul_keys::UnSq2__6"});
  AddNode(*graph, "Unsqueeze", {"present_key_values_value_0", "init7_s1_2__6"},
          {"values::UnSq2__6"});
  AddNode(*graph, "Expand", {"_onx_mul_keys::UnSq2__6", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_mul_keys::UnSq2__6"});
  AddNode(*graph, "Expand", {"values::UnSq2__6", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_values::UnSq2__6"});
  AddNode(*graph, "Reshape", {"_onx_expand_mul_keys::UnSq2__6", "init7_s4_0_16_-1_128"},
          {"_onx_expand_mul_keys::UnSq2::RSh__6"});
  AddNode(*graph, "Reshape", {"_onx_expand_values::UnSq2__6", "init7_s4_0_16_-1_128"},
          {"_onx_expand_values::UnSq2::RSh__6"});
  AddNode(*graph, "Mul", {"add_8", "init10_s1_"}, {"_onx_mul_query__6"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"_onx_expand_mul_keys::UnSq2::RSh__6"},
                           {"_onx_expand_mul_keys::UnSq2::RSh::T0132__6"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(1), INT64_C(3), INT64_C(2)});
  }
  AddNode(*graph, "MatMul", {"_onx_mul_query__6", "_onx_expand_mul_keys::UnSq2::RSh::T0132__6"},
          {"_onx_matmul_mul_query__6"});
  AddNode(*graph, "Where", {"and_2", "_onx_matmul_mul_query__6", "init10_s1___6"},
          {"_onx_where_mask__6"});
  {
    NodeProto &n =
        AddNode(*graph, "Softmax", {"_onx_where_mask__6"}, {"_onx_softmax_where_mask__6"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "IsNaN", {"_onx_softmax_where_mask__6"}, {"_onx_isnan_softmax_where_mask__6"});
  AddNode(*graph, "Where",
          {"_onx_isnan_softmax_where_mask__6", "init10_s1_2__6", "_onx_softmax_where_mask__6"},
          {"_onx_where_isnan_softmax_where_mask__6"});
  AddNode(*graph, "MatMul",
          {"_onx_where_isnan_softmax_where_mask__6", "_onx_expand_values::UnSq2::RSh__6"},
          {"scaled_dot_product_attention"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"scaled_dot_product_attention"}, {"transpose_4"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "Reshape", {"transpose_4", "init7_s3_0_0_2048"}, {"reshape_2"});
  AddNode(*graph, "MatMul", {"reshape_2", "p_model_layers_0_self_attn_o_proj_weight::T10"},
          {"_onx_matmul_reshape_2"});
  AddNode(*graph, "Add", {"embedding", "_onx_matmul_reshape_2"}, {"add_10"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_10"}, {"to_16"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_16", "init1_s_"}, {"pow_4"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_4", "init7_s1_-1"}, {"mean_3"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_3", "init1_s_2::RSh1"}, {"add_11"});
  AddNode(*graph, "Sqrt", {"add_11"}, {"_onx_sqrt_add_11"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_11"}, {"rsqrt_3"});
  AddNode(*graph, "Mul", {"to_16", "rsqrt_3"}, {"mul_20"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_20"}, {"to_17"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.0.post_attention_layernorm.weight", "to_17"}, {"mul_21"});
  AddNode(*graph, "MatMul", {"mul_21", "p_model_layers_0_mlp_gate_proj_weight::T10"},
          {"_onx_matmul_mul_21"});
  AddNode(*graph, "Sigmoid", {"_onx_matmul_mul_21"}, {"_onx_sigmoid_linear_4"});
  AddNode(*graph, "Mul", {"_onx_matmul_mul_21", "_onx_sigmoid_linear_4"}, {"silu"});
  AddNode(*graph, "MatMul", {"mul_21", "p_model_layers_0_mlp_up_proj_weight::T10"},
          {"_onx_matmul_mul_212"});
  AddNode(*graph, "Mul", {"silu", "_onx_matmul_mul_212"}, {"mul_22"});
  AddNode(*graph, "MatMul", {"mul_22", "p_model_layers_0_mlp_down_proj_weight::T10"},
          {"_onx_matmul_mul_22"});
  AddNode(*graph, "Add", {"add_10", "_onx_matmul_mul_22"}, {"add_12"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_12"}, {"to_18"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_18", "init1_s_"}, {"pow_5"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_5", "init7_s1_-1"}, {"mean_4"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_4", "init1_s_2::RSh1"}, {"add_13"});
  AddNode(*graph, "Sqrt", {"add_13"}, {"_onx_sqrt_add_13"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_13"}, {"rsqrt_4"});
  AddNode(*graph, "Mul", {"to_18", "rsqrt_4"}, {"mul_23"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_23"}, {"to_19"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.1.input_layernorm.weight", "to_19"}, {"mul_24"});
  AddNode(*graph, "MatMul", {"mul_24", "p_model_layers_1_self_attn_q_proj_weight::T10"},
          {"_onx_matmul_mul_24"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_24"}, {"SwapUnaryPattern--view_3"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_3", "init7_s4_0_0_16_128"}, {"to_20"});
  AddNode(*graph, "Pow", {"to_20", "init1_s_"}, {"pow_6"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_6", "init7_s1_-1"}, {"mean_5"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_5", "init1_s_2::RSh1"}, {"add_14"});
  AddNode(*graph, "Sqrt", {"add_14"}, {"_onx_sqrt_add_14"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_14"}, {"rsqrt_5"});
  AddNode(*graph, "Mul", {"to_20", "rsqrt_5"}, {"mul_25"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_25"}, {"to_21"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.1.self_attn.q_norm.weight", "to_21"}, {"mul_26"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_26"}, {"transpose_5"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_24", "p_model_layers_1_self_attn_k_proj_weight::T10"},
          {"_onx_matmul_mul_242"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_242"}, {"SwapUnaryPattern--view_4"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_4", "init7_s4_0_0_8_128"}, {"to_22"});
  AddNode(*graph, "Pow", {"to_22", "init1_s_"}, {"pow_7"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_7", "init7_s1_-1"}, {"mean_6"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_6", "init1_s_2::RSh1"}, {"add_15"});
  AddNode(*graph, "Sqrt", {"add_15"}, {"_onx_sqrt_add_15"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_15"}, {"rsqrt_6"});
  AddNode(*graph, "Mul", {"to_22", "rsqrt_6"}, {"mul_27"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_27"}, {"to_23"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.1.self_attn.k_norm.weight", "to_23"}, {"mul_28"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_28"}, {"transpose_6"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_24", "p_model_layers_1_self_attn_v_proj_weight::T10"},
          {"_onx_matmul_mul_243"});
  AddNode(*graph, "Reshape", {"_onx_matmul_mul_243", "init7_s4_0_0_8_128"}, {"view_5"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"view_5"}, {"transpose_7"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_5"}, {"_onx_split_X_0__7", "_onx_split_X_1__7"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__7"}, {"_onx_neg_split_X_1__7"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__7", "_onx_split_X_0__7"},
                           {"_onx_concat_neg_split_X_1__7"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_5", "unsqueeze_16"}, {"_onx_mul_X__7"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__7", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__7"});
  AddNode(*graph, "Add", {"_onx_mul_X__7", "_onx_mul_concat_neg_split_X_1__7"}, {"add_16"});
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_6"}, {"_onx_split_X_0__8", "_onx_split_X_1__8"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__8"}, {"_onx_neg_split_X_1__8"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__8", "_onx_split_X_0__8"},
                           {"_onx_concat_neg_split_X_1__8"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_6", "unsqueeze_16"}, {"_onx_mul_X__8"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__8", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__8"});
  AddNode(*graph, "Add", {"_onx_mul_X__8", "_onx_mul_concat_neg_split_X_1__8"}, {"add_17"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_key_1", "add_17"},
                           {"present_key_values_key_1"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_value_1", "transpose_7"},
                           {"present_key_values_value_1"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  AddNode(*graph, "Mul", {"present_key_values_key_1", "init10_s1_"}, {"_onx_mul_keys__9"});
  AddNode(*graph, "Unsqueeze", {"_onx_mul_keys__9", "init7_s1_2__9"}, {"_onx_mul_keys::UnSq2__9"});
  AddNode(*graph, "Unsqueeze", {"present_key_values_value_1", "init7_s1_2__9"},
          {"values::UnSq2__9"});
  AddNode(*graph, "Expand", {"_onx_mul_keys::UnSq2__9", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_mul_keys::UnSq2__9"});
  AddNode(*graph, "Expand", {"values::UnSq2__9", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_values::UnSq2__9"});
  AddNode(*graph, "Reshape", {"_onx_expand_mul_keys::UnSq2__9", "init7_s4_0_16_-1_128"},
          {"_onx_expand_mul_keys::UnSq2::RSh__9"});
  AddNode(*graph, "Reshape", {"_onx_expand_values::UnSq2__9", "init7_s4_0_16_-1_128"},
          {"_onx_expand_values::UnSq2::RSh__9"});
  AddNode(*graph, "Mul", {"add_16", "init10_s1_"}, {"_onx_mul_query__9"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"_onx_expand_mul_keys::UnSq2::RSh__9"},
                           {"_onx_expand_mul_keys::UnSq2::RSh::T0132__9"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(1), INT64_C(3), INT64_C(2)});
  }
  AddNode(*graph, "MatMul", {"_onx_mul_query__9", "_onx_expand_mul_keys::UnSq2::RSh::T0132__9"},
          {"_onx_matmul_mul_query__9"});
  AddNode(*graph, "Where", {"and_2", "_onx_matmul_mul_query__9", "init10_s1___9"},
          {"_onx_where_mask__9"});
  {
    NodeProto &n =
        AddNode(*graph, "Softmax", {"_onx_where_mask__9"}, {"_onx_softmax_where_mask__9"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "IsNaN", {"_onx_softmax_where_mask__9"}, {"_onx_isnan_softmax_where_mask__9"});
  AddNode(*graph, "Where",
          {"_onx_isnan_softmax_where_mask__9", "init10_s1_2__9", "_onx_softmax_where_mask__9"},
          {"_onx_where_isnan_softmax_where_mask__9"});
  AddNode(*graph, "MatMul",
          {"_onx_where_isnan_softmax_where_mask__9", "_onx_expand_values::UnSq2::RSh__9"},
          {"scaled_dot_product_attention_1"});
  {
    NodeProto &n =
        AddNode(*graph, "Transpose", {"scaled_dot_product_attention_1"}, {"transpose_8"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "Reshape", {"transpose_8", "init7_s3_0_0_2048"}, {"reshape_5"});
  AddNode(*graph, "MatMul", {"reshape_5", "p_model_layers_1_self_attn_o_proj_weight::T10"},
          {"_onx_matmul_reshape_5"});
  AddNode(*graph, "Add", {"add_12", "_onx_matmul_reshape_5"}, {"add_18"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_18"}, {"to_24"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_24", "init1_s_"}, {"pow_8"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_8", "init7_s1_-1"}, {"mean_7"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_7", "init1_s_2::RSh1"}, {"add_19"});
  AddNode(*graph, "Sqrt", {"add_19"}, {"_onx_sqrt_add_19"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_19"}, {"rsqrt_7"});
  AddNode(*graph, "Mul", {"to_24", "rsqrt_7"}, {"mul_41"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_41"}, {"to_25"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.1.post_attention_layernorm.weight", "to_25"}, {"mul_42"});
  AddNode(*graph, "MatMul", {"mul_42", "p_model_layers_1_mlp_gate_proj_weight::T10"},
          {"_onx_matmul_mul_42"});
  AddNode(*graph, "Sigmoid", {"_onx_matmul_mul_42"}, {"_onx_sigmoid_linear_11"});
  AddNode(*graph, "Mul", {"_onx_matmul_mul_42", "_onx_sigmoid_linear_11"}, {"silu_1"});
  AddNode(*graph, "MatMul", {"mul_42", "p_model_layers_1_mlp_up_proj_weight::T10"},
          {"_onx_matmul_mul_422"});
  AddNode(*graph, "Mul", {"silu_1", "_onx_matmul_mul_422"}, {"mul_43"});
  AddNode(*graph, "MatMul", {"mul_43", "p_model_layers_1_mlp_down_proj_weight::T10"},
          {"_onx_matmul_mul_43"});
  AddNode(*graph, "Add", {"add_18", "_onx_matmul_mul_43"}, {"add_20"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_20"}, {"to_26"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_26", "init1_s_"}, {"pow_9"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_9", "init7_s1_-1"}, {"mean_8"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_8", "init1_s_2::RSh1"}, {"add_21"});
  AddNode(*graph, "Sqrt", {"add_21"}, {"_onx_sqrt_add_21"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_21"}, {"rsqrt_8"});
  AddNode(*graph, "Mul", {"to_26", "rsqrt_8"}, {"mul_44"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_44"}, {"to_27"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.2.input_layernorm.weight", "to_27"}, {"mul_45"});
  AddNode(*graph, "MatMul", {"mul_45", "p_model_layers_2_self_attn_q_proj_weight::T10"},
          {"_onx_matmul_mul_45"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_45"}, {"SwapUnaryPattern--view_6"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_6", "init7_s4_0_0_16_128"}, {"to_28"});
  AddNode(*graph, "Pow", {"to_28", "init1_s_"}, {"pow_10"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_10", "init7_s1_-1"}, {"mean_9"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_9", "init1_s_2::RSh1"}, {"add_22"});
  AddNode(*graph, "Sqrt", {"add_22"}, {"_onx_sqrt_add_22"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_22"}, {"rsqrt_9"});
  AddNode(*graph, "Mul", {"to_28", "rsqrt_9"}, {"mul_46"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_46"}, {"to_29"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.2.self_attn.q_norm.weight", "to_29"}, {"mul_47"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_47"}, {"transpose_9"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_45", "p_model_layers_2_self_attn_k_proj_weight::T10"},
          {"_onx_matmul_mul_452"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_452"}, {"SwapUnaryPattern--view_7"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_7", "init7_s4_0_0_8_128"}, {"to_30"});
  AddNode(*graph, "Pow", {"to_30", "init1_s_"}, {"pow_11"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_11", "init7_s1_-1"}, {"mean_10"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_10", "init1_s_2::RSh1"}, {"add_23"});
  AddNode(*graph, "Sqrt", {"add_23"}, {"_onx_sqrt_add_23"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_23"}, {"rsqrt_10"});
  AddNode(*graph, "Mul", {"to_30", "rsqrt_10"}, {"mul_48"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_48"}, {"to_31"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.2.self_attn.k_norm.weight", "to_31"}, {"mul_49"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_49"}, {"transpose_10"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_45", "p_model_layers_2_self_attn_v_proj_weight::T10"},
          {"_onx_matmul_mul_453"});
  AddNode(*graph, "Reshape", {"_onx_matmul_mul_453", "init7_s4_0_0_8_128"}, {"view_8"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"view_8"}, {"transpose_11"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_9"}, {"_onx_split_X_0__10", "_onx_split_X_1__10"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__10"}, {"_onx_neg_split_X_1__10"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__10", "_onx_split_X_0__10"},
                           {"_onx_concat_neg_split_X_1__10"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_9", "unsqueeze_16"}, {"_onx_mul_X__10"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__10", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__10"});
  AddNode(*graph, "Add", {"_onx_mul_X__10", "_onx_mul_concat_neg_split_X_1__10"}, {"add_24"});
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_10"}, {"_onx_split_X_0__11", "_onx_split_X_1__11"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__11"}, {"_onx_neg_split_X_1__11"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__11", "_onx_split_X_0__11"},
                           {"_onx_concat_neg_split_X_1__11"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_10", "unsqueeze_16"}, {"_onx_mul_X__11"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__11", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__11"});
  AddNode(*graph, "Add", {"_onx_mul_X__11", "_onx_mul_concat_neg_split_X_1__11"}, {"add_25"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_key_2", "add_25"},
                           {"present_key_values_key_2"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_value_2", "transpose_11"},
                           {"present_key_values_value_2"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  AddNode(*graph, "Mul", {"present_key_values_key_2", "init10_s1_"}, {"_onx_mul_keys__12"});
  AddNode(*graph, "Unsqueeze", {"_onx_mul_keys__12", "init7_s1_2__12"},
          {"_onx_mul_keys::UnSq2__12"});
  AddNode(*graph, "Unsqueeze", {"present_key_values_value_2", "init7_s1_2__12"},
          {"values::UnSq2__12"});
  AddNode(*graph, "Expand", {"_onx_mul_keys::UnSq2__12", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_mul_keys::UnSq2__12"});
  AddNode(*graph, "Expand", {"values::UnSq2__12", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_values::UnSq2__12"});
  AddNode(*graph, "Reshape", {"_onx_expand_mul_keys::UnSq2__12", "init7_s4_0_16_-1_128"},
          {"_onx_expand_mul_keys::UnSq2::RSh__12"});
  AddNode(*graph, "Reshape", {"_onx_expand_values::UnSq2__12", "init7_s4_0_16_-1_128"},
          {"_onx_expand_values::UnSq2::RSh__12"});
  AddNode(*graph, "Mul", {"add_24", "init10_s1_"}, {"_onx_mul_query__12"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"_onx_expand_mul_keys::UnSq2::RSh__12"},
                           {"_onx_expand_mul_keys::UnSq2::RSh::T0132__12"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(1), INT64_C(3), INT64_C(2)});
  }
  AddNode(*graph, "MatMul", {"_onx_mul_query__12", "_onx_expand_mul_keys::UnSq2::RSh::T0132__12"},
          {"_onx_matmul_mul_query__12"});
  AddNode(*graph, "Where", {"and_2", "_onx_matmul_mul_query__12", "init10_s1___12"},
          {"_onx_where_mask__12"});
  {
    NodeProto &n =
        AddNode(*graph, "Softmax", {"_onx_where_mask__12"}, {"_onx_softmax_where_mask__12"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "IsNaN", {"_onx_softmax_where_mask__12"}, {"_onx_isnan_softmax_where_mask__12"});
  AddNode(*graph, "Where",
          {"_onx_isnan_softmax_where_mask__12", "init10_s1_2__12", "_onx_softmax_where_mask__12"},
          {"_onx_where_isnan_softmax_where_mask__12"});
  AddNode(*graph, "MatMul",
          {"_onx_where_isnan_softmax_where_mask__12", "_onx_expand_values::UnSq2::RSh__12"},
          {"scaled_dot_product_attention_2"});
  {
    NodeProto &n =
        AddNode(*graph, "Transpose", {"scaled_dot_product_attention_2"}, {"transpose_12"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "Reshape", {"transpose_12", "init7_s3_0_0_2048"}, {"reshape_8"});
  AddNode(*graph, "MatMul", {"reshape_8", "p_model_layers_2_self_attn_o_proj_weight::T10"},
          {"_onx_matmul_reshape_8"});
  AddNode(*graph, "Add", {"add_20", "_onx_matmul_reshape_8"}, {"add_26"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_26"}, {"to_32"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_32", "init1_s_"}, {"pow_12"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_12", "init7_s1_-1"}, {"mean_11"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_11", "init1_s_2::RSh1"}, {"add_27"});
  AddNode(*graph, "Sqrt", {"add_27"}, {"_onx_sqrt_add_27"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_27"}, {"rsqrt_11"});
  AddNode(*graph, "Mul", {"to_32", "rsqrt_11"}, {"mul_62"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_62"}, {"to_33"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.2.post_attention_layernorm.weight", "to_33"}, {"mul_63"});
  AddNode(*graph, "MatMul", {"mul_63", "p_model_layers_2_mlp_gate_proj_weight::T10"},
          {"_onx_matmul_mul_63"});
  AddNode(*graph, "Sigmoid", {"_onx_matmul_mul_63"}, {"_onx_sigmoid_linear_18"});
  AddNode(*graph, "Mul", {"_onx_matmul_mul_63", "_onx_sigmoid_linear_18"}, {"silu_2"});
  AddNode(*graph, "MatMul", {"mul_63", "p_model_layers_2_mlp_up_proj_weight::T10"},
          {"_onx_matmul_mul_632"});
  AddNode(*graph, "Mul", {"silu_2", "_onx_matmul_mul_632"}, {"mul_64"});
  AddNode(*graph, "MatMul", {"mul_64", "p_model_layers_2_mlp_down_proj_weight::T10"},
          {"_onx_matmul_mul_64"});
  AddNode(*graph, "Add", {"add_26", "_onx_matmul_mul_64"}, {"add_28"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_28"}, {"to_34"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_34", "init1_s_"}, {"pow_13"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_13", "init7_s1_-1"}, {"mean_12"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_12", "init1_s_2::RSh1"}, {"add_29"});
  AddNode(*graph, "Sqrt", {"add_29"}, {"_onx_sqrt_add_29"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_29"}, {"rsqrt_12"});
  AddNode(*graph, "Mul", {"to_34", "rsqrt_12"}, {"mul_65"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_65"}, {"to_35"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.3.input_layernorm.weight", "to_35"}, {"mul_66"});
  AddNode(*graph, "MatMul", {"mul_66", "p_model_layers_3_self_attn_q_proj_weight::T10"},
          {"_onx_matmul_mul_66"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_66"}, {"SwapUnaryPattern--view_9"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_9", "init7_s4_0_0_16_128"}, {"to_36"});
  AddNode(*graph, "Pow", {"to_36", "init1_s_"}, {"pow_14"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_14", "init7_s1_-1"}, {"mean_13"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_13", "init1_s_2::RSh1"}, {"add_30"});
  AddNode(*graph, "Sqrt", {"add_30"}, {"_onx_sqrt_add_30"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_30"}, {"rsqrt_13"});
  AddNode(*graph, "Mul", {"to_36", "rsqrt_13"}, {"mul_67"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_67"}, {"to_37"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.3.self_attn.q_norm.weight", "to_37"}, {"mul_68"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_68"}, {"transpose_13"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_66", "p_model_layers_3_self_attn_k_proj_weight::T10"},
          {"_onx_matmul_mul_662"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"_onx_matmul_mul_662"}, {"SwapUnaryPattern--view_10"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Reshape", {"SwapUnaryPattern--view_10", "init7_s4_0_0_8_128"}, {"to_38"});
  AddNode(*graph, "Pow", {"to_38", "init1_s_"}, {"pow_15"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_15", "init7_s1_-1"}, {"mean_14"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_14", "init1_s_2::RSh1"}, {"add_31"});
  AddNode(*graph, "Sqrt", {"add_31"}, {"_onx_sqrt_add_31"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_31"}, {"rsqrt_14"});
  AddNode(*graph, "Mul", {"to_38", "rsqrt_14"}, {"mul_69"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_69"}, {"to_39"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.3.self_attn.k_norm.weight", "to_39"}, {"mul_70"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"mul_70"}, {"transpose_14"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "MatMul", {"mul_66", "p_model_layers_3_self_attn_v_proj_weight::T10"},
          {"_onx_matmul_mul_663"});
  AddNode(*graph, "Reshape", {"_onx_matmul_mul_663", "init7_s4_0_0_8_128"}, {"view_11"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"view_11"}, {"transpose_15"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_13"}, {"_onx_split_X_0__13", "_onx_split_X_1__13"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__13"}, {"_onx_neg_split_X_1__13"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__13", "_onx_split_X_0__13"},
                           {"_onx_concat_neg_split_X_1__13"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_13", "unsqueeze_16"}, {"_onx_mul_X__13"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__13", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__13"});
  AddNode(*graph, "Add", {"_onx_mul_X__13", "_onx_mul_concat_neg_split_X_1__13"}, {"add_32"});
  {
    NodeProto &n =
        AddNode(*graph, "Split", {"transpose_14"}, {"_onx_split_X_0__14", "_onx_split_X_1__14"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
    AddAttribute<int64_t>(n, "num_outputs", INT64_C(2));
  }
  AddNode(*graph, "Neg", {"_onx_split_X_1__14"}, {"_onx_neg_split_X_1__14"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"_onx_neg_split_X_1__14", "_onx_split_X_0__14"},
                           {"_onx_concat_neg_split_X_1__14"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "Mul", {"transpose_14", "unsqueeze_16"}, {"_onx_mul_X__14"});
  AddNode(*graph, "Mul", {"_onx_concat_neg_split_X_1__14", "unsqueeze_17"},
          {"_onx_mul_concat_neg_split_X_1__14"});
  AddNode(*graph, "Add", {"_onx_mul_X__14", "_onx_mul_concat_neg_split_X_1__14"}, {"add_33"});
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_key_3", "add_33"},
                           {"present_key_values_key_3"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  {
    NodeProto &n = AddNode(*graph, "Concat", {"past_key_values_value_3", "transpose_15"},
                           {"present_key_values_value_3"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-2));
  }
  AddNode(*graph, "Mul", {"present_key_values_key_3", "init10_s1_"}, {"_onx_mul_keys__15"});
  AddNode(*graph, "Unsqueeze", {"_onx_mul_keys__15", "init7_s1_2__15"},
          {"_onx_mul_keys::UnSq2__15"});
  AddNode(*graph, "Unsqueeze", {"present_key_values_value_3", "init7_s1_2__15"},
          {"values::UnSq2__15"});
  AddNode(*graph, "Expand", {"_onx_mul_keys::UnSq2__15", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_mul_keys::UnSq2__15"});
  AddNode(*graph, "Expand", {"values::UnSq2__15", "init7_s5_1_1_2_1_1"},
          {"_onx_expand_values::UnSq2__15"});
  AddNode(*graph, "Reshape", {"_onx_expand_mul_keys::UnSq2__15", "init7_s4_0_16_-1_128"},
          {"_onx_expand_mul_keys::UnSq2::RSh__15"});
  AddNode(*graph, "Reshape", {"_onx_expand_values::UnSq2__15", "init7_s4_0_16_-1_128"},
          {"_onx_expand_values::UnSq2::RSh__15"});
  AddNode(*graph, "Mul", {"add_32", "init10_s1_"}, {"_onx_mul_query__15"});
  {
    NodeProto &n = AddNode(*graph, "Transpose", {"_onx_expand_mul_keys::UnSq2::RSh__15"},
                           {"_onx_expand_mul_keys::UnSq2::RSh::T0132__15"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(1), INT64_C(3), INT64_C(2)});
  }
  AddNode(*graph, "MatMul", {"_onx_mul_query__15", "_onx_expand_mul_keys::UnSq2::RSh::T0132__15"},
          {"_onx_matmul_mul_query__15"});
  AddNode(*graph, "Where", {"and_2", "_onx_matmul_mul_query__15", "init10_s1___15"},
          {"_onx_where_mask__15"});
  {
    NodeProto &n =
        AddNode(*graph, "Softmax", {"_onx_where_mask__15"}, {"_onx_softmax_where_mask__15"});
    AddAttribute<int64_t>(n, "axis", INT64_C(-1));
  }
  AddNode(*graph, "IsNaN", {"_onx_softmax_where_mask__15"}, {"_onx_isnan_softmax_where_mask__15"});
  AddNode(*graph, "Where",
          {"_onx_isnan_softmax_where_mask__15", "init10_s1_2__15", "_onx_softmax_where_mask__15"},
          {"_onx_where_isnan_softmax_where_mask__15"});
  AddNode(*graph, "MatMul",
          {"_onx_where_isnan_softmax_where_mask__15", "_onx_expand_values::UnSq2::RSh__15"},
          {"scaled_dot_product_attention_3"});
  {
    NodeProto &n =
        AddNode(*graph, "Transpose", {"scaled_dot_product_attention_3"}, {"transpose_16"});
    AddAttribute<std::vector<int64_t>>(n, "perm", {INT64_C(0), INT64_C(2), INT64_C(1), INT64_C(3)});
  }
  AddNode(*graph, "Reshape", {"transpose_16", "init7_s3_0_0_2048"}, {"reshape_11"});
  AddNode(*graph, "MatMul", {"reshape_11", "p_model_layers_3_self_attn_o_proj_weight::T10"},
          {"_onx_matmul_reshape_11"});
  AddNode(*graph, "Add", {"add_28", "_onx_matmul_reshape_11"}, {"add_34"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_34"}, {"to_40"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_40", "init1_s_"}, {"pow_16"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_16", "init7_s1_-1"}, {"mean_15"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_15", "init1_s_2::RSh1"}, {"add_35"});
  AddNode(*graph, "Sqrt", {"add_35"}, {"_onx_sqrt_add_35"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_35"}, {"rsqrt_15"});
  AddNode(*graph, "Mul", {"to_40", "rsqrt_15"}, {"mul_75"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_75"}, {"to_41"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.layers.3.post_attention_layernorm.weight", "to_41"}, {"mul_76"});
  AddNode(*graph, "MatMul", {"mul_76", "p_model_layers_3_mlp_gate_proj_weight::T10"},
          {"_onx_matmul_mul_76"});
  AddNode(*graph, "Sigmoid", {"_onx_matmul_mul_76"}, {"_onx_sigmoid_linear_25"});
  AddNode(*graph, "Mul", {"_onx_matmul_mul_76", "_onx_sigmoid_linear_25"}, {"silu_3"});
  AddNode(*graph, "MatMul", {"mul_76", "p_model_layers_3_mlp_up_proj_weight::T10"},
          {"_onx_matmul_mul_762"});
  AddNode(*graph, "Mul", {"silu_3", "_onx_matmul_mul_762"}, {"mul_77"});
  AddNode(*graph, "MatMul", {"mul_77", "p_model_layers_3_mlp_down_proj_weight::T10"},
          {"_onx_matmul_mul_77"});
  AddNode(*graph, "Add", {"add_34", "_onx_matmul_mul_77"}, {"add_36"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"add_36"}, {"to_42"});
    AddAttribute<int64_t>(n, "to", INT64_C(1));
  }
  AddNode(*graph, "Pow", {"to_42", "init1_s_"}, {"pow_17"});
  {
    NodeProto &n = AddNode(*graph, "ReduceMean", {"pow_17", "init7_s1_-1"}, {"mean_16"});
    AddAttribute<int64_t>(n, "keepdims", INT64_C(1));
  }
  AddNode(*graph, "Add", {"mean_16", "init1_s_2::RSh1"}, {"add_37"});
  AddNode(*graph, "Sqrt", {"add_37"}, {"_onx_sqrt_add_37"});
  AddNode(*graph, "Reciprocal", {"_onx_sqrt_add_37"}, {"rsqrt_16"});
  AddNode(*graph, "Mul", {"to_42", "rsqrt_16"}, {"mul_78"});
  {
    NodeProto &n = AddNode(*graph, "Cast", {"mul_78"}, {"to_43"});
    AddAttribute<int64_t>(n, "to", INT64_C(10));
  }
  AddNode(*graph, "Mul", {"model.norm.weight", "to_43"}, {"mul_79"});
  AddNode(*graph, "MatMul", {"mul_79", "p_lm_head_weight::T10"}, {"output_0"});

  // ---- Graph inputs -------------------------------------------------------
  AppendValueInfo(*graph->add_input(), "input_ids", DataType::INT64,
                  {DimSpec("batch_size"), DimSpec("sequence_length")});
  AppendValueInfo(*graph->add_input(), "attention_mask", DataType::INT64,
                  {DimSpec("batch_size"), DimSpec("total_sequence_length")});
  AppendValueInfo(*graph->add_input(), "past_key_values_key_0", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_value_0", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_key_1", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_value_1", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_key_2", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_value_2", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_key_3", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_input(), "past_key_values_value_3", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)), DimSpec("past_sequence_length"),
                   DimSpec(INT64_C(128))});

  // ---- Graph outputs ------------------------------------------------------
  AppendValueInfo(*graph->add_output(), "output_0", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec("sequence_length"), DimSpec(INT64_C(151936))});
  AppendValueInfo(*graph->add_output(), "present_key_values_key_0", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_value_0", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_key_1", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_value_1", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_key_2", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_value_2", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_key_3", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});
  AppendValueInfo(*graph->add_output(), "present_key_values_value_3", DataType::FLOAT16,
                  {DimSpec("batch_size"), DimSpec(INT64_C(8)),
                   DimSpec("past_sequence_length+sequence_length"), DimSpec(INT64_C(128))});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
