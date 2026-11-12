// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

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

// Compact Whisper configuration: four encoder and four decoder blocks, with
// real random weights but no datasets (these cases exercise shape inference).
constexpr int64_t kHidden = 16;
constexpr int64_t kHeads = 4;
constexpr int64_t kHeadSize = kHidden / kHeads;
constexpr int64_t kIntermediate = 4 * kHidden;
constexpr int64_t kMelBins = 80;
constexpr int64_t kFrames = 32;
constexpr int64_t kSourceLength = kFrames / 2;
constexpr int64_t kMaxTargetLength = 16;
constexpr int64_t kVocab = 32;

class WhisperGraph {
public:
  GraphProto &graph;
  bool fused;
  uint32_t seed = 1;

  void Weight(const std::string &name, const std::vector<int64_t> &dims) {
    size_t count = 1;
    for (int64_t dim : dims)
      count *= static_cast<size_t>(dim);
    AddInitializer<float>(graph, name.c_str(), dims, RandomWeights(count, seed++));
  }

  std::string Linear(const std::string &input, const std::string &name, int64_t in_size,
                     int64_t out_size, bool bias = true) {
    Weight(name + ".weight", {in_size, out_size});
    AddNode(graph, "MatMul", {input, name + ".weight"}, {name + ".matmul"});
    if (!bias)
      return name + ".matmul";
    Weight(name + ".bias", {out_size});
    AddNode(graph, "Add", {name + ".matmul", name + ".bias"}, {name});
    return name;
  }

  std::string Norm(const std::string &input, const std::string &name) {
    AddInitializer<float>(graph, (name + ".weight").c_str(), {kHidden},
                          std::vector<float>(kHidden, 1.0f));
    AddInitializer<float>(graph, (name + ".bias").c_str(), {kHidden},
                          std::vector<float>(kHidden, 0.0f));
    auto &node =
        AddNode(graph, "LayerNormalization", {input, name + ".weight", name + ".bias"}, {name});
    AddAttribute<float>(node, "epsilon", 1e-5f);
    return name;
  }

  std::string Attention(const std::string &input, const std::string &memory,
                        const std::string &name, bool causal) {
    const std::string query = Linear(input, name + ".q_proj", kHidden, kHidden);
    const std::string key = Linear(memory, name + ".k_proj", kHidden, kHidden, false);
    const std::string value = Linear(memory, name + ".v_proj", kHidden, kHidden);
    if (fused) {
      auto &node = AddNode(graph, "Attention", {query, key, value}, {name + ".attention"});
      AddAttribute<int64_t>(node, "q_num_heads", kHeads);
      AddAttribute<int64_t>(node, "kv_num_heads", kHeads);
      AddAttribute<int64_t>(node, "is_causal", causal ? 1 : 0);
    } else {
      for (const auto &projection : {query, key, value}) {
        AddNode(graph, "Reshape", {projection, "split_heads"}, {projection + ".4d"});
        auto &node = AddNode(graph, "Transpose", {projection + ".4d"}, {projection + ".heads"});
        AddAttribute(node, "perm", std::vector<int64_t>{0, 2, 1, 3});
      }
      auto &transpose = AddNode(graph, "Transpose", {key + ".heads"}, {name + ".key_t"});
      AddAttribute(transpose, "perm", std::vector<int64_t>{0, 1, 3, 2});
      AddNode(graph, "MatMul", {query + ".heads", name + ".key_t"}, {name + ".scores"});
      AddNode(graph, "Mul", {name + ".scores", "attention_scale"}, {name + ".scaled"});
      std::string scores = name + ".scaled";
      if (causal) {
        AddNode(graph, "Add", {scores, "causal_mask"}, {name + ".masked"});
        scores = name + ".masked";
      }
      auto &softmax = AddNode(graph, "Softmax", {scores}, {name + ".probabilities"});
      AddAttribute<int64_t>(softmax, "axis", -1);
      AddNode(graph, "MatMul", {name + ".probabilities", value + ".heads"}, {name + ".context"});
      auto &merge = AddNode(graph, "Transpose", {name + ".context"}, {name + ".context_t"});
      AddAttribute(merge, "perm", std::vector<int64_t>{0, 2, 1, 3});
      AddNode(graph, "Reshape", {name + ".context_t", "merge_heads"}, {name + ".attention"});
    }
    return Linear(name + ".attention", name + ".out_proj", kHidden, kHidden);
  }

  std::string Layer(const std::string &input, const std::string &memory, const std::string &name,
                    bool decoder) {
    const std::string norm = Norm(input, name + ".self_attn_layer_norm");
    const std::string attention = Attention(norm, norm, name + ".self_attn", decoder);
    AddNode(graph, "Add", {input, attention}, {name + ".self_residual"});
    std::string residual = name + ".self_residual";
    if (decoder) {
      const std::string cross_norm = Norm(residual, name + ".encoder_attn_layer_norm");
      const std::string cross = Attention(cross_norm, memory, name + ".encoder_attn", false);
      AddNode(graph, "Add", {residual, cross}, {name + ".cross_residual"});
      residual = name + ".cross_residual";
    }
    const std::string mlp_norm = Norm(residual, name + ".final_layer_norm");
    const std::string fc1 = Linear(mlp_norm, name + ".fc1", kHidden, kIntermediate);
    AddNode(graph, "Gelu", {fc1}, {name + ".gelu"});
    const std::string fc2 = Linear(name + ".gelu", name + ".fc2", kIntermediate, kHidden);
    AddNode(graph, "Add", {residual, fc2}, {name + ".output"});
    return name + ".output";
  }
};

} // namespace

void RegisterWhisper4LayersLikeShapeInferenceCases(std::vector<TestCase> &registry,
                                                   TestMode /*mode*/) {
  for (bool fused : {false, true}) {
    const std::string name =
        std::string("test_cc_shape_inference_big_whisper_4_layers_like") + (fused ? "_fused" : "");
    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE);
    ModelProto &model = tc.emplace_model();
    InitModel(model, 10, {DefaultOpset(fused ? 23 : 20)});
    GraphProto &graph = *model.add_graph();
    graph.set_name(name);
    WhisperGraph builder{graph, fused};

    std::string audio = "input_features";
    for (int64_t conv = 1; conv <= 2; ++conv) {
      const std::string prefix = std::string("encoder.conv") + std::to_string(conv);
      builder.Weight(prefix + ".weight", {kHidden, conv == 1 ? kMelBins : kHidden, 3});
      builder.Weight(prefix + ".bias", {kHidden});
      auto &node = AddNode(graph, "Conv", {audio, prefix + ".weight", prefix + ".bias"}, {prefix});
      AddAttribute(node, "pads", std::vector<int64_t>{1, 1});
      AddAttribute(node, "strides", std::vector<int64_t>{conv});
      AddNode(graph, "Gelu", {prefix}, {prefix + ".gelu"});
      audio = prefix + ".gelu";
    }
    auto &transpose = AddNode(graph, "Transpose", {audio}, {"encoder.embeddings"});
    AddAttribute(transpose, "perm", std::vector<int64_t>{0, 2, 1});
    // Whisper uses fixed sinusoidal encoder positions and learned decoder positions.
    std::vector<float> positions(kSourceLength * kHidden);
    for (int64_t pos = 0; pos < kSourceLength; ++pos) {
      for (int64_t i = 0; i < kHidden / 2; ++i) {
        const double angle = pos * std::exp(-std::log(10000.0) * i / (kHidden / 2 - 1));
        positions[pos * kHidden + i] = static_cast<float>(std::sin(angle));
        positions[pos * kHidden + i + kHidden / 2] = static_cast<float>(std::cos(angle));
      }
    }
    AddInitializer<float>(graph, "encoder.embed_positions", {kSourceLength, kHidden}, positions);
    AddNode(graph, "Add", {"encoder.embeddings", "encoder.embed_positions"}, {"encoder.hidden"});

    if (!fused) {
      AddInitializer<int64_t>(graph, "split_heads", {4}, {0, 0, kHeads, kHeadSize});
      AddInitializer<int64_t>(graph, "merge_heads", {3}, {0, 0, kHidden});
      AddInitializer<float>(graph, "attention_scale", {}, {1.0f / std::sqrt(float(kHeadSize))});
      AddInitializer<int64_t>(graph, "sequence_axis", {}, {1});
      AddInitializer<int64_t>(graph, "range_start", {}, {0});
      AddInitializer<int64_t>(graph, "range_step", {}, {1});
      AddInitializer<int64_t>(graph, "row_axis", {1}, {1});
      AddInitializer<int64_t>(graph, "column_axis", {1}, {0});
      AddInitializer<float>(graph, "mask_zero", {}, {0.0f});
      AddInitializer<float>(graph, "mask_negative", {}, {-1e9f});
      AddNode(graph, "Shape", {"decoder_input_ids"}, {"decoder.shape"});
      AddNode(graph, "Gather", {"decoder.shape", "sequence_axis"}, {"decoder.length"});
      AddNode(graph, "Range", {"range_start", "decoder.length", "range_step"}, {"decoder.range"});
      AddNode(graph, "Unsqueeze", {"decoder.range", "row_axis"}, {"decoder.rows"});
      AddNode(graph, "Unsqueeze", {"decoder.range", "column_axis"}, {"decoder.columns"});
      AddNode(graph, "GreaterOrEqual", {"decoder.rows", "decoder.columns"}, {"decoder.allowed"});
      AddNode(graph, "Where", {"decoder.allowed", "mask_zero", "mask_negative"}, {"causal_mask"});
    }

    std::string encoder = "encoder.hidden";
    for (int layer = 0; layer < 4; ++layer)
      encoder =
          builder.Layer(encoder, "", std::string("encoder.layers.") + std::to_string(layer), false);
    encoder = builder.Norm(encoder, "encoder.layer_norm");

    builder.Weight("decoder.embed_tokens.weight", {kVocab, kHidden});
    builder.Weight("decoder.embed_positions.weight", {kMaxTargetLength, kHidden});
    AddNode(graph, "Gather", {"decoder.embed_tokens.weight", "decoder_input_ids"},
            {"decoder.tokens"});
    AddNode(graph, "Gather", {"decoder.embed_positions.weight", "decoder_position_ids"},
            {"decoder.positions"});
    AddNode(graph, "Add", {"decoder.tokens", "decoder.positions"}, {"decoder.hidden"});
    std::string decoder = "decoder.hidden";
    for (int layer = 0; layer < 4; ++layer)
      decoder = builder.Layer(decoder, encoder,
                              std::string("decoder.layers.") + std::to_string(layer), true);
    decoder = builder.Norm(decoder, "decoder.layer_norm");
    auto &head = AddNode(graph, "Transpose", {"decoder.embed_tokens.weight"}, {"proj_out.weight"});
    AddAttribute(head, "perm", std::vector<int64_t>{1, 0});
    AddNode(graph, "MatMul", {decoder, "proj_out.weight"}, {"logits"});

    AppendValueInfo(*graph.add_input(), "input_features", DataType::FLOAT,
                    {"batch_size", DimSpec(kMelBins), DimSpec(kFrames)});
    AppendValueInfo(*graph.add_input(), "decoder_input_ids", DataType::INT64,
                    {"batch_size", "sequence_length"});
    AppendValueInfo(*graph.add_input(), "decoder_position_ids", DataType::INT64,
                    {"sequence_length"});
    AppendValueInfo(*graph.add_output(), encoder.c_str(), DataType::FLOAT,
                    {"batch_size", DimSpec(kSourceLength), DimSpec(kHidden)});
    AppendValueInfo(*graph.add_output(), "logits", DataType::FLOAT,
                    {"batch_size", "sequence_length", DimSpec(kVocab)});
    for (int layer = 0; layer < 4; ++layer) {
      const std::string suffix = ".layers." + std::to_string(layer);
      AppendValueInfo(*graph.add_value_info(), ("encoder" + suffix + ".output").c_str(),
                      DataType::FLOAT, {"batch_size", DimSpec(kSourceLength), DimSpec(kHidden)});
      AppendValueInfo(*graph.add_value_info(), ("decoder" + suffix + ".output").c_str(),
                      DataType::FLOAT, {"batch_size", "sequence_length", DimSpec(kHidden)});
    }
    registry.push_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
