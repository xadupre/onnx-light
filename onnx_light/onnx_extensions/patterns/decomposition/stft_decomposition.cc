// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/decomposition/stft_decomposition.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::FreeInitializerName;
using collections::IsDefaultOp;
using core::builder::BuilderError;
using core::symbolic::TensorType;

constexpr std::size_t kMaxWeightElements = 16U * 1024U * 1024U;

bool ReadPositiveInt64Scalar(core::builder::GraphGraph &graph, const std::string &name,
                             int64_t &value) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr || tensor->data_type() != TensorProto::DataType::INT64 ||
      (tensor->dims_size() != 0 &&
       (tensor->dims_size() != 1 || tensor->dims(0) != static_cast<int64_t>(1)))) {
    return false;
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(*tensor, values) || values.size() != 1 || values[0] <= 0) {
    return false;
  }
  value = values[0];
  return true;
}

bool ReadOnesided(const NodeProto &node, bool &onesided) {
  onesided = true;
  for (const AttributeProto &attribute : node.attribute()) {
    if (attribute.name().value() != "onesided") {
      continue;
    }
    if (attribute.type() != AttributeProto::AttributeType::INT || !attribute.has_i() ||
        (attribute.i() != 0 && attribute.i() != 1)) {
      return false;
    }
    onesided = attribute.i() != 0;
  }
  return true;
}

bool HasSupportedSignal(core::builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasType(name) || !graph.HasShape(name)) {
    return false;
  }
  const TensorType type = graph.GetType(name);
  if (type != TensorType::kFloat && type != TensorType::kDouble) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  return shape.Rank() == 3 && shape[2].IsInt() && shape[2].AsInt() == 1;
}

bool ReadWindow(core::builder::GraphGraph &graph, const NodeProto &node, int64_t frame_length,
                std::vector<double> &window) {
  window.assign(static_cast<std::size_t>(frame_length), 1.0);
  if (node.input_size() < 3 || node.input()[2].value().empty()) {
    return true;
  }
  const std::string &name = node.input()[2].value();
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr || !graph.HasType(name) ||
      graph.GetType(name) != graph.GetType(node.input()[0].value()) || tensor->dims_size() != 1 ||
      tensor->dims(0) != frame_length) {
    return false;
  }
  return ReadFloatingValues(*tensor, window) &&
         window.size() == static_cast<std::size_t>(frame_length);
}

bool Validate(core::builder::GraphGraph &graph, const NodeProto &candidate, int64_t &frame_step,
              int64_t &frame_length, bool &onesided, std::vector<double> &window) {
  if (!IsDefaultOp(candidate, "STFT") || candidate.input_size() < 2 || candidate.input_size() > 4 ||
      candidate.output_size() != 1 || candidate.input()[0].value().empty() ||
      candidate.input()[1].value().empty() || candidate.input_size() < 4 ||
      candidate.input()[3].value().empty() ||
      !HasSupportedSignal(graph, candidate.input()[0].value()) ||
      !ReadPositiveInt64Scalar(graph, candidate.input()[1].value(), frame_step) ||
      !ReadPositiveInt64Scalar(graph, candidate.input()[3].value(), frame_length) ||
      !ReadOnesided(candidate, onesided)) {
    return false;
  }

  const int64_t bins = onesided ? frame_length / 2 + 1 : frame_length;
  if (static_cast<uint64_t>(frame_length) > std::numeric_limits<std::size_t>::max() ||
      static_cast<uint64_t>(bins) > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const std::size_t length = static_cast<std::size_t>(frame_length);
  const std::size_t bin_count = static_cast<std::size_t>(bins);
  if (length != 0 && bin_count > kMaxWeightElements / length) {
    return false;
  }

  if (graph.HasShape(candidate.input()[0].value())) {
    const core::symbolic::SymShape &shape = graph.GetShape(candidate.input()[0].value()).Shape();
    if (shape[1].IsInt() && shape[1].AsInt() < frame_length) {
      return false;
    }
  }
  return ReadWindow(graph, candidate, frame_length, window);
}

template <typename T>
void AddWeights(core::builder::GraphBuilder &builder, const std::string &real_name,
                const std::string &imag_name, int64_t frame_length, int64_t bins,
                const std::vector<double> &window) {
  const std::size_t length = static_cast<std::size_t>(frame_length);
  const std::size_t bin_count = static_cast<std::size_t>(bins);
  std::vector<T> real(bin_count * length);
  std::vector<T> imag(bin_count * length);
  for (std::size_t k = 0; k < bin_count; ++k) {
    for (std::size_t n = 0; n < length; ++n) {
      const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) *
                           static_cast<double>(n) / static_cast<double>(frame_length);
      const std::size_t index = k * length + n;
      real[index] = static_cast<T>(std::cos(angle) * window[n]);
      imag[index] = static_cast<T>(std::sin(angle) * window[n]);
    }
  }
  builder.MakeInitializer(MakeInitializer<T>(real_name.c_str(), {bins, 1, frame_length}, real));
  builder.MakeInitializer(MakeInitializer<T>(imag_name.c_str(), {bins, 1, frame_length}, imag));
}

NodeProto NamedNode(const NodeProto &source, const char *op_type,
                    const std::vector<std::string> &inputs, const std::vector<std::string> &outputs,
                    const std::string &suffix) {
  const std::string source_name =
      source.name().value().empty() ? source.output()[0].value() : source.name().value();
  const std::string name = "STFTDecomposition--" + source_name + suffix;
  NodeProto node = MakeNode(op_type, inputs, outputs, "", name.c_str());
  if (source.has_doc_string()) {
    node.set_doc_string(source.doc_string().value());
  }
  return node;
}

} // namespace

std::set<std::string> STFTDecompositionPattern::FastOpType() const { return {"STFT"}; }

core::builder::MatchResult STFTDecompositionPattern::Match(core::builder::GraphGraph &graph,
                                                           const NodeProto &candidate) const {
  int64_t frame_step = 0;
  int64_t frame_length = 0;
  bool onesided = true;
  std::vector<double> window;
  if (!Validate(graph, candidate, frame_step, frame_length, onesided, window)) {
    return NoMatch(candidate, "unsupported or unsafe STFT form");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
STFTDecompositionPattern::Apply(core::builder::GraphGraph &graph,
                                const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("STFTDecompositionPattern::Apply expects one STFT node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("STFTDecompositionPattern::Apply received an unsafe or inconsistent match.");
  }

  const NodeProto &stft = *nodes[0];
  int64_t frame_step = 0;
  int64_t frame_length = 0;
  bool onesided = true;
  std::vector<double> window;
  if (!Validate(graph, stft, frame_step, frame_length, onesided, window)) {
    throw BuilderError("STFTDecompositionPattern::Apply could not validate the STFT node.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const int64_t bins = onesided ? frame_length / 2 + 1 : frame_length;
  const std::string prefix =
      "STFTDecomposition--" +
      (stft.name().value().empty() ? stft.output()[0].value() : stft.name().value());
  const std::string real_weights = FreeInitializerName(builder, prefix + "_real_weights");
  const std::string imag_weights = FreeInitializerName(builder, prefix + "_imag_weights");
  const std::string axes = FreeInitializerName(builder, prefix + "_axes");

  if (graph.GetType(stft.input()[0].value()) == TensorType::kFloat) {
    AddWeights<float>(builder, real_weights, imag_weights, frame_length, bins, window);
  } else {
    AddWeights<double>(builder, real_weights, imag_weights, frame_length, bins, window);
  }
  builder.MakeInitializer(MakeInitializer<int64_t>(axes.c_str(), {1}, {3}));

  const std::string transposed_signal = builder.UniqueName(prefix + "_signal");
  const std::string real = builder.UniqueName(prefix + "_real");
  const std::string imag = builder.UniqueName(prefix + "_imag");
  const std::string real_unsqueezed = builder.UniqueName(prefix + "_real_unsqueezed");
  const std::string imag_unsqueezed = builder.UniqueName(prefix + "_imag_unsqueezed");
  const std::string complex = builder.UniqueName(prefix + "_complex");

  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto transpose_signal =
      NamedNode(stft, "Transpose", {stft.input()[0].value()}, {transposed_signal}, "_signal");
  AddAttribute<std::vector<int64_t>>(transpose_signal, "perm", {0, 2, 1});
  replacements.push_back(std::move(transpose_signal));

  NodeProto real_conv = NamedNode(stft, "Conv", {transposed_signal, real_weights}, {real}, "_real");
  AddAttribute<std::vector<int64_t>>(real_conv, "strides", {frame_step});
  replacements.push_back(std::move(real_conv));

  NodeProto imag_conv = NamedNode(stft, "Conv", {transposed_signal, imag_weights}, {imag}, "_imag");
  AddAttribute<std::vector<int64_t>>(imag_conv, "strides", {frame_step});
  replacements.push_back(std::move(imag_conv));

  replacements.push_back(
      NamedNode(stft, "Unsqueeze", {real, axes}, {real_unsqueezed}, "_real_unsqueeze"));
  replacements.push_back(
      NamedNode(stft, "Unsqueeze", {imag, axes}, {imag_unsqueezed}, "_imag_unsqueeze"));

  NodeProto concat =
      NamedNode(stft, "Concat", {real_unsqueezed, imag_unsqueezed}, {complex}, "_concat");
  AddAttribute<int64_t>(concat, "axis", 3);
  replacements.push_back(std::move(concat));

  NodeProto transpose_output =
      NamedNode(stft, "Transpose", {complex}, {stft.output()[0].value()}, "_output");
  AddAttribute<std::vector<int64_t>>(transpose_output, "perm", {0, 2, 1, 3});
  replacements.push_back(std::move(transpose_output));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
