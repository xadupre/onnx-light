// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/stft_pattern.h"

#include <algorithm>
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

struct MatchInfo {
  const NodeProto *signal_transpose = nullptr;
  const NodeProto *real_conv = nullptr;
  const NodeProto *imag_conv = nullptr;
  const NodeProto *real_unsqueeze = nullptr;
  const NodeProto *imag_unsqueeze = nullptr;
  const NodeProto *concat = nullptr;
  int64_t frame_step = 0;
  int64_t frame_length = 0;
  int64_t onesided = 0;
  int data_type = TensorProto::DataType::UNDEFINED;
  std::vector<double> window;
};

bool HasInts(const NodeProto &node, const char *name, const std::vector<int64_t> &expected) {
  std::vector<int64_t> values;
  return GetAttributeInts(node, name, values) && values == expected;
}

bool HasOnlyAttributes(const NodeProto &node, const std::vector<std::string> &allowed) {
  for (const AttributeProto &attribute : node.attribute()) {
    if (std::find(allowed.begin(), allowed.end(), attribute.name().value()) == allowed.end()) {
      return false;
    }
  }
  return true;
}

bool IsTranspose(const NodeProto &node, const std::vector<int64_t> &perm) {
  return IsDefaultOp(node, "Transpose") && node.input_size() == 1 && node.output_size() == 1 &&
         !node.input()[0].value().empty() && !node.output()[0].value().empty() &&
         HasOnlyAttributes(node, {"perm"}) && HasInts(node, "perm", perm);
}

bool IsPrivateTo(core::builder::GraphGraph &graph, const NodeProto &producer,
                 const std::vector<const NodeProto *> &expected_consumers) {
  if (producer.output_size() != 1 || producer.output()[0].value().empty()) {
    return false;
  }
  const std::string &output = producer.output()[0].value();
  if (graph.IsOutput(output) || graph.IsUsedBySubgraph(output)) {
    return false;
  }
  const std::vector<const NodeProto *> &actual = graph.NextNodes(output);
  if (actual.size() != expected_consumers.size()) {
    return false;
  }
  for (const NodeProto *expected : expected_consumers) {
    if (std::find(actual.begin(), actual.end(), expected) == actual.end()) {
      return false;
    }
  }
  return true;
}

bool ReadAxisThree(core::builder::GraphGraph &graph, const NodeProto &node) {
  if (!IsDefaultOp(node, "Unsqueeze") || node.input_size() != 2 || node.output_size() != 1 ||
      node.input()[0].value().empty() || node.input()[1].value().empty() ||
      node.output()[0].value().empty() || node.attribute_size() != 0) {
    return false;
  }
  const TensorProto *axes = graph.GetComputedConstant(node.input()[1].value());
  std::vector<int64_t> values;
  return axes != nullptr && axes->data_type() == TensorProto::DataType::INT64 &&
         axes->dims_size() == 1 && axes->dims(0) == 1 && ReadIntegerValues(*axes, values) &&
         values == std::vector<int64_t>{3};
}

bool ReadConv(core::builder::GraphGraph &graph, const NodeProto &conv, int64_t &stride,
              const TensorProto *&weights, std::vector<double> &values) {
  if (!IsDefaultOp(conv, "Conv") || conv.input_size() != 2 || conv.output_size() != 1 ||
      conv.input()[0].value().empty() || conv.input()[1].value().empty() ||
      conv.output()[0].value().empty() ||
      !HasOnlyAttributes(conv,
                         {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"}) ||
      GetAttributeOr<std::string>(conv, "auto_pad", "NOTSET") != "NOTSET" ||
      GetAttributeOr<int64_t>(conv, "group", 1) != 1) {
    return false;
  }

  std::vector<int64_t> dilations;
  if (GetAttributeInts(conv, "dilations", dilations) && dilations != std::vector<int64_t>{1}) {
    return false;
  }
  std::vector<int64_t> pads;
  if (GetAttributeInts(conv, "pads", pads) && pads != std::vector<int64_t>{0, 0}) {
    return false;
  }
  std::vector<int64_t> strides;
  if (!GetAttributeInts(conv, "strides", strides)) {
    strides = {1};
  }
  if (strides.size() != 1 || strides[0] <= 0) {
    return false;
  }
  stride = strides[0];

  weights = graph.GetComputedConstant(conv.input()[1].value());
  if (weights == nullptr ||
      (weights->data_type() != TensorProto::DataType::FLOAT &&
       weights->data_type() != TensorProto::DataType::DOUBLE) ||
      weights->dims_size() != 3 || weights->dims(0) <= 0 || weights->dims(1) != 1 ||
      weights->dims(2) <= 0 || !ReadFloatingValues(*weights, values)) {
    return false;
  }
  const int64_t bins = weights->dims(0);
  const int64_t frame_length = weights->dims(2);
  if (static_cast<uint64_t>(bins) > std::numeric_limits<std::size_t>::max() ||
      static_cast<uint64_t>(frame_length) > std::numeric_limits<std::size_t>::max() ||
      static_cast<std::size_t>(bins) >
          std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(frame_length) ||
      values.size() != static_cast<std::size_t>(bins) * static_cast<std::size_t>(frame_length) ||
      !std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); })) {
    return false;
  }
  std::vector<int64_t> kernel_shape;
  return !GetAttributeInts(conv, "kernel_shape", kernel_shape) ||
         kernel_shape == std::vector<int64_t>{frame_length};
}

bool CloseEnough(double actual, double expected, double window, int data_type) {
  const double tolerance = data_type == TensorProto::DataType::FLOAT ? 2.0e-5 : 2.0e-12;
  const double scale = std::max(std::abs(window), std::abs(expected));
  return scale == 0.0 ? actual == expected : std::abs(actual - expected) <= tolerance * scale;
}

bool SameDims(const TensorProto &left, const TensorProto &right) {
  if (left.dims_size() != right.dims_size()) {
    return false;
  }
  for (int i = 0; i < left.dims_size(); ++i) {
    if (left.dims(i) != right.dims(i)) {
      return false;
    }
  }
  return true;
}

bool ValidateDftWeights(const TensorProto &real_weights, const std::vector<double> &real,
                        const std::vector<double> &imag, MatchInfo &info) {
  const int64_t bins = real_weights.dims(0);
  const int64_t frame_length = real_weights.dims(2);
  const int64_t half_bins = frame_length / 2 + 1;
  if (bins == half_bins) {
    info.onesided = 1;
  } else if (bins == frame_length) {
    info.onesided = 0;
  } else {
    return false;
  }

  info.frame_length = frame_length;
  info.data_type = real_weights.data_type();
  info.window.assign(real.begin(), real.begin() + frame_length);
  for (int64_t k = 0; k < bins; ++k) {
    for (int64_t n = 0; n < frame_length; ++n) {
      const std::size_t index = static_cast<std::size_t>(k * frame_length + n);
      const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) *
                           static_cast<double>(n) / static_cast<double>(frame_length);
      if (!CloseEnough(real[index], std::cos(angle) * info.window[n], info.window[n],
                       info.data_type) ||
          !CloseEnough(imag[index], std::sin(angle) * info.window[n], info.window[n],
                       info.data_type)) {
        return false;
      }
    }
  }
  return true;
}

bool Validate(core::builder::GraphGraph &graph, const NodeProto &candidate, MatchInfo &info) {
  if (graph.Builder().OpsetVersion("") < 17 || !IsTranspose(candidate, {0, 2, 1, 3})) {
    return false;
  }

  info.concat = graph.NodeBefore(candidate.input()[0].value());
  if (info.concat == nullptr || !IsDefaultOp(*info.concat, "Concat") ||
      info.concat->input_size() != 2 || info.concat->output_size() != 1 ||
      !HasOnlyAttributes(*info.concat, {"axis"}) ||
      GetAttributeOr<int64_t>(*info.concat, "axis", 0) != 3 ||
      !IsPrivateTo(graph, *info.concat, {&candidate})) {
    return false;
  }

  info.real_unsqueeze = graph.NodeBefore(info.concat->input()[0].value());
  info.imag_unsqueeze = graph.NodeBefore(info.concat->input()[1].value());
  if (info.real_unsqueeze == nullptr || info.imag_unsqueeze == nullptr ||
      info.real_unsqueeze == info.imag_unsqueeze || !ReadAxisThree(graph, *info.real_unsqueeze) ||
      !ReadAxisThree(graph, *info.imag_unsqueeze) ||
      !IsPrivateTo(graph, *info.real_unsqueeze, {info.concat}) ||
      !IsPrivateTo(graph, *info.imag_unsqueeze, {info.concat})) {
    return false;
  }

  info.real_conv = graph.NodeBefore(info.real_unsqueeze->input()[0].value());
  info.imag_conv = graph.NodeBefore(info.imag_unsqueeze->input()[0].value());
  const TensorProto *real_weights = nullptr;
  const TensorProto *imag_weights = nullptr;
  std::vector<double> real;
  std::vector<double> imag;
  int64_t real_stride = 0;
  int64_t imag_stride = 0;
  if (info.real_conv == nullptr || info.imag_conv == nullptr || info.real_conv == info.imag_conv ||
      !ReadConv(graph, *info.real_conv, real_stride, real_weights, real) ||
      !ReadConv(graph, *info.imag_conv, imag_stride, imag_weights, imag) ||
      real_stride != imag_stride ||
      info.real_conv->input()[0].value() != info.imag_conv->input()[0].value() ||
      real_weights->data_type() != imag_weights->data_type() ||
      !SameDims(*real_weights, *imag_weights) ||
      !IsPrivateTo(graph, *info.real_conv, {info.real_unsqueeze}) ||
      !IsPrivateTo(graph, *info.imag_conv, {info.imag_unsqueeze})) {
    return false;
  }
  info.frame_step = real_stride;

  info.signal_transpose = graph.NodeBefore(info.real_conv->input()[0].value());
  if (info.signal_transpose == nullptr || !IsTranspose(*info.signal_transpose, {0, 2, 1}) ||
      !IsPrivateTo(graph, *info.signal_transpose, {info.real_conv, info.imag_conv})) {
    return false;
  }
  const std::string &signal = info.signal_transpose->input()[0].value();
  if (!graph.HasType(signal) || !graph.HasShape(signal) ||
      ((real_weights->data_type() == TensorProto::DataType::FLOAT &&
        graph.GetType(signal) != TensorType::kFloat) ||
       (real_weights->data_type() == TensorProto::DataType::DOUBLE &&
        graph.GetType(signal) != TensorType::kDouble))) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(signal).Shape();
  if (shape.Rank() != 3 || !shape[2].IsInt() || shape[2].AsInt() != 1 ||
      (shape[1].IsInt() && shape[1].AsInt() < real_weights->dims(2))) {
    return false;
  }
  return ValidateDftWeights(*real_weights, real, imag, info);
}

template <typename T>
TensorProto MakeWindow(const std::string &name, const std::vector<double> &values) {
  std::vector<T> converted;
  converted.reserve(values.size());
  for (double value : values) {
    converted.push_back(static_cast<T>(value));
  }
  return MakeInitializer<T>(name.c_str(), {static_cast<int64_t>(values.size())}, converted);
}

} // namespace

std::set<std::string> STFTFusionPattern::FastOpType() const { return {"Transpose"}; }

core::builder::MatchResult STFTFusionPattern::Match(core::builder::GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  MatchInfo info;
  if (!Validate(graph, candidate, info)) {
    return NoMatch(candidate, "not a private canonical convolution-based DFT");
  }
  return core::builder::MatchResult{this,
                                    {info.signal_transpose, info.real_conv, info.imag_conv,
                                     info.real_unsqueeze, info.imag_unsqueeze, info.concat,
                                     &candidate},
                                    &candidate};
}

utils::RepeatedProtoField<NodeProto>
STFTFusionPattern::Apply(core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 7 || nodes.back() == nullptr) {
    throw BuilderError("STFTFusionPattern::Apply expects seven matched nodes.");
  }
  MatchInfo info;
  if (!Validate(graph, *nodes.back(), info)) {
    throw BuilderError("STFTFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const std::vector<const NodeProto *> expected = {
      info.signal_transpose, info.real_conv, info.imag_conv, info.real_unsqueeze,
      info.imag_unsqueeze,   info.concat,    nodes.back()};
  if (nodes != expected) {
    throw BuilderError("STFTFusionPattern::Apply received an inconsistent node sequence.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const NodeProto &output_transpose = *nodes.back();
  const std::string base = "STFTFusion--" + (output_transpose.name().value().empty()
                                                 ? output_transpose.output()[0].value()
                                                 : output_transpose.name().value());
  const std::string frame_step = FreeInitializerName(builder, base + "_frame_step");
  const std::string frame_length = FreeInitializerName(builder, base + "_frame_length");
  const std::string window = FreeInitializerName(builder, base + "_window");
  builder.MakeInitializer(MakeInitializer<int64_t>(frame_step.c_str(), {}, {info.frame_step}));
  builder.MakeInitializer(MakeInitializer<int64_t>(frame_length.c_str(), {}, {info.frame_length}));
  if (info.data_type == TensorProto::DataType::FLOAT) {
    builder.MakeInitializer(MakeWindow<float>(window, info.window));
  } else {
    builder.MakeInitializer(MakeWindow<double>(window, info.window));
  }

  NodeProto stft = MakeNode(
      "STFT", {info.signal_transpose->input()[0].value(), frame_step, window, frame_length},
      {output_transpose.output()[0].value()}, "", builder.UniqueName(base + "_node").c_str());
  AddAttribute<int64_t>(stft, "onesided", info.onesided);
  if (output_transpose.has_doc_string()) {
    stft.set_doc_string(output_transpose.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(stft));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
