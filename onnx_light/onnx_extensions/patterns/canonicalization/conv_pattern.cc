// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultConv(const NodeProto &node) {
  return node.op_type().value() == "Conv" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

bool IsDefaultPad(const NodeProto &node) {
  return node.op_type().value() == "Pad" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

bool IsDefaultOp(const NodeProto &node, const std::string &op_type) {
  return node.op_type().value() == op_type &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

bool IsSupportedFloatingType(const TensorProto &tensor) {
  return tensor.data_type() == TensorProto::DataType::FLOAT ||
         tensor.data_type() == TensorProto::DataType::DOUBLE;
}

bool HasTensorShape(const TensorProto &tensor, const std::vector<int64_t> &dims) {
  if (tensor.dims_size() != static_cast<int>(dims.size())) {
    return false;
  }
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (tensor.dims()[i] != dims[i]) {
      return false;
    }
  }
  return true;
}

bool TensorElementCount(const TensorProto &tensor, std::size_t &count) {
  count = 1;
  for (int64_t dim : tensor.dims()) {
    if (dim < 0 || (dim != 0 && count > std::numeric_limits<std::size_t>::max() /
                                            static_cast<std::size_t>(dim))) {
      return false;
    }
    count *= static_cast<std::size_t>(dim);
  }
  return true;
}

bool ReadFloatingTensor(const TensorProto &tensor, std::vector<double> &values) {
  std::size_t count = 0;
  if (!IsSupportedFloatingType(tensor) || !TensorElementCount(tensor, count) ||
      !ReadFloatingValues(tensor, values) || values.size() != count) {
    return false;
  }
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool ValuesRepresentable(const TensorProto &tensor, const std::vector<double> &values) {
  const double limit = tensor.data_type() == TensorProto::DataType::FLOAT
                           ? static_cast<double>(std::numeric_limits<float>::max())
                           : std::numeric_limits<double>::max();
  return std::all_of(values.begin(), values.end(), [limit](double value) {
    return std::isfinite(value) && std::abs(value) <= limit;
  });
}

TensorProto MakeFloatingInitializer(const TensorProto &source, const std::string &name,
                                    const std::vector<int64_t> &dims,
                                    const std::vector<double> &values) {
  if (source.data_type() == TensorProto::DataType::FLOAT) {
    std::vector<float> converted;
    converted.reserve(values.size());
    for (double value : values) {
      converted.push_back(static_cast<float>(value));
    }
    return MakeInitializer<float>(name.c_str(), dims, converted);
  }
  return MakeInitializer<double>(name.c_str(), dims, values);
}

std::vector<int64_t> TensorDims(const TensorProto &tensor) {
  return std::vector<int64_t>(tensor.dims().begin(), tensor.dims().end());
}

std::string FreeInitializerName(core::builder::GraphBuilder &builder, const std::string &base) {
  if (!builder.HasName(base)) {
    return base;
  }
  for (int suffix = 0;; ++suffix) {
    const std::string candidate = base + "_" + std::to_string(suffix);
    if (!builder.HasName(candidate)) {
      return candidate;
    }
  }
}

const TensorProto *GetFloatingConstant(core::builder::GraphGraph &graph, const std::string &name) {
  if (name.empty() || !graph.IsConstant(name)) {
    return nullptr;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && IsSupportedFloatingType(*tensor) ? tensor : nullptr;
}

const NodeProto *SingleFollowingNode(core::builder::GraphGraph &graph, const NodeProto &conv,
                                     const std::string &op_type) {
  if (!IsDefaultConv(conv) || conv.input_size() < 2 || conv.input_size() > 3 ||
      conv.output_size() != 1 || conv.output()[0].value().empty() ||
      graph.IsUsedMoreThanOnce(conv.output()[0].value())) {
    return nullptr;
  }
  const std::vector<const NodeProto *> &consumers = graph.NextNodes(conv.output()[0].value());
  if (consumers.size() != 1 || consumers[0] == nullptr || !IsDefaultOp(*consumers[0], op_type)) {
    return nullptr;
  }
  return consumers[0];
}

const TensorProto *BinaryConstant(core::builder::GraphGraph &graph, const NodeProto &binary,
                                  const std::string &conv_output) {
  if (binary.input_size() != 2 || binary.output_size() != 1 || binary.output()[0].value().empty()) {
    return nullptr;
  }
  const bool first_is_conv = binary.input()[0].value() == conv_output;
  const bool second_is_conv = binary.input()[1].value() == conv_output;
  if (first_is_conv == second_is_conv) {
    return nullptr;
  }
  return GetFloatingConstant(graph, binary.input()[first_is_conv ? 1 : 0].value());
}

bool ReadConvConstants(core::builder::GraphGraph &graph, const NodeProto &conv,
                       const TensorProto *&weights, const TensorProto *&bias,
                       std::vector<double> &weight_values, std::vector<double> &bias_values,
                       int64_t &channels) {
  weights = GetFloatingConstant(graph, conv.input()[1].value());
  if (weights == nullptr || weights->dims_size() <= 2 || weights->dims()[0] <= 0 ||
      !ReadFloatingTensor(*weights, weight_values)) {
    return false;
  }
  for (int64_t dim : weights->dims()) {
    if (dim <= 0) {
      return false;
    }
  }
  channels = weights->dims()[0];
  bias = nullptr;
  bias_values.assign(static_cast<std::size_t>(channels), 0.0);
  if (conv.input_size() == 3 && !conv.input()[2].value().empty()) {
    bias = GetFloatingConstant(graph, conv.input()[2].value());
    if (bias == nullptr || bias->data_type() != weights->data_type() ||
        !HasTensorShape(*bias, {channels}) || !ReadFloatingTensor(*bias, bias_values)) {
      return false;
    }
  }
  return true;
}

bool ReadChannelBroadcast(const TensorProto &tensor, int weight_rank, int64_t channels,
                          bool allow_scalar, std::vector<double> &values) {
  if (!ReadFloatingTensor(tensor, values)) {
    return false;
  }
  if (allow_scalar && tensor.dims_size() == 0) {
    return values.size() == 1;
  }
  int channel_axis = -1;
  if (tensor.dims_size() == weight_rank) {
    channel_axis = 1;
  } else if (tensor.dims_size() == weight_rank - 1) {
    channel_axis = 0;
  } else {
    return false;
  }
  if (tensor.dims()[channel_axis] != channels) {
    return false;
  }
  for (int i = 0; i < tensor.dims_size(); ++i) {
    if (i != channel_axis && tensor.dims()[i] != 1) {
      return false;
    }
  }
  return values.size() == static_cast<std::size_t>(channels);
}

NodeProto FoldedConv(const NodeProto &conv, const NodeProto &tail,
                     const std::vector<std::string> &inputs, const std::string &pattern_name) {
  const std::vector<std::string> outputs = {tail.output()[0].value()};
  const std::string name = pattern_name + "--" + conv.name().value();
  NodeProto replacement = MakeNode("Conv", inputs, outputs, "", name.c_str());
  for (const AttributeProto &attribute : conv.attribute()) {
    replacement.mutable_attribute()->push_back(attribute);
  }
  if (conv.has_doc_string()) {
    replacement.set_doc_string(conv.doc_string().value());
  }
  return replacement;
}

// Returns ``true`` when every element of ``tensor`` is zero. Only numeric
// integer and float element types are inspected; other encodings are reported
// as non-zero (conservative).
bool IsAllZero(const TensorProto &tensor) {
  std::vector<double> floats;
  if (ReadFloatingValues(tensor, floats)) {
    for (double value : floats) {
      if (value != 0.0) {
        return false;
      }
    }
    return !floats.empty();
  }
  std::vector<int64_t> integers;
  if (ReadIntegerValues(tensor, integers)) {
    for (int64_t value : integers) {
      if (value != 0) {
        return false;
      }
    }
    return !integers.empty();
  }
  return false;
}

// Returns ``true`` when ``name`` is materialized as an all-zero tensor or is
// produced by expanding one to a constant target shape.
bool IsKnownAllZero(core::builder::GraphGraph &graph, const std::string &name) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor != nullptr) {
    return IsAllZero(*tensor);
  }

  const NodeProto *producer = graph.NodeBefore(name);
  if (producer == nullptr || producer->op_type().value() != "Expand" ||
      NormaliseDomain(producer->domain().value()) != kDefaultOnnxDomain ||
      producer->input_size() < 2 || !graph.IsConstant(producer->input()[1].value())) {
    return false;
  }
  const TensorProto *expanded = graph.GetComputedConstant(producer->input()[0].value());
  return expanded != nullptr && IsAllZero(*expanded);
}

// Reads the optional string input at ``index``. Returns an empty string when
// the input is absent or explicitly omitted.
std::string OptionalInput(const NodeProto &node, int index) {
  if (index >= node.input_size()) {
    return {};
  }
  return node.input()[index].value();
}

// Reconstructs the full ``[begin_0, ..., begin_{ndim-1}, end_0, ...]`` padding
// array of the Pad node ``pad_node`` into ``full_pads`` and returns ``true`` on
// success. ``ndim`` receives the tensor rank. Fails when the pads or axes
// constants are missing, the constant value is a non-zero scalar, or an axis is
// out of range.
bool ReadPadPads(core::builder::GraphGraph &graph, const NodeProto &pad_node,
                 std::vector<int64_t> &full_pads, std::size_t &ndim) {
  if (pad_node.input_size() < 2 || !graph.IsConstant(pad_node.input()[1].value())) {
    return false;
  }
  const TensorProto *pads_cst = graph.GetComputedConstant(pad_node.input()[1].value());
  if (pads_cst == nullptr) {
    return false;
  }
  std::vector<int64_t> pads_values;
  if (!ReadIntegerValues(*pads_cst, pads_values) || pads_values.size() % 2 != 0) {
    return false;
  }
  ndim = pads_values.size() / 2;
  if (ndim < 2) {
    return false;
  }

  const std::string constant_value = OptionalInput(pad_node, 2);
  if (!constant_value.empty()) {
    if (!graph.IsConstant(constant_value)) {
      return false;
    }
    const TensorProto *cv = graph.GetComputedConstant(constant_value);
    double scalar = 0.0;
    if (cv == nullptr || !ReadScalarAsDouble(*cv, scalar) || scalar != 0.0) {
      return false;
    }
  }

  const std::string axes_name = OptionalInput(pad_node, 3);
  if (axes_name.empty()) {
    full_pads = std::move(pads_values);
    return true;
  }
  if (!graph.IsConstant(axes_name)) {
    return false;
  }
  const TensorProto *axes_cst = graph.GetComputedConstant(axes_name);
  std::vector<int64_t> axes;
  if (axes_cst == nullptr || !ReadIntegerValues(*axes_cst, axes)) {
    return false;
  }
  full_pads.assign(2 * ndim, 0);
  for (std::size_t idx = 0; idx < axes.size(); ++idx) {
    int64_t axis = axes[idx];
    if (axis < 0) {
      axis += static_cast<int64_t>(ndim);
    }
    if (axis < 0 || static_cast<std::size_t>(axis) >= ndim ||
        idx + axes.size() >= pads_values.size()) {
      return false;
    }
    full_pads[static_cast<std::size_t>(axis)] = pads_values[idx];
    full_pads[static_cast<std::size_t>(axis) + ndim] = pads_values[idx + axes.size()];
  }
  return true;
}

} // namespace

std::set<std::string> ConvBiasNullPattern::FastOpType() const { return {"Conv"}; }

core::builder::MatchResult ConvBiasNullPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsDefaultConv(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Conv");
  }
  if (candidate.input_size() < 3 || candidate.input()[2].value().empty()) {
    return NoMatch(candidate, "the Conv node has no bias input");
  }
  if (!graph.IsConstant(candidate.input()[2].value())) {
    return NoMatch(candidate, "the Conv bias is not a constant");
  }
  if (!IsKnownAllZero(graph, candidate.input()[2].value())) {
    return NoMatch(candidate, "the Conv bias is not a known all-zero constant");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConvBiasNullPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ConvBiasNullPattern::Apply expects one Conv node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConvBiasNullPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];

  const std::string name = "ConvBiasNullPattern--" + node.name().value();
  std::vector<std::string> outputs;
  outputs.reserve(node.output_size());
  for (const auto &output : node.output()) {
    outputs.push_back(output.value());
  }
  NodeProto replacement = MakeNode("Conv", {node.input()[0].value(), node.input()[1].value()},
                                   outputs, "", name.c_str());
  for (const AttributeProto &attribute : node.attribute()) {
    replacement.mutable_attribute()->push_back(attribute);
  }
  if (node.has_doc_string()) {
    replacement.set_doc_string(node.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> PadConvPattern::FastOpType() const { return {"Conv"}; }

core::builder::MatchResult PadConvPattern::Match(core::builder::GraphGraph &graph,
                                                 const NodeProto &candidate) const {
  if (!IsDefaultConv(candidate) || candidate.input_size() < 1) {
    return NoMatch(candidate, "candidate is not a default-domain Conv");
  }
  if (GetAttributeOr<std::string>(candidate, "auto_pad", "NOTSET") != "NOTSET") {
    return NoMatch(candidate, "the Conv node uses auto_pad");
  }
  const NodeProto *pad_node = graph.NodeBefore(candidate.input()[0].value());
  if (pad_node == nullptr || !IsDefaultPad(*pad_node)) {
    return NoMatch(candidate, "the Conv input is not produced by a Pad node");
  }
  if (graph.NextNodes(candidate.input()[0].value()).size() != 1) {
    return NoMatch(candidate, "the Pad output has another use");
  }
  if (GetAttributeOr<std::string>(*pad_node, "mode", "constant") != "constant") {
    return NoMatch(candidate, "the Pad node does not use constant mode");
  }

  std::vector<int64_t> pads;
  std::size_t ndim = 0;
  if (!ReadPadPads(graph, *pad_node, pads, ndim)) {
    return NoMatch(candidate, "the Pad padding cannot be resolved to a zero-padded constant");
  }
  if (pads[0] != 0 || pads[1] != 0 || pads[ndim] != 0 || pads[ndim + 1] != 0) {
    return NoMatch(candidate, "the Pad node pads the batch or channel dimension");
  }
  for (std::size_t i = 2; i < ndim; ++i) {
    if (pads[i] < 0 || pads[ndim + i] < 0) {
      return NoMatch(candidate, "the Pad node uses negative spatial padding");
    }
  }
  return core::builder::MatchResult{this, {pad_node, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
PadConvPattern::Apply(core::builder::GraphGraph &graph,
                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("PadConvPattern::Apply expects one Pad node and one Conv node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("PadConvPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &pad_node = *nodes[0];
  const NodeProto &conv_node = *nodes[1];

  std::vector<int64_t> pads;
  std::size_t ndim = 0;
  if (!ReadPadPads(graph, pad_node, pads, ndim)) {
    throw BuilderError("PadConvPattern::Apply could not read the Pad padding.");
  }
  const std::size_t n_spatial = ndim - 2;

  std::vector<int64_t> existing_pads;
  GetAttributeInts(conv_node, "pads", existing_pads);
  if (existing_pads.size() < 2 * n_spatial) {
    existing_pads.resize(2 * n_spatial, 0);
  }
  std::vector<int64_t> new_pads(2 * n_spatial, 0);
  for (std::size_t i = 0; i < n_spatial; ++i) {
    new_pads[i] = existing_pads[i] + pads[2 + i];
    new_pads[n_spatial + i] = existing_pads[n_spatial + i] + pads[ndim + 2 + i];
  }

  std::vector<std::string> inputs;
  inputs.reserve(conv_node.input_size());
  inputs.push_back(pad_node.input()[0].value());
  for (int i = 1; i < conv_node.input_size(); ++i) {
    inputs.push_back(conv_node.input()[i].value());
  }
  std::vector<std::string> outputs;
  outputs.reserve(conv_node.output_size());
  for (const auto &output : conv_node.output()) {
    outputs.push_back(output.value());
  }

  const std::string name = "PadConvPattern--" + conv_node.name().value();
  NodeProto replacement = MakeNode("Conv", inputs, outputs, "", name.c_str());
  for (const AttributeProto &attribute : conv_node.attribute()) {
    if (attribute.name().value() != "pads") {
      replacement.mutable_attribute()->push_back(attribute);
    }
  }
  AddAttribute<std::vector<int64_t>>(replacement, "pads", new_pads);
  if (conv_node.has_doc_string()) {
    replacement.set_doc_string(conv_node.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> ConvAddFusionPattern::FastOpType() const { return {"Conv"}; }

core::builder::MatchResult ConvAddFusionPattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("") < 7) {
    return NoMatch(candidate, "the default-domain opset predates multidirectional broadcasting");
  }
  const NodeProto *add = SingleFollowingNode(graph, candidate, "Add");
  if (add == nullptr) {
    return NoMatch(candidate,
                   "the Conv output is not exclusively consumed by a default-domain Add");
  }
  const TensorProto *addend = BinaryConstant(graph, *add, candidate.output()[0].value());
  if (addend == nullptr) {
    return NoMatch(candidate, "the other Add input is not a materialized floating constant");
  }

  const TensorProto *weights = nullptr;
  const TensorProto *bias = nullptr;
  std::vector<double> weight_values;
  std::vector<double> bias_values;
  int64_t channels = 0;
  if (!ReadConvConstants(graph, candidate, weights, bias, weight_values, bias_values, channels)) {
    return NoMatch(candidate, "the Conv weights or bias are not foldable floating constants");
  }
  std::vector<double> add_values;
  if (addend->data_type() != weights->data_type() ||
      !ReadChannelBroadcast(*addend, weights->dims_size(), channels, false, add_values)) {
    return NoMatch(candidate, "the Add constant has an incompatible dtype or channel broadcast");
  }
  for (std::size_t i = 0; i < bias_values.size(); ++i) {
    bias_values[i] += add_values[i];
  }
  if (!ValuesRepresentable(*weights, bias_values)) {
    return NoMatch(candidate, "the folded Conv bias is not finite in the source dtype");
  }
  return core::builder::MatchResult{this, {&candidate, add}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConvAddFusionPattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      Match(graph, *nodes[0]).nodes != nodes) {
    throw BuilderError("ConvAddFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &conv = *nodes[0];
  const NodeProto &add = *nodes[1];
  const TensorProto *weights = nullptr;
  const TensorProto *bias = nullptr;
  std::vector<double> weight_values;
  std::vector<double> bias_values;
  int64_t channels = 0;
  ReadConvConstants(graph, conv, weights, bias, weight_values, bias_values, channels);
  const TensorProto *addend = BinaryConstant(graph, add, conv.output()[0].value());
  std::vector<double> add_values;
  ReadChannelBroadcast(*addend, weights->dims_size(), channels, false, add_values);
  for (std::size_t i = 0; i < bias_values.size(); ++i) {
    bias_values[i] += add_values[i];
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string bias_name =
      FreeInitializerName(builder, "ConvAddFusion_B_" + addend->name().value());
  builder.MakeInitializer(MakeFloatingInitializer(*weights, bias_name, {channels}, bias_values));
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(FoldedConv(conv, add,
                                    {conv.input()[0].value(), conv.input()[1].value(), bias_name},
                                    "ConvAddFusionPattern"));
  return replacements;
}

std::set<std::string> ConvMulFusionPattern::FastOpType() const { return {"Conv"}; }

core::builder::MatchResult ConvMulFusionPattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("") < 7) {
    return NoMatch(candidate, "the default-domain opset predates multidirectional broadcasting");
  }
  const NodeProto *mul = SingleFollowingNode(graph, candidate, "Mul");
  if (mul == nullptr) {
    return NoMatch(candidate,
                   "the Conv output is not exclusively consumed by a default-domain Mul");
  }
  const TensorProto *multiplier = BinaryConstant(graph, *mul, candidate.output()[0].value());
  if (multiplier == nullptr) {
    return NoMatch(candidate, "the other Mul input is not a materialized floating constant");
  }

  const TensorProto *weights = nullptr;
  const TensorProto *bias = nullptr;
  std::vector<double> weight_values;
  std::vector<double> bias_values;
  int64_t channels = 0;
  if (!ReadConvConstants(graph, candidate, weights, bias, weight_values, bias_values, channels)) {
    return NoMatch(candidate, "the Conv weights or bias are not foldable floating constants");
  }
  std::vector<double> mul_values;
  if (multiplier->data_type() != weights->data_type() ||
      !ReadChannelBroadcast(*multiplier, weights->dims_size(), channels, true, mul_values)) {
    return NoMatch(candidate, "the Mul constant has an incompatible dtype or channel broadcast");
  }
  const std::size_t channel_block = weight_values.size() / static_cast<std::size_t>(channels);
  for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
    const double scale = mul_values.size() == 1 ? mul_values[0] : mul_values[channel];
    for (std::size_t i = 0; i < channel_block; ++i) {
      weight_values[channel * channel_block + i] *= scale;
    }
    bias_values[channel] *= scale;
  }
  if (!ValuesRepresentable(*weights, weight_values) ||
      !ValuesRepresentable(*weights, bias_values)) {
    return NoMatch(candidate, "the folded Conv constants are not finite in the source dtype");
  }
  return core::builder::MatchResult{this, {&candidate, mul}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConvMulFusionPattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      Match(graph, *nodes[0]).nodes != nodes) {
    throw BuilderError("ConvMulFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &conv = *nodes[0];
  const NodeProto &mul = *nodes[1];
  const TensorProto *weights = nullptr;
  const TensorProto *bias = nullptr;
  std::vector<double> weight_values;
  std::vector<double> bias_values;
  int64_t channels = 0;
  ReadConvConstants(graph, conv, weights, bias, weight_values, bias_values, channels);
  const TensorProto *multiplier = BinaryConstant(graph, mul, conv.output()[0].value());
  std::vector<double> mul_values;
  ReadChannelBroadcast(*multiplier, weights->dims_size(), channels, true, mul_values);

  const std::size_t channel_block = weight_values.size() / static_cast<std::size_t>(channels);
  for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
    const double scale = mul_values.size() == 1 ? mul_values[0] : mul_values[channel];
    for (std::size_t i = 0; i < channel_block; ++i) {
      weight_values[channel * channel_block + i] *= scale;
    }
    bias_values[channel] *= scale;
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string weight_name =
      FreeInitializerName(builder, "ConvMulFusion_W_" + weights->name().value());
  builder.MakeInitializer(
      MakeFloatingInitializer(*weights, weight_name, TensorDims(*weights), weight_values));
  std::vector<std::string> inputs = {conv.input()[0].value(), weight_name};
  if (bias != nullptr) {
    const std::string bias_name =
        FreeInitializerName(builder, "ConvMulFusion_B_" + bias->name().value());
    builder.MakeInitializer(MakeFloatingInitializer(*weights, bias_name, {channels}, bias_values));
    inputs.push_back(bias_name);
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(FoldedConv(conv, mul, inputs, "ConvMulFusionPattern"));
  return replacements;
}

std::set<std::string> ConvBatchNormalizationFusionPattern::FastOpType() const { return {"Conv"}; }

core::builder::MatchResult
ConvBatchNormalizationFusionPattern::Match(core::builder::GraphGraph &graph,
                                           const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("") < 7) {
    return NoMatch(candidate, "the default-domain opset is unsupported for this fusion");
  }
  const NodeProto *bn = SingleFollowingNode(graph, candidate, "BatchNormalization");
  if (bn == nullptr) {
    return NoMatch(
        candidate,
        "the Conv output is not exclusively consumed by a default-domain BatchNormalization");
  }
  if (bn->input_size() != 5 || bn->input()[0].value() != candidate.output()[0].value() ||
      bn->output_size() < 1 || bn->output()[0].value().empty()) {
    return NoMatch(candidate, "the BatchNormalization signature is incomplete");
  }
  const int opset = graph.Builder().OpsetVersion("");
  if ((opset < 14 && bn->output_size() != 1) ||
      (opset >= 14 && GetAttributeOr<int64_t>(*bn, "training_mode", 0) != 0)) {
    return NoMatch(candidate, "the BatchNormalization is in training mode");
  }
  for (int i = 1; i < bn->output_size(); ++i) {
    const std::string &output = bn->output()[i].value();
    if (!output.empty() && graph.IsUsed(output)) {
      return NoMatch(candidate, "the BatchNormalization has a used optional output");
    }
  }

  const TensorProto *weights = nullptr;
  const TensorProto *conv_bias = nullptr;
  std::vector<double> weight_values;
  std::vector<double> bias_values;
  int64_t channels = 0;
  if (!ReadConvConstants(graph, candidate, weights, conv_bias, weight_values, bias_values,
                         channels)) {
    return NoMatch(candidate, "the Conv weights or bias are not foldable floating constants");
  }
  for (int i = 1; i < 5; ++i) {
    const TensorProto *constant = GetFloatingConstant(graph, bn->input()[i].value());
    std::vector<double> values;
    if (constant == nullptr || constant->data_type() != weights->data_type() ||
        !HasTensorShape(*constant, {channels}) || !ReadFloatingTensor(*constant, values)) {
      return NoMatch(candidate,
                     "a BatchNormalization parameter has an incompatible dtype or shape");
    }
  }
  const TensorProto *variance = graph.GetComputedConstant(bn->input()[4].value());
  std::vector<double> variance_values;
  ReadFloatingTensor(*variance, variance_values);
  const double epsilon = static_cast<double>(GetAttributeOr<float>(*bn, "epsilon", 1.0e-5f));
  if (!std::isfinite(epsilon) || epsilon < 0.0) {
    return NoMatch(candidate, "the BatchNormalization epsilon is invalid");
  }
  for (double value : variance_values) {
    if (value + epsilon <= 0.0 || !std::isfinite(value + epsilon)) {
      return NoMatch(candidate, "the BatchNormalization variance is invalid");
    }
  }
  std::vector<double> scale;
  std::vector<double> bn_bias;
  std::vector<double> mean;
  ReadFloatingTensor(*graph.GetComputedConstant(bn->input()[1].value()), scale);
  ReadFloatingTensor(*graph.GetComputedConstant(bn->input()[2].value()), bn_bias);
  ReadFloatingTensor(*graph.GetComputedConstant(bn->input()[3].value()), mean);
  const std::size_t channel_block = weight_values.size() / static_cast<std::size_t>(channels);
  for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
    const double factor = scale[channel] / std::sqrt(variance_values[channel] + epsilon);
    for (std::size_t i = 0; i < channel_block; ++i) {
      weight_values[channel * channel_block + i] *= factor;
    }
    bias_values[channel] = (bias_values[channel] - mean[channel]) * factor + bn_bias[channel];
  }
  if (!ValuesRepresentable(*weights, weight_values) ||
      !ValuesRepresentable(*weights, bias_values)) {
    return NoMatch(candidate, "the folded Conv constants are not finite in the source dtype");
  }
  return core::builder::MatchResult{this, {&candidate, bn}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConvBatchNormalizationFusionPattern::Apply(core::builder::GraphGraph &graph,
                                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      Match(graph, *nodes[0]).nodes != nodes) {
    throw BuilderError(
        "ConvBatchNormalizationFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &conv = *nodes[0];
  const NodeProto &bn = *nodes[1];
  const TensorProto *weights = nullptr;
  const TensorProto *conv_bias = nullptr;
  std::vector<double> weight_values;
  std::vector<double> bias_values;
  int64_t channels = 0;
  ReadConvConstants(graph, conv, weights, conv_bias, weight_values, bias_values, channels);
  std::vector<double> scale;
  std::vector<double> bn_bias;
  std::vector<double> mean;
  std::vector<double> variance;
  ReadFloatingTensor(*graph.GetComputedConstant(bn.input()[1].value()), scale);
  ReadFloatingTensor(*graph.GetComputedConstant(bn.input()[2].value()), bn_bias);
  ReadFloatingTensor(*graph.GetComputedConstant(bn.input()[3].value()), mean);
  ReadFloatingTensor(*graph.GetComputedConstant(bn.input()[4].value()), variance);
  const double epsilon = static_cast<double>(GetAttributeOr<float>(bn, "epsilon", 1.0e-5f));

  const std::size_t channel_block = weight_values.size() / static_cast<std::size_t>(channels);
  for (std::size_t channel = 0; channel < static_cast<std::size_t>(channels); ++channel) {
    const double factor = scale[channel] / std::sqrt(variance[channel] + epsilon);
    for (std::size_t i = 0; i < channel_block; ++i) {
      weight_values[channel * channel_block + i] *= factor;
    }
    bias_values[channel] = (bias_values[channel] - mean[channel]) * factor + bn_bias[channel];
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string weight_name =
      FreeInitializerName(builder, "ConvBatchNormalizationFusion_W_" + weights->name().value());
  const std::string bias_name =
      FreeInitializerName(builder, "ConvBatchNormalizationFusion_B_" + bn.input()[2].value());
  builder.MakeInitializer(
      MakeFloatingInitializer(*weights, weight_name, TensorDims(*weights), weight_values));
  builder.MakeInitializer(MakeFloatingInitializer(*weights, bias_name, {channels}, bias_values));
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(FoldedConv(conv, bn, {conv.input()[0].value(), weight_name, bias_name},
                                    "ConvBatchNormalizationFusionPattern"));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
