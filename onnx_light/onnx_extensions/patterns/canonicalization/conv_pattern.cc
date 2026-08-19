// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"

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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
