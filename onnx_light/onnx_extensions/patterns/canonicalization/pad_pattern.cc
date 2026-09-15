// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/pad_pattern.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::FreeInitializerName;
using core::builder::BuilderError;

struct PadInfo {
  std::vector<int64_t> pads;
  std::string constant_value;
  bool value_is_integer = false;
  int64_t integer_value = 0;
  double floating_value = 0.0;
};

bool IsDefaultPad(const NodeProto &node) {
  return node.op_type().value() == "Pad" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() >= 1 &&
         node.output_size() == 1;
}

bool ReadNumericScalar(const TensorProto &tensor, PadInfo &info) {
  std::vector<int64_t> integers;
  if (ReadIntegerValues(tensor, integers) && integers.size() == 1) {
    info.value_is_integer = true;
    info.integer_value = integers[0];
    return true;
  }
  std::vector<double> floating;
  if (ReadFloatingValues(tensor, floating) && floating.size() == 1) {
    info.value_is_integer = false;
    info.floating_value = floating[0];
    return true;
  }
  return false;
}

bool EqualValues(const PadInfo &left, const PadInfo &right) {
  if (left.value_is_integer != right.value_is_integer) {
    return false;
  }
  return left.value_is_integer ? left.integer_value == right.integer_value
                               : left.floating_value == right.floating_value;
}

bool ReadPadInfo(core::builder::GraphGraph &graph, const NodeProto &node, int opset,
                 PadInfo &info) {
  if (!IsDefaultPad(node) || GetAttributeOr<std::string>(node, "mode", "constant") != "constant") {
    return false;
  }

  std::vector<int64_t> compact_pads;
  std::vector<int64_t> axes;
  if (opset < 11) {
    if (!GetAttributeInts(node, "pads", compact_pads)) {
      return false;
    }
    info.value_is_integer = false;
    info.floating_value = GetAttributeOr<float>(node, "value", 0.0F);
  } else {
    if (node.input_size() < 2 || node.input()[1].value().empty() ||
        !graph.IsConstant(node.input()[1].value())) {
      return false;
    }
    const TensorProto *pads = graph.GetComputedConstant(node.input()[1].value());
    if (pads == nullptr || !ReadIntegerValues(*pads, compact_pads)) {
      return false;
    }
    if (node.input_size() > 2 && !node.input()[2].value().empty()) {
      info.constant_value = node.input()[2].value();
      if (!graph.IsConstant(info.constant_value)) {
        return false;
      }
      const TensorProto *value = graph.GetComputedConstant(info.constant_value);
      if (value == nullptr || !ReadNumericScalar(*value, info)) {
        return false;
      }
    } else {
      info.value_is_integer = false;
      info.floating_value = 0.0;
    }
    if (node.input_size() > 3 && !node.input()[3].value().empty()) {
      if (opset < 18 || !graph.IsConstant(node.input()[3].value())) {
        return false;
      }
      const TensorProto *axes_tensor = graph.GetComputedConstant(node.input()[3].value());
      if (axes_tensor == nullptr || !ReadIntegerValues(*axes_tensor, axes)) {
        return false;
      }
    }
  }

  if (compact_pads.empty() || compact_pads.size() % 2 != 0) {
    return false;
  }
  for (int64_t pad : compact_pads) {
    if (pad < 0) {
      return false;
    }
  }

  if (axes.empty()) {
    info.pads = std::move(compact_pads);
    return true;
  }
  if (axes.size() * 2 != compact_pads.size() || !graph.HasShape(node.input()[0].value())) {
    return false;
  }
  const std::size_t rank = graph.GetShape(node.input()[0].value()).Shape().Rank();
  info.pads.assign(rank * 2, 0);
  std::vector<bool> seen(rank, false);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    if (axis < 0) {
      axis += static_cast<int64_t>(rank);
    }
    if (axis < 0 || static_cast<std::size_t>(axis) >= rank ||
        seen[static_cast<std::size_t>(axis)]) {
      return false;
    }
    seen[static_cast<std::size_t>(axis)] = true;
    info.pads[static_cast<std::size_t>(axis)] = compact_pads[i];
    info.pads[rank + static_cast<std::size_t>(axis)] = compact_pads[axes.size() + i];
  }
  return true;
}

} // namespace

std::set<std::string> PadPadFusionPattern::FastOpType() const { return {"Pad"}; }

core::builder::MatchResult PadPadFusionPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  const int opset = graph.Builder().OpsetVersion("");
  if (opset == core::shapes::kUnknownOpsetVersion || opset < 1) {
    return NoMatch(candidate, "the Pad opset is unsupported or unknown");
  }
  if (!IsDefaultPad(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Pad");
  }
  const NodeProto *before = graph.NodeBefore(candidate.input()[0].value());
  if (before == nullptr || !IsDefaultPad(*before)) {
    return NoMatch(candidate, "the Pad input is not produced by another Pad");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the first Pad output has another use");
  }

  PadInfo first;
  PadInfo second;
  if (!ReadPadInfo(graph, *before, opset, first) || !ReadPadInfo(graph, candidate, opset, second)) {
    return NoMatch(candidate, "both Pads need constant-mode non-negative constant parameters");
  }
  if (!EqualValues(first, second)) {
    return NoMatch(candidate, "the Pad constant values differ");
  }
  if (first.pads.size() != second.pads.size()) {
    return NoMatch(candidate, "the Pad vectors describe different ranks");
  }
  for (std::size_t i = 0; i < first.pads.size(); ++i) {
    if (first.pads[i] > std::numeric_limits<int64_t>::max() - second.pads[i]) {
      return NoMatch(candidate, "the summed Pad value overflows int64");
    }
  }
  return core::builder::MatchResult{this, {before, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
PadPadFusionPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("PadPadFusionPattern::Apply expects two Pad nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("PadPadFusionPattern::Apply received an unsafe or inconsistent match.");
  }

  const int opset = graph.Builder().OpsetVersion("");
  PadInfo first;
  PadInfo second;
  if (!ReadPadInfo(graph, *nodes[0], opset, first) ||
      !ReadPadInfo(graph, *nodes[1], opset, second)) {
    throw BuilderError("PadPadFusionPattern::Apply could not read the Pad parameters.");
  }
  std::vector<int64_t> summed(first.pads.size());
  for (std::size_t i = 0; i < summed.size(); ++i) {
    summed[i] = first.pads[i] + second.pads[i];
  }

  const NodeProto &before = *nodes[0];
  const NodeProto &node = *nodes[1];
  const std::string name = "PadPadFusionPattern--" + node.name().value();
  NodeProto replacement;
  if (opset < 11) {
    replacement =
        MakeNode("Pad", {before.input()[0].value()}, {node.output()[0].value()}, "", name.c_str());
    AddAttribute<std::vector<int64_t>>(replacement, "pads", summed);
    AddAttribute<std::string>(replacement, "mode", "constant");
    if (first.floating_value != 0.0) {
      AddAttribute<float>(replacement, "value", static_cast<float>(first.floating_value));
    }
  } else {
    const std::string pads_name = FreeInitializerName(graph.Builder(), name + "_pads");
    TensorProto pads_initializer = MakeInitializerShape(pads_name.c_str(), summed);
    graph.Builder().MakeInitializer(pads_initializer);
    graph.SetComputedConstant(pads_name, std::move(pads_initializer));
    std::vector<std::string> inputs{before.input()[0].value(), pads_name};
    const std::string value =
        !second.constant_value.empty() ? second.constant_value : first.constant_value;
    if (!value.empty()) {
      inputs.push_back(value);
    }
    replacement = MakeNode("Pad", inputs, {node.output()[0].value()}, "", name.c_str());
    AddAttribute<std::string>(replacement, "mode", "constant");
  }
  if (node.has_doc_string()) {
    replacement.set_doc_string(node.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
