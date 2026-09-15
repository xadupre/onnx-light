// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"

#include <cmath>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultClip(const NodeProto &node) {
  return node.op_type().value() == "Clip" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() >= 1 &&
         node.output_size() == 1;
}

// Returns the optional input at ``index`` or an empty string when the input is
// absent or explicitly omitted.
std::string OptionalInput(const NodeProto &node, int index) {
  if (index >= node.input_size()) {
    return {};
  }
  return node.input()[index].value();
}

} // namespace

std::set<std::string> ReluClipFusionPattern::FastOpType() const { return {"Clip"}; }

core::builder::MatchResult ReluClipFusionPattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsDefaultClip(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Clip");
  }
  const int opset = graph.Builder().OpsetVersion("");
  if (opset == core::shapes::kUnknownOpsetVersion || opset < 6) {
    return NoMatch(candidate, "the Clip opset is unsupported or unknown");
  }
  const NodeProto *relu = graph.NodeBefore(candidate.input()[0].value());
  if (relu == nullptr || relu->op_type().value() != "Relu" ||
      NormaliseDomain(relu->domain().value()) != kDefaultOnnxDomain || relu->input_size() != 1 ||
      relu->output_size() != 1) {
    return NoMatch(candidate, "the Clip input is not produced by a default-domain Relu");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Relu output has another use");
  }

  double minimum = 0.0;
  if (opset < 11) {
    const AttributeProto *attribute = nullptr;
    for (const AttributeProto &current : candidate.attribute()) {
      if (current.name().value() == "min") {
        attribute = &current;
        break;
      }
    }
    if (attribute == nullptr || !attribute->has_f()) {
      return NoMatch(candidate, "the attribute-form Clip needs a floating-point minimum");
    }
    minimum = attribute->f();
  } else {
    const std::string minimum_name = OptionalInput(candidate, 1);
    if (minimum_name.empty() || !graph.IsConstant(minimum_name)) {
      return NoMatch(candidate, "the input-form Clip needs a constant minimum");
    }
    const TensorProto *tensor = graph.GetComputedConstant(minimum_name);
    if (tensor == nullptr || !ReadScalarAsDouble(*tensor, minimum)) {
      return NoMatch(candidate, "the Clip minimum is not a readable scalar");
    }
  }
  if (std::isnan(minimum) || minimum < 0.0) {
    return NoMatch(candidate, "the Clip effective minimum is not non-negative");
  }
  return core::builder::MatchResult{this, {relu, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ReluClipFusionPattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ReluClipFusionPattern::Apply expects one Relu and one Clip node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ReluClipFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &relu = *nodes[0];
  const NodeProto &clip = *nodes[1];
  std::vector<std::string> inputs;
  inputs.reserve(clip.input_size());
  inputs.push_back(relu.input()[0].value());
  for (int i = 1; i < clip.input_size(); ++i) {
    inputs.push_back(clip.input()[i].value());
  }
  NodeProto replacement = MakeNode("Clip", inputs, {clip.output()[0].value()}, "",
                                   ("ReluClipFusionPattern--" + clip.name().value()).c_str());
  for (const AttributeProto &attribute : clip.attribute()) {
    replacement.mutable_attribute()->push_back(attribute);
  }
  if (clip.has_doc_string()) {
    replacement.set_doc_string(clip.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> ClipClipPattern::FastOpType() const { return {"Clip"}; }

core::builder::MatchResult ClipClipPattern::Match(core::builder::GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  if (!IsDefaultClip(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Clip");
  }
  const NodeProto *before = graph.NodeBefore(candidate.input()[0].value());
  if (before == nullptr) {
    return NoMatch(candidate, "the Clip input is not produced by another node");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the first Clip output has another use");
  }
  if (!IsDefaultClip(*before)) {
    return NoMatch(candidate, "the preceding node is not a default-domain Clip");
  }

  const bool min1 = !OptionalInput(*before, 1).empty();
  const bool min2 = !OptionalInput(candidate, 1).empty();
  if (min1 == min2) {
    return NoMatch(candidate, "exactly one Clip must define the minimum bound");
  }
  const bool max1 = !OptionalInput(*before, 2).empty();
  const bool max2 = !OptionalInput(candidate, 2).empty();
  if (max1 == max2) {
    return NoMatch(candidate, "exactly one Clip must define the maximum bound");
  }
  return core::builder::MatchResult{this, {before, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ClipClipPattern::Apply(core::builder::GraphGraph &graph,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ClipClipPattern::Apply expects two consecutive Clip nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ClipClipPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &before = *nodes[0];
  const NodeProto &node = *nodes[1];

  const std::string min1 = OptionalInput(before, 1);
  const std::string min = min1.empty() ? OptionalInput(node, 1) : min1;
  const std::string max1 = OptionalInput(before, 2);
  const std::string max = max1.empty() ? OptionalInput(node, 2) : max1;

  const std::string name = "ClipClipPattern--" + node.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto replacement = MakeNode("Clip", {before.input()[0].value(), min, max},
                                   {node.output()[0].value()}, "", name.c_str());
  if (node.has_doc_string()) {
    replacement.set_doc_string(node.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
