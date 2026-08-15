// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"

#include <string>

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
