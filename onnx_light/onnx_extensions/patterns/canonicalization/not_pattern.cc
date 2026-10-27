// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/not_pattern.h"

#include <string>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultNot(const NodeProto &node) {
  return node.op_type().value() == "Not" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() == 1 &&
         node.output_size() == 1;
}

} // namespace

std::set<std::string> NotNotPattern::FastOpType() const { return {"Not"}; }

core::builder::MatchResult NotNotPattern::Match(core::builder::GraphGraph &graph,
                                                const NodeProto &candidate) const {
  if (!IsDefaultNot(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Not");
  }
  const NodeProto *before = graph.NodeBefore(candidate.input()[0].value());
  if (before == nullptr || !IsDefaultNot(*before)) {
    return NoMatch(candidate, "the preceding node is not a default-domain Not");
  }
  return core::builder::MatchResult{this, {before, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
NotNotPattern::Apply(core::builder::GraphGraph &graph,
                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("NotNotPattern::Apply expects two consecutive Not nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("NotNotPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &before = *nodes[0];
  const NodeProto &node = *nodes[1];

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(before.output()[0].value())) {
    replacements.add() = before;
  }
  const std::string name = "NotNotPattern--" + node.name().value();
  replacements.push_back(MakeNode("Identity", {before.input()[0].value()},
                                  {node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
