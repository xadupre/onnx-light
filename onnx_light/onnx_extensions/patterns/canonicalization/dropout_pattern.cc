// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/dropout_pattern.h"

#include <string>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultDropout(const NodeProto &node) {
  return node.op_type().value() == "Dropout" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() >= 1 &&
         node.output_size() >= 1;
}

} // namespace

std::set<std::string> DropoutPattern::FastOpType() const { return {"Dropout"}; }

core::builder::MatchResult DropoutPattern::Match(core::builder::GraphGraph &graph,
                                                 const NodeProto &candidate) const {
  if (!IsDefaultDropout(candidate)) {
    return NoMatch(candidate, "candidate is not a default-domain Dropout");
  }
  for (int i = 1; i < candidate.output_size(); ++i) {
    const std::string &name = candidate.output()[i].value();
    if (!name.empty() && graph.IsUsed(name)) {
      return NoMatch(candidate, "a Dropout mask output is used");
    }
  }
  if (candidate.input_size() >= 3) {
    const std::string &training_mode = candidate.input()[2].value();
    if (!training_mode.empty() && graph.IsConstantScalar(training_mode) &&
        !graph.IsConstantScalar(training_mode, 0.0, false)) {
      return NoMatch(candidate, "the Dropout training mode is enabled");
    }
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
DropoutPattern::Apply(core::builder::GraphGraph &graph,
                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("DropoutPattern::Apply expects one Dropout node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("DropoutPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];

  const std::string name = "DropoutPattern--" + node.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto replacement =
      MakeNode("Identity", {node.input()[0].value()}, {node.output()[0].value()}, "", name.c_str());
  if (node.has_doc_string()) {
    replacement.set_doc_string(node.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
