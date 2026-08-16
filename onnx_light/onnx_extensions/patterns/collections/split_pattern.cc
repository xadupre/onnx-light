// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/split_pattern.h"

#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::GetAxis;
using collections::IsDefaultOp;
using core::builder::BuilderError;

// Returns the rank of ``name`` into ``rank`` when its shape is known.
bool TryGetRank(core::builder::GraphGraph &graph, const std::string &name, int64_t &rank) {
  if (!graph.HasShape(name)) {
    return false;
  }
  rank = static_cast<int64_t>(graph.GetShape(name).Shape().Rank());
  return true;
}

} // namespace

std::set<std::string> SplitConcatPattern::FastOpType() const { return {"Split"}; }

core::builder::MatchResult SplitConcatPattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Split")) {
    return NoMatch(candidate, "candidate is not a default-domain Split");
  }
  const NodeProto *concat = nullptr;
  for (const auto &output : candidate.output()) {
    const std::vector<const NodeProto *> &consumers = graph.NextNodes(output.value());
    if (consumers.size() != 1) {
      return NoMatch(candidate, "a Split output is not consumed by exactly one node");
    }
    if (concat == nullptr) {
      concat = consumers[0];
    } else if (concat != consumers[0]) {
      return NoMatch(candidate, "the Split outputs feed more than one consumer");
    }
  }
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat")) {
    return NoMatch(candidate, "the single consumer is not a default-domain Concat");
  }

  int64_t axis_split = GetAxis(candidate, 0);
  int64_t axis_concat = GetAxis(*concat, 0);
  if ((axis_split < 0) != (axis_concat < 0)) {
    int64_t rank = 0;
    if (!TryGetRank(graph, candidate.input()[0].value(), rank)) {
      return NoMatch(candidate, "the input rank is required to compare mixed-sign axes");
    }
    if (axis_split < 0) {
      axis_split += rank;
    }
    if (axis_concat < 0) {
      axis_concat += rank;
    }
  }
  if (axis_split != axis_concat) {
    return NoMatch(candidate, "the Split and Concat axes differ");
  }

  if (candidate.output_size() != concat->input_size()) {
    return NoMatch(candidate, "the Concat does not re-join every Split output");
  }
  for (int i = 0; i < candidate.output_size(); ++i) {
    if (candidate.output()[i].value() != concat->input()[i].value()) {
      return NoMatch(candidate, "the Concat inputs are not the Split outputs in order");
    }
  }

  return core::builder::MatchResult{this, {&candidate, concat}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SplitConcatPattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SplitConcatPattern::Apply expects a Split and a Concat node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SplitConcatPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &split = *nodes[0];
  const NodeProto &concat = *nodes[1];

  const std::string name = "SplitConcatPattern--" + split.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Identity", {split.input()[0].value()},
                                  {concat.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
