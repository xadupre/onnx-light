// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/layer_pattern.h"

#include <stdexcept>
#include <utility>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/graph/layer_extraction.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

RepeatedLayerPattern::RepeatedLayerPattern(int priority, std::size_t max_growth_attempts)
    : PatternOptimization(priority, "RepeatedLayer"), options_{max_growth_attempts} {
  if (max_growth_attempts == 0)
    throw std::invalid_argument("RepeatedLayer requires max_growth_attempts > 0.");
}

std::set<std::string> RepeatedLayerPattern::FastOpType() const { return {}; }

core::builder::MatchResult RepeatedLayerPattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (&candidate != graph.FirstPatternCandidate())
    return NoMatch(candidate, "whole-graph detection runs only at the first eligible graph node");
  std::shared_ptr<const core::builder::PatternApplication> application;
  auto proposal = core::graph::ProposeRepeatedLayers(graph, options_, &application);
  if (proposal.status == core::graph::OperatorGroupStatus::kInsideFunction)
    return NoMatch(candidate, "existing Layer functions must not contain nested Layer helpers");
  if (proposal.status == core::graph::OperatorGroupStatus::kIncompatibleOpset)
    return NoMatch(candidate, "RepeatedLayer requires ai.rt opset 1 for its local functions");
  if (proposal.status == core::graph::OperatorGroupStatus::kResourceLimit)
    return NoMatch(candidate,
                   "RepeatedLayer capture analysis resource limit (4096 nested nodes or scope "
                   "depth 16 per graph node); no safe repeated region found");
  if (proposal.status != core::graph::OperatorGroupStatus::kFound)
    return NoMatch(candidate, "no bounded, nonoverlapping repeated region with a safe contraction");
  auto nodes = proposal.MatchedNodes(graph.Builder().Nodes());
  return {this, std::move(nodes), &graph.Builder().Nodes()[proposal.first], std::nullopt,
          std::move(application)};
}

utils::RepeatedProtoField<NodeProto>
RepeatedLayerPattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  // MakeRepeatedLayerReplacements repeats the search once and rejects inconsistent matches.
  return core::graph::MakeRepeatedLayerReplacements(graph, nodes, options_);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
