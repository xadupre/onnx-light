// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/layer_pattern.h"

#include <utility>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/graph/layer_extraction.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

std::set<std::string> RepeatedLayerPattern::FastOpType() const { return {}; }

core::builder::MatchResult RepeatedLayerPattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (graph.Builder().Nodes().empty() || &candidate != &graph.Builder().Nodes()[0])
    return NoMatch(candidate, "whole-graph detection runs only at the first graph node");
  auto proposal = core::graph::ProposeRepeatedLayers(graph);
  if (proposal.status == core::graph::RepeatedLayerStatus::kInsideLayer)
    return NoMatch(candidate, "existing Layer functions must not contain nested Layer helpers");
  if (proposal.status == core::graph::RepeatedLayerStatus::kIncompatibleOpset)
    return NoMatch(candidate, "RepeatedLayer requires ai.rt opset 1 for its local functions");
  if (proposal.status != core::graph::RepeatedLayerStatus::kFound)
    return NoMatch(candidate, "no bounded, nonoverlapping repeated region with a safe contraction");
  return {this, std::move(proposal.nodes), proposal.insert_at};
}

utils::RepeatedProtoField<NodeProto>
RepeatedLayerPattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr || graph.Builder().Nodes().empty())
    throw core::builder::BuilderError("RepeatedLayer::Apply expects a nonempty match.");
  graph.Position(*nodes[0]);
  const auto verified = Match(graph, graph.Builder().Nodes()[0]);
  if (verified.pattern != this || verified.nodes != nodes)
    throw core::builder::BuilderError("RepeatedLayer::Apply received an inconsistent match.");
  return core::graph::MakeRepeatedLayerReplacements(graph, nodes);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
