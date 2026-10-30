// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/partition_group_pattern.h"

#include "onnx_core/builder/graph_graph.h"

#include <stdexcept>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

PartitionGroupPattern::PartitionGroupPattern(int priority,
                                             std::optional<uint64_t> max_boundary_bytes,
                                             uint64_t min_group_nodes, uint64_t max_group_nodes,
                                             std::optional<uint64_t> max_constant_inputs)
    : PatternOptimization(priority, "PartitionGroup"),
      options_{max_boundary_bytes, min_group_nodes, max_group_nodes, max_constant_inputs} {
  if (min_group_nodes < 2 || max_group_nodes < min_group_nodes)
    throw std::invalid_argument("PartitionGroup requires 2 <= min_group_nodes <= max_group_nodes.");
}

std::set<std::string> PartitionGroupPattern::FastOpType() const { return {}; }

core::builder::MatchResult PartitionGroupPattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (graph.Builder().Nodes().empty() || &candidate != &graph.Builder().Nodes()[0])
    return NoMatch(candidate, "operator grouping runs only at the first graph node");
  auto proposal = core::graph::ProposePartitionGroups(graph, options_);
  if (proposal.status == core::graph::OperatorGroupStatus::kInsideFunction)
    return NoMatch(candidate, "local-function bodies must not introduce recursive groups");
  if (proposal.status == core::graph::OperatorGroupStatus::kIncompatibleOpset)
    return NoMatch(candidate, "PartitionGroup requires ai.rt opset 1");
  if (proposal.status != core::graph::OperatorGroupStatus::kFound)
    return NoMatch(candidate, "no extractable low-transfer operator interval");
  auto nodes = proposal.MatchedNodes(graph.Builder().Nodes());
  return {this, std::move(nodes), &graph.Builder().Nodes()[proposal.first]};
}

utils::RepeatedProtoField<NodeProto>
PartitionGroupPattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr || graph.Builder().Nodes().empty())
    throw core::builder::BuilderError("PartitionGroup::Apply expects a nonempty match.");
  const auto verified = Match(graph, graph.Builder().Nodes()[0]);
  if (verified.pattern != this || verified.nodes != nodes)
    throw core::builder::BuilderError("PartitionGroup::Apply received an inconsistent match.");
  return core::graph::MakePartitionGroupReplacements(graph, nodes, options_);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
