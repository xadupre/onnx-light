// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/operator_groups.h"

#include "onnx_core/builder/graph_graph.h"

#include <algorithm>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

void OperatorGroupProposal::Validate(std::size_t node_count) const {
  const auto fail = []() {
    throw builder::BuilderError("Invalid operator-group proposal or replacement schedule.");
  };
  switch (status) {
  case OperatorGroupStatus::kFound:
    break;
  case OperatorGroupStatus::kNoMatch:
  case OperatorGroupStatus::kInsideFunction:
  case OperatorGroupStatus::kIncompatibleOpset:
    if (!groups.empty() || !schedule.empty())
      fail();
    return;
  default:
    fail();
  }
  if (groups.empty() || first > last || last >= node_count)
    fail();
  const std::size_t unowned = groups.size();
  std::vector<std::size_t> owners(last - first + 1, unowned);
  for (std::size_t group = 0; group < groups.size(); ++group) {
    const auto &indices = groups[group].node_indices;
    if (indices.empty())
      fail();
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const auto position = indices[i];
      if (position < first || position > last || (i && position <= indices[i - 1]))
        fail();
      auto &owner = owners[position - first];
      if (owner != unowned)
        fail();
      owner = group;
    }
  }
  std::vector<bool> emitted_groups(groups.size(), false), emitted_nodes(owners.size(), false);
  for (const auto &[is_group, index] : schedule) {
    if (is_group) {
      if (index >= groups.size() || emitted_groups[index])
        fail();
      emitted_groups[index] = true;
      for (const auto position : groups[index].node_indices)
        emitted_nodes[position - first] = true;
    } else {
      if (index < first || index > last || owners[index - first] != unowned ||
          emitted_nodes[index - first])
        fail();
      emitted_nodes[index - first] = true;
    }
  }
  if (std::find(emitted_nodes.begin(), emitted_nodes.end(), false) != emitted_nodes.end() ||
      std::find(emitted_groups.begin(), emitted_groups.end(), false) != emitted_groups.end())
    fail();
}

std::vector<const NodeProto *>
OperatorGroupProposal::MatchedNodes(const utils::RepeatedProtoField<NodeProto> &nodes) const {
  Validate(nodes.size());
  if (status != OperatorGroupStatus::kFound)
    return {};
  std::vector<const NodeProto *> matched;
  matched.reserve(last - first + 1);
  for (std::size_t i = first; i <= last; ++i)
    matched.push_back(&nodes[i]);
  return matched;
}

OperatorGroupProposal MakeIntervalGroupProposal(std::vector<OperatorGroup> groups,
                                                std::size_t node_count) {
  OperatorGroupProposal proposal;
  if (groups.empty())
    return proposal;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const auto &indices = groups[i].node_indices;
    if (indices.empty() || !std::is_sorted(indices.begin(), indices.end()) ||
        std::adjacent_find(indices.begin(), indices.end()) != indices.end() ||
        (i && indices.front() <= groups[i - 1].node_indices.back()) || indices.back() >= node_count)
      throw builder::BuilderError("Operator-group intervals must be ordered and disjoint.");
  }
  proposal.status = OperatorGroupStatus::kFound;
  proposal.first = groups.front().node_indices.front();
  proposal.last = groups.back().node_indices.back();
  std::size_t position = proposal.first;
  for (std::size_t group = 0; group < groups.size(); ++group) {
    for (const auto index : groups[group].node_indices) {
      while (position < index)
        proposal.schedule.push_back({false, position++});
      ++position;
    }
    proposal.schedule.push_back({true, group});
  }
  proposal.groups = std::move(groups);
  return proposal;
}

utils::RepeatedProtoField<NodeProto>
MakeOperatorGroupReplacements(builder::GraphGraph &graph, const OperatorGroupProposal &proposal,
                              const std::function<NodeProto(std::size_t)> &make_group) {
  proposal.Validate(graph.Builder().Nodes().size());
  if (proposal.status != OperatorGroupStatus::kFound || !make_group)
    throw builder::BuilderError("Operator-group replacement requires a successful proposal.");
  utils::RepeatedProtoField<NodeProto> replacement;
  for (const auto &[is_group, index] : proposal.schedule) {
    if (is_group)
      replacement.push_back(make_group(index));
    else
      replacement.push_back(utils::CopyWithSharedTensorStorage(graph.Builder().Nodes()[index]));
  }
  return replacement;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
