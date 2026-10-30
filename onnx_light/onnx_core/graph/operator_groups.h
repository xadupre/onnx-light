// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::builder {
class GraphGraph;
}

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/// Describes a function body through source indices and ordered interfaces.
struct OperatorGroup {
  std::vector<std::size_t> node_indices;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  /// Transfer estimate when reported by the proposing algorithm.
  std::optional<uint64_t> transfer_bytes;
};

/// Describes why a grouping algorithm accepted or declined a graph.
enum class OperatorGroupStatus {
  kFound,
  kNoMatch,
  kInsideFunction,
  kIncompatibleOpset,
};

/// Selects a proposed group or an untouched source node in replacement order.
struct OperatorGroupStep {
  bool is_group;
  std::size_t index;
};

/**
 * Describes a non-mutating, scheduled operator-group proposal.
 *
 * Both partitioning and repeated-layer detection return this class. Group node
 * indices are sorted in source order; schedule indices refer either to groups
 * or source nodes. The schedule covers the inclusive [first, last] interval.
 * The proposing algorithm guarantees dependency-safe contraction. Source
 * indices and interfaces are valid only while the source graph is unchanged.
 */
struct OperatorGroupProposal {
  OperatorGroupStatus status = OperatorGroupStatus::kNoMatch;
  std::vector<OperatorGroup> groups;
  std::vector<OperatorGroupStep> schedule;
  std::size_t first = 0;
  std::size_t last = 0;

  /// Checks bounds, disjoint groups and complete, nonduplicated schedule coverage.
  void Validate(std::size_t node_count) const;

  /// Returns borrowed nodes covering the validated replacement interval.
  std::vector<const NodeProto *>
  MatchedNodes(const utils::RepeatedProtoField<NodeProto> &nodes) const;
};

/// Builds the source-order schedule for disjoint intervals, retaining Constants.
OperatorGroupProposal MakeIntervalGroupProposal(std::vector<OperatorGroup> groups,
                                                std::size_t node_count);

/// Validates a freshly rederived proposal before creating any local functions.
/// Copies untouched nodes with shared tensor storage and delegates each body to
/// make_group. The caller must install the result at proposal.first.
utils::RepeatedProtoField<NodeProto>
MakeOperatorGroupReplacements(builder::GraphGraph &graph, const OperatorGroupProposal &proposal,
                              const std::function<NodeProto(std::size_t)> &make_group);

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
