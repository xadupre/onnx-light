#pragma once

#include "onnx_core/graph/operator_groups.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

struct OperatorGroupOptions {
  std::optional<uint64_t> max_boundary_bytes;
  uint64_t min_group_nodes = 2;
  uint64_t max_group_nodes = 128;
  std::optional<uint64_t> max_constant_inputs = std::nullopt;
};

/**
 * Proposes disjoint intervals of non-Constant operators using dynamic programming.
 *
 * Minimizes unknown transfers, then known bytes, then ungrouped nodes. Equal scores
 * retain the first discovered partition. Constants and non-overridable initializers
 * remain explicit inputs but cost no bytes. Captures participate in dependencies.
 * Unrepresentable static sizes are unknown; explicit estimates override descriptors.
 * Accepts unknown boundaries by default and reports their transfer_bytes as nullopt.
 * An explicit byte cap requires fully known boundaries within that cap.
 * max_constant_inputs caps distinct constant boundary inputs before internalization.
 * excluded_nodes remain outside groups and split eligible intervals.
 *
 * Throws std::invalid_argument for invalid limits, invalid dataflow, or an optimal
 * transfer cost exceeding uint64.
 * Does not read tensor payloads or modify the graph.
 * Returns the same OperatorGroupProposal as repeated-layer detection, with
 * groups, explicit interfaces and a dependency-safe replacement schedule.
 */
OperatorGroupProposal
ProposeOperatorGroups(const GraphProto &graph, const OperatorGroupOptions &options = {},
                      const std::unordered_map<std::string, uint64_t> &value_sizes = {},
                      const std::set<size_t> &excluded_nodes = {});

/// Proposes extractable groups using the builder's inferred boundary descriptors.
OperatorGroupProposal ProposePartitionGroups(builder::GraphGraph &graph,
                                             const OperatorGroupOptions &options = {});

/// Revalidates the interval and builds local-function calls in proposal order.
utils::RepeatedProtoField<NodeProto>
MakePartitionGroupReplacements(builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes,
                               const OperatorGroupOptions &options = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
