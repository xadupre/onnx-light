#pragma once

#include "onnx_proto/onnx.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/// Describes a non-mutating proposal with source node indices and ordered interfaces.
struct OperatorGroup {
  std::vector<size_t> node_indices;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  uint64_t transfer_bytes = 0;
};

struct OperatorGroupOptions {
  uint64_t max_boundary_bytes = 1024;
  uint64_t min_group_nodes = 2;
  uint64_t max_group_nodes = 128;
};

/**
 * Proposes disjoint intervals of non-Constant operators using dynamic programming.
 *
 * Minimizes unknown transfers, then known bytes, then ungrouped nodes. Equal scores
 * retain the first discovered partition. Constants and non-overridable initializers
 * remain explicit inputs but cost no bytes. Captures participate in dependencies.
 * Unrepresentable static sizes are unknown; explicit estimates override descriptors.
 *
 * Throws std::invalid_argument for invalid limits, invalid dataflow, or cost overflow.
 * Does not read tensor payloads or modify the graph.
 */
std::vector<OperatorGroup>
ProposeOperatorGroups(const GraphProto &graph, const OperatorGroupOptions &options = {},
                      const std::unordered_map<std::string, uint64_t> &value_sizes = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
