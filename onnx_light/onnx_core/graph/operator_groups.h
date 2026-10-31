// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::builder {
class GraphBuilder;
class GraphGraph;
} // namespace ONNX_LIGHT_NAMESPACE::core::builder

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/// Describes a function body through source indices and ordered interfaces.
struct OperatorGroup {
  std::vector<std::size_t> node_indices;
  /// Source-boundary inputs, before optional constant/Shape internalization.
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

/// Detects exact attention/MatMul anchors, including bounded local-function traversal.
/// Shares a cumulative work budget across calls; zero starts a fresh search.
bool IsOperatorGroupAnchor(const NodeProto &node, const builder::GraphBuilder &builder,
                           std::size_t &work);

/// Returns prefix_<mainop>_<index>, advancing the caller's naming counter.
/// Uses the first anchor in source order, or the first operator without anchors.
/// Avoids function, graph-value and registered-operator collisions in ai.rt.
/// Requires ai.rt opset 1 in source; the caller installs the returned name.
/// Throws builder::BuilderError for an empty prefix/body or exhausted indices.
std::string MakeOperatorGroupFunctionName(builder::GraphBuilder &source,
                                          const utils::RepeatedProtoField<NodeProto> &nodes,
                                          const std::string &prefix, std::size_t &suffix);

/// Describes a Constant or Shape producer replacing a formal input.
struct OperatorGroupInput {
  std::size_t input_index;
  std::optional<std::size_t> source_input;
  NodeProto node;
};

/// Finds common small constants and Shape producers of corresponding boundary inputs.
/// Requires matching Shape attributes and source-input positions across all instances.
/// Never loads external/deferred payloads or specializes overridable graph inputs.
std::vector<OperatorGroupInput>
FindCommonOperatorGroupInputs(const builder::GraphGraph &graph,
                              std::span<const OperatorGroup> instances);

/// Internalizes selected producers in dependency order and returns the remaining arguments.
std::vector<std::string>
InlineOperatorGroupInputs(GraphProto &body, const OperatorGroup &group,
                          const std::vector<OperatorGroupInput> &producers);

/// Reports candidate-side work performed by a function index.
struct OperatorGroupFunctionStats {
  std::size_t candidate_normalizations = 0;
  std::size_t exact_comparisons = 0;
};

/// Caches and indexes normalized definitions for one append-only extraction operation.
/// Binds the source and prefix explicitly; existing definitions and source opsets must
/// remain unchanged while the cache is used. Tensor payloads remain shared.
class OperatorGroupFunctions {
public:
  OperatorGroupFunctions(builder::GraphBuilder &source, std::string prefix);

  /// Returns an equivalent function name, using exact comparisons within hash buckets.
  /// Returns an empty string when no safely comparable definition matches.
  std::string Find(const GraphProto &body);

  /// Creates or reuses a function and advances suffix only for newly created names.
  std::string Add(GraphProto body, std::size_t &suffix);

  /// Returns cumulative candidate normalization and exact comparison counts.
  const OperatorGroupFunctionStats &Stats() const noexcept { return stats_; }

private:
  builder::GraphBuilder &source_;
  std::string prefix_;
  std::size_t indexed_functions_ = 0;
  OperatorGroupFunctionStats stats_;
  std::unordered_map<std::size_t, std::vector<std::pair<std::string, GraphProto>>> definitions_;
};

/// Consumes a closed body to create or reuse a model-local ai.rt function with shared naming.
std::string AddOperatorGroupFunction(builder::GraphBuilder &source, GraphProto body,
                                     const std::string &prefix, std::size_t &suffix);

/// Validates a freshly rederived proposal before creating any local functions.
/// Copies untouched nodes with shared tensor storage and delegates each body to
/// make_group. The caller must install the result at proposal.first.
utils::RepeatedProtoField<NodeProto>
MakeOperatorGroupReplacements(builder::GraphGraph &graph, const OperatorGroupProposal &proposal,
                              const std::function<NodeProto(std::size_t)> &make_group);

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
