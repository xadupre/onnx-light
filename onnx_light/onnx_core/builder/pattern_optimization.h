// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

class GraphGraph;
class PatternOptimization;

/// Aggregated timing and activity counters for one optimization pattern.
struct PatternOptimizationStatistics {
  /// Stable diagnostic pattern name.
  std::string pattern_name;
  /// Number of candidate nodes passed to the pattern matcher.
  std::size_t attempts = 0;
  /// Number of accepted, disjoint matches.
  std::size_t matches = 0;
  /// Total time spent matching candidates, in nanoseconds.
  int64_t match_time_ns = 0;
  /// Total time spent building replacement nodes, in nanoseconds.
  int64_t apply_time_ns = 0;

  /// Returns a concise summary of these statistics.
  std::string ToString() const;
};

/// Optional timing report populated by :cpp:func:`GraphGraph::Optimize`.
struct OptimizationReport {
  /// Number of optimization iterations executed.
  std::size_t iterations = 0;
  /// Number of local rewrites applied.
  std::size_t rewrites = 0;
  /// Total time spent matching candidate nodes, in nanoseconds.
  int64_t matching_time_ns = 0;
  /// Total time spent applying matches and rebuilding nodes, in nanoseconds.
  int64_t rewriting_time_ns = 0;
  /// Total time spent in graph cleanup passes, in nanoseconds.
  int64_t cleanup_time_ns = 0;
  /// Total time spent folding all-constant replacement nodes, in nanoseconds.
  int64_t constant_folding_time_ns = 0;
  /// Reserved for recursive subgraph optimization added by a later step.
  int64_t subgraph_optimization_time_ns = 0;
  /// Activity and timing counters in pattern evaluation order.
  std::vector<PatternOptimizationStatistics> patterns;

  /// Returns the sum of all phase durations, in nanoseconds.
  int64_t TotalTimeNs() const noexcept;

  /// Returns a concise, printable optimization report.
  std::string ToString() const;
};

/// Persistent description of one applied local graph rewrite.
struct LocalRewriting {
  /// Shared link to the pattern that produced the rewrite.
  std::shared_ptr<const PatternOptimization> pattern;
  /**
   * Positions of the nodes selected by the match.
   *
   * Positions refer to the graph at the start of the rewrite batch identified
   * by ``iteration``, not to the original model: a later batch may match nodes
   * added by an earlier one. This batch-local coordinate system is also used by
   * :cpp:var:`insert_at`.
   */
  std::vector<std::size_t> matched_nodes;
  /// Replacement nodes owned by this record.
  utils::RepeatedProtoField<NodeProto> added_nodes;
  /// Initializers created while applying the pattern.
  utils::RepeatedProtoField<TensorProto> added_initializers;
  /// Positions occupied by added initializers after this rewrite batch.
  std::vector<std::size_t> added_initializer_positions;
  /// Positions of initializers removed by this rewrite.
  std::vector<std::size_t> removed_initializers;
  /// Value-name replacements applied to node inputs and nested captures.
  std::vector<std::pair<std::string, std::string>> value_renames;
  /// Insertion position in the batch-local graph; ``-1`` means the first matched position.
  std::ptrdiff_t insert_at = -1;
  /// Ordered rewrite batch; records with the same value are applied together.
  std::size_t iteration = 0;
  /// Time spent recognizing this match, in nanoseconds.
  int64_t match_time_ns = 0;
  /// Time spent building its replacement nodes, in nanoseconds.
  int64_t apply_time_ns = 0;

  /// Returns a concise summary of this rewrite.
  std::string ToString() const;
};

/// Describes one subgraph recognized by an optimization pattern.
struct MatchResult {
  /// Pattern that produced this match.
  const PatternOptimization *pattern = nullptr;
  /// Nodes involved in the rewrite, in the order expected by Apply.
  std::vector<const NodeProto *> nodes;
  /// Optional node before which the replacement should be inserted.
  const NodeProto *insert_at = nullptr;

  /// Returns a concise summary of this match.
  std::string ToString() const;
};

/// Stateless interface implemented by graph-rewriting patterns.
class PatternOptimization {
public:
  /// Creates a pattern with the given optimization priority.
  explicit PatternOptimization(int priority = 1, std::string name = {})
      : priority(priority), name_(std::move(name)) {}

  virtual ~PatternOptimization() = default;

  /// Returns the stable diagnostic name of this pattern.
  const std::string &Name() const noexcept { return name_; }

  /// Assigns the registry name, rejecting a conflicting intrinsic name.
  void SetRegisteredName(const std::string &name);

  /// Returns a concise summary of this pattern.
  std::string ToString() const;

  /// Returns operator types from which this pattern can start.
  virtual std::set<std::string> FastOpType() const { return {}; }

  /// Returns the match rooted at ``candidate``. A null pattern means no match.
  virtual MatchResult Match(GraphGraph &graph, const NodeProto &candidate) const = 0;

  /// Builds the replacement nodes for one match.
  virtual utils::RepeatedProtoField<NodeProto>
  Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const = 0;

  /// Priority used by the optimization driver.
  int priority;

private:
  std::string name_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
