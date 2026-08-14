// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

class GraphGraph;
class PatternOptimization;

/// Aggregated reason and source location for rejected pattern candidates.
struct PatternNoMatchStatistics {
  /// Source file containing the rejection condition.
  std::string source_file;
  /// Source line containing the rejection condition.
  std::uint_least32_t source_line = 0;
  /// Human-readable reason why the candidate did not match.
  std::string reason;
  /// Number of match attempts rejected at this condition.
  std::size_t occurrences = 0;

  /// Returns a concise summary of this rejection.
  std::string ToString() const;
};

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
  /// Rejection conditions aggregated by source location and reason.
  std::vector<PatternNoMatchStatistics> no_matches;

  /// Returns a concise summary of these statistics.
  std::string ToString() const;
};

/// Activity and elapsed time for one recursively optimized subgraph.
struct SubgraphOptimizationStatistics {
  /// Nested builder path from the root graph.
  std::vector<std::string> graph_path;
  /// Number of optimization iterations executed in this subgraph and its children.
  std::size_t iterations = 0;
  /// Number of rewrites applied in this subgraph and its children.
  std::size_t rewrites = 0;
  /// Total elapsed optimization time for this subgraph, in nanoseconds.
  int64_t elapsed_time_ns = 0;

  /// Returns a concise summary of these statistics.
  std::string ToString() const;
};

/// Optional timing report populated by :cpp:func:`GraphGraph::Optimize`.
struct OptimizationReport {
  /// Number of optimization iterations executed.
  std::size_t iterations = 0;
  /// Number of local rewrites applied.
  std::size_t rewrites = 0;
  /// Time spent matching candidates in this graph, excluding subgraphs.
  int64_t matching_time_ns = 0;
  /// Time spent applying matches and rebuilding this graph, excluding subgraphs.
  int64_t rewriting_time_ns = 0;
  /// Time spent in cleanup passes for this graph, excluding subgraphs.
  int64_t cleanup_time_ns = 0;
  /// Time spent folding replacements in this graph, excluding subgraphs.
  int64_t constant_folding_time_ns = 0;
  /// Total wall-clock time spent recursively optimizing subgraphs.
  int64_t subgraph_optimization_time_ns = 0;
  /// Aggregated root and subgraph counters in pattern evaluation order.
  std::vector<PatternOptimizationStatistics> patterns;
  /// Flat, deterministic list of recursively optimized subgraphs.
  std::vector<SubgraphOptimizationStatistics> subgraphs;

  /// Returns the sum of all phase durations, in nanoseconds.
  int64_t TotalTimeNs() const noexcept;

  /// Returns a concise, printable optimization report.
  std::string ToString() const;
};

/// Persistent description of one applied local graph rewrite.
struct LocalRewriting {
  /// Shared link to the pattern that produced the rewrite.
  std::shared_ptr<const PatternOptimization> pattern;
  /// Nested builder path from the root graph; empty means the root graph.
  std::vector<std::string> graph_path;
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

/// Explains why one candidate did not match a pattern.
struct PatternNoMatch {
  /// Candidate inspected by the matcher.
  const NodeProto *candidate = nullptr;
  /// Source file containing the rejection condition.
  std::string_view source_file;
  /// Source line containing the rejection condition.
  std::uint_least32_t source_line = 0;
  /// Human-readable reason why the candidate did not match.
  std::string_view reason;

  /// Returns a concise summary of this rejection.
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
  /// Diagnostic attached to an empty result, when the pattern supplied one.
  std::optional<PatternNoMatch> no_match;

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

protected:
  /**
   * Returns an empty match with a rejection reason and its call-site location.
   *
   * This is the normal result for a candidate that does not satisfy a pattern.
   * ``Apply`` is only called for successful matches; invalid arguments passed
   * directly to it remain contract violations.
   */
  template <std::size_t N>
  MatchResult NoMatch(const NodeProto &candidate, const char (&reason)[N],
                      std::source_location location = std::source_location::current()) const {
    return NoMatchImpl(candidate, std::string_view(reason, N - 1), location);
  }

private:
  MatchResult NoMatchImpl(const NodeProto &candidate, std::string_view reason,
                          std::source_location location) const;

  std::string name_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
