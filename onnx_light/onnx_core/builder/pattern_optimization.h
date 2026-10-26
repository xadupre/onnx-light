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

#include "onnx_core/symbolic/sym_tensor.h"
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
  std::vector<PatternNoMatchStatistics> no_matches = {};

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

/// Explains why one candidate did not match a pattern.
struct PatternNoMatch {
  /// Candidate inspected by the matcher.
  const NodeProto *candidate = nullptr;
  /// Source file containing the rejection condition.
  std::string source_file;
  /// Source line containing the rejection condition.
  std::uint_least32_t source_line = 0;
  /// Human-readable reason why the candidate did not match.
  std::string reason;

  /// Returns a concise summary of this rejection.
  std::string ToString() const;
};

/// Describes one subgraph recognized by an optimization pattern.
struct MatchResult {
  /// Pattern that produced this match.
  const PatternOptimization *pattern = nullptr;
  /**
   * Nodes involved in the rewrite, in the order expected by Apply.
   *
   * A null pointer is an ignored positional placeholder for an optional role.
   * It is not positioned, marked, removed, or persisted in the rewrite.
   */
  std::vector<const NodeProto *> nodes;
  /// Optional node before which the replacement should be inserted.
  const NodeProto *insert_at = nullptr;
  /// Diagnostic attached to an empty result, when the pattern supplied one.
  std::optional<PatternNoMatch> no_match = std::nullopt;

  /// Returns a concise summary of this match.
  std::string ToString() const;
};

/// Stateless interface implemented by graph-rewriting patterns.
class PatternOptimization {
public:
  /// Creates a pattern; kUndefined marks device-independent patterns.
  explicit PatternOptimization(int priority = 1, std::string name = {},
                               symbolic::Device device = symbolic::Device::kUndefined)
      : priority(priority), device(device), name_(std::move(name)) {}

  virtual ~PatternOptimization() = default;

  /// Returns the stable diagnostic name of this pattern.
  const std::string &Name() const noexcept { return name_; }

  /// Assigns the registry name, rejecting a conflicting intrinsic name.
  void SetRegisteredName(const std::string &name);

  /// Returns a concise summary of this pattern.
  std::string ToString() const;

  /// Returns the invariant operator types from which this pattern can start.
  /// The optimization driver caches this value once per optimization.
  virtual std::set<std::string> FastOpType() const { return {}; }

  /**
   * Returns whether Match may run concurrently on worker threads.
   *
   * Opt-in requires an audited, reentrant Match implementation: it must not
   * mutate pattern state, the graph, its builder, or shared caches. Every graph
   * query it uses must support concurrent reads of the frozen matching state;
   * a const-qualified method alone does not establish thread safety.
   */
  virtual bool CanMatchInParallel() const noexcept { return false; }

  /// Returns the match rooted at ``candidate``. A null pattern means no match.
  virtual MatchResult Match(GraphGraph &graph, const NodeProto &candidate) const = 0;

  /// Builds the replacement nodes for one match.
  virtual utils::RepeatedProtoField<NodeProto>
  Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const = 0;

  /// Priority used by the optimization driver.
  int priority;

  /// Exact target device, or kUndefined for a device-independent pattern.
  symbolic::Device device;

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
