// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

/**
 * Applies registered or application-provided rewrite patterns to a builder.
 */
class GraphBuilderPatternOptimization {
public:
  /// Predicate that protects matching nodes from removal.
  using DoNotRemovePredicate = std::function<bool(const NodeProto &)>;

  /// Creates an optimizer using all patterns currently in the core registry.
  explicit GraphBuilderPatternOptimization(GraphBuilder &builder);

  /// Creates an optimizer using the supplied patterns in their given order.
  GraphBuilderPatternOptimization(GraphBuilder &builder,
                                  std::vector<std::unique_ptr<PatternOptimization>> patterns,
                                  DoNotRemovePredicate do_not_remove = {});

  /// Returns the builder being optimized.
  GraphBuilder &Builder() noexcept { return builder_; }

  /// Returns the current read-only graph index.
  const GraphGraph &Graph() const noexcept { return *graph_; }

  /**
   * Applies patterns and cleanup passes until convergence.
   *
   * Patterns are considered in ascending priority order. A negative
   * ``max_iter`` selects ``max(node_count, 10) * priority_count``.
   *
   * Returns:
   *   The number of pattern matches applied.
   */
  std::size_t Optimize(int max_iter = -1);

  /// Returns the patterns owned by this optimizer.
  const std::vector<std::unique_ptr<PatternOptimization>> &Patterns() const noexcept {
    return patterns_;
  }

private:
  void RebuildGraph();

  GraphBuilder &builder_;
  std::unique_ptr<GraphGraph> graph_;
  std::vector<std::unique_ptr<PatternOptimization>> patterns_;
  DoNotRemovePredicate do_not_remove_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
