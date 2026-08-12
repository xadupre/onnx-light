// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

/**
 * Provides patterns with access to a builder and its current graph index.
 *
 * The match/apply driver is added separately; this class establishes the
 * stable context consumed by PatternOptimization implementations.
 */
class GraphBuilderPatternOptimization {
public:
  /// Creates a pattern context for ``builder``.
  explicit GraphBuilderPatternOptimization(GraphBuilder &builder)
      : builder_(builder), graph_(builder) {}

  /// Returns the builder being optimized.
  GraphBuilder &Builder() noexcept { return builder_; }

  /// Returns the current read-only graph index.
  const GraphGraph &Graph() const noexcept { return graph_; }

private:
  GraphBuilder &builder_;
  GraphGraph graph_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
