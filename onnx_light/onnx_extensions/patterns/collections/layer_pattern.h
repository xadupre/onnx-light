// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/graph/layer_extraction.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Extracts repeated operator regions using core::graph::ProposeRepeatedLayers.
 *
 * Runs detection at the first eligible graph node of each optimizer scan and delegates
 * function construction to core::graph::MakeRepeatedLayerReplacements.
 * Preserves the core algorithm's boundary, scheduling and reuse safeguards.
 * The pattern is opt-in and does not wrap existing ai.rt Layer bodies.
 * max_growth_attempts limits anchor comparisons and region extensions per search.
 *
 * @code
 * Before:
 *             ┌─────────────────┐       ┌─────────────────┐
 *   x, w0 ───→│ repeated region │───y───→│ repeated region │───→ z
 *             └─────────────────┘       └─────────────────┘
 *                                               ↑
 *                                               w1
 *
 * After:
 *             ┌─────────────────┐       ┌─────────────────┐
 *   x, w0 ───→│ Layer(x, w0)    │───y───→│ Layer(y, w1)    │───→ z
 *             └─────────────────┘       └─────────────────┘
 * @endcode
 */
class RepeatedLayerPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with an explicit positive search budget.
  explicit RepeatedLayerPattern(
      int priority = 0,
      std::size_t max_growth_attempts = core::graph::RepeatedLayerOptions{}.max_growth_attempts);

  /// Returns an empty filter because local-function names are unrestricted.
  std::set<std::string> FastOpType() const override;

  /// Finds occurrences from the first eligible node using the core analysis.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Delegates match verification and local-function construction to core.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  const core::graph::RepeatedLayerOptions options_;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
