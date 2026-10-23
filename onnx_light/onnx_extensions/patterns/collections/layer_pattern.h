// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Finds repeated, connected, structurally corresponding regions around exact
 * attention or matrix-multiplication identities, including local functions.
 * Extracts every selected occurrence together, retaining weights as explicit
 * inputs and preserving side outputs and lexical subgraph dependencies.
 *
 * Detection runs once per optimizer scan, at the first graph node. It considers
 * at most 32 seed groups, 32 emitted occurrences plus a tail probe, 64 nodes per occurrence,
 * 4096 shared compatibility/growth attempts and 4096 nodes in the rewritten interval.
 * Seed families prefer greater anchor and node coverage, then more occurrences
 * (a smaller period), then region size.
 * Within each grown anchor skeleton, cuts are ranked by crossing-edge
 * multiplicity, unknown-size edges, saturating
 * known byte size, decreasing region size and original topological order.
 * Attributes, input/output port roles and internal topology must correspond;
 * parameter values need not be equal. Incompatible known feature dimensions,
 * element types and ranks are rejected. Leading activation dimensions may vary.
 * Existing Layer nodes and their nested bodies are never wrapped.
 * Stochastic operators, Dropout, and opaque custom operators remain outside
 * regions. Local functions and nested graphs are checked recursively.
 * Nested outputs forwarded directly from an outer scope are conservatively
 * excluded unless forwarded through a locally defined Identity.
 * A matching region has at least two nodes; it may be a repeated partial
 * computation rather than an entire model-defined or transformer layer.
 * Oversized occurrence families are processed in bounded batches, shortened
 * when necessary to leave at least two occurrences for the next batch.
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
 *   x, w0 ───→│ ai.rt::Layer    │───y───→│ ai.rt::Layer    │───→ z
 *             │ body(x, w0) → y │       │ body(y, w1) → z │
 *             └─────────────────┘       └─────────────────┘
 *                                               ↑
 *                                               w1
 * @endcode
 */
class RepeatedLayerPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit RepeatedLayerPattern(int priority = 0)
      : PatternOptimization(priority, "RepeatedLayer") {}

  /// Returns an empty filter because local-function names are unrestricted.
  std::set<std::string> FastOpType() const override;

  /// Finds occurrences from the first graph node and a dependency-safe rewrite interval.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rechecks the match and emits Layer bodies with explicit formal interfaces.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
