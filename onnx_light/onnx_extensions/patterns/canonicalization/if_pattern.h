// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Inlines an If whose branches are structurally identical up to local names.
 *
 * @code
 * Before:
 *           ┌──────────────────────────┐
 *   cond ──→│ If: Neg(x) / Neg(x)        │──→ y
 *           └──────────────────────────┘
 *
 * After:
 *           ┌─────┐
 *   x ─────→│ Neg │────────────────────────→ y
 *           └─────┘
 * @endcode
 *
 * Captured names must agree; local initializers and tensor attributes are
 * compared by value, loading deferred payloads only when their backing sources
 * differ. Graph attributes are compared recursively with lexical bindings.
 * Output order and declared types must agree. Configured nodes, external
 * tensors, sparse/extended initializers, quantization annotations, non-tensor
 * value declarations, value and initializer metadata, and persistent bindings
 * are conservatively excluded, as are If nodes carrying metadata. Branch nodes
 * without shape inference are excluded because their type declarations cannot
 * be retained during inlining. Local values receive fresh names; forwarded
 * outputs use Identity nodes.
 * Materialized GRAPH attributes are accepted by Match/Apply, including directly
 * forwarded captures. GraphBuilder imports retain their stricter requirement
 * that a subgraph output has a local definition.
 */
class IfIdenticalBranchesPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit IfIdenticalBranchesPattern(int priority = 1)
      : PatternOptimization(priority, "IfIdenticalBranches") {}

  /// Returns If as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an If with equivalent branches.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Inlines one branch, preserving the If output names.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
