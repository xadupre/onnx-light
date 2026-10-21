// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Merges two consecutive ``Transpose`` nodes into a single one.
 *
 * @code
 * Before:
 *          ┌───────────────────┐
 *   x ────→│ Transpose [1,0,2] │────→ t
 *          └───────────────────┘
 *                     │
 *                     ↓
 *              ┌───────────────────┐
 *              │ Transpose [0,2,1] │────→ y
 *              └───────────────────┘
 *
 * After:
 *          ┌───────────────────┐
 *   x ────→│ Transpose [1,2,0] │────→ y
 *          └───────────────────┘
 *
 * After (identity composed permutation):
 *          ┌──────────┐
 *   x ────→│ Identity │────→ y
 *          └──────────┘
 * @endcode
 *
 * When the composed permutation is the identity the pair is replaced by an
 * ``Identity`` node. The first ``Transpose`` is preserved when its output is
 * consumed elsewhere; in that case the rewrite only applies when the composed
 * permutation is the identity, so no additional ``Transpose`` is introduced.
 */
class TransposeTransposePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit TransposeTransposePattern(int priority = 0)
      : PatternOptimization(priority, "TransposeTranspose") {}

  /// Returns ``Transpose`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Transpose`` followed by another default-domain ``Transpose``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched pair with a single ``Transpose`` or an ``Identity``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes an unnecessary ``Transpose`` feeding a ``Gather`` with a scalar index.
 *
 * @code
 * Before:
 *          ┌───────────────────┐
 *   x ────→│ Transpose [2,0,1] │────→ t
 *          └───────────────────┘
 *                     │
 *                     ↓
 *              ┌──────────────┐
 *              │ Gather axis0 │←──── scalar k
 *              └──────────────┘
 *                     │
 *                     ↓
 *                     y
 *
 * After (remaining permutation is sorted):
 *                    ┌──────────────┐
 *   x, scalar k ────→│ Gather axis2 │────→ y
 *                    └──────────────┘
 *
 * After (remaining permutation is not sorted):
 *                    ┌──────────────┐
 *   x, scalar k ────→│ Gather axis2 │────→ g
 *                    └──────────────┘
 *                           │
 *                           ↓
 *                  ┌─────────────────┐
 *                  │ Transpose [1,0] │────→ y
 *                  └─────────────────┘
 * @endcode
 *
 * When the remaining permutation is no longer sorted the rewrite gathers on the
 * original tensor first and transposes the smaller result. For example,
 * ``Transpose(x,[1,2,0])`` followed by scalar ``Gather(axis=1)`` becomes
 * ``Gather(x,axis=2)`` followed by ``Transpose([1,0])``. The original
 * ``Transpose`` is preserved when its output is consumed elsewhere.
 */
class TransposeGatherPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit TransposeGatherPattern(int priority = 0)
      : PatternOptimization(priority, "TransposeGather") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Gather`` with a scalar index fed by a default-domain ``Transpose``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the matched pair, swapping the ``Transpose`` past the ``Gather``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
