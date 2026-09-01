// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces classic TreeEnsembleRegressor and TreeEnsembleClassifier nodes with
 * the unified ai.onnx.ml TreeEnsemble operator from opset 5.
 *
 * @code
 * Before (regressor):
 *          ┌───────────────────────┐
 *   x ────→│ TreeEnsembleRegressor │────→ y
 *          └───────────────────────┘
 *
 * After (regressor):
 *          ┌──────────────┐
 *   x ────→│ TreeEnsemble │────→ y
 *          └──────────────┘
 *
 * After (regressor with base values):
 *          ┌──────────────┐
 *   x ────→│ TreeEnsemble │────→ tree scores
 *          └──────────────┘
 *
 *                               ┌─────┐
 *   tree scores, base values ──→│ Add │────→ y
 *                               └─────┘
 *
 * Before (classifier):
 *          ┌────────────────────────┐
 *   x ────→│ TreeEnsembleClassifier │────→ labels, scores
 *          └────────────────────────┘
 *
 * After (classifier with integer labels):
 *          ┌──────────────┐
 *   x ────→│ TreeEnsemble │────→ scores
 *          └──────────────┘
 *                    │
 *                    ↓
 *               ┌────────┐
 *               │ ArgMax │────→ indices
 *               └────────┘
 *                    │
 *                    ↓
 *               ┌────────┐
 *   labels ────→│ Gather │────→ labels output
 *               └────────┘
 *
 * After (classifier with string labels):
 *          ┌──────────────┐
 *   x ────→│ TreeEnsemble │────→ scores
 *          └──────────────┘
 *                    │
 *                    ↓
 *               ┌────────┐
 *               │ ArgMax │────→ indices
 *               └────────┘
 *                    │
 *                    ↓
 *             ┌──────────────┐
 *             │ LabelEncoder │────→ labels output
 *             └──────────────┘
 *
 * @endcode
 *
 * The rewrite applies to FLOAT ensembles already imported with ai.onnx.ml
 * opset 5, using SUM aggregation and no post-transform. Classifier labels are
 * reconstructed with ArgMax followed by Gather for integer labels or by
 * LabelEncoder for string labels. Base values insert Add between TreeEnsemble
 * and its final numeric output when the default-domain opset supports
 * multidirectional broadcasting.
 */
class TreeEnsemblePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit TreeEnsemblePattern(int priority = 1) : PatternOptimization(priority, "TreeEnsemble") {}

  /// Returns the two classic tree ensemble operators as possible roots.
  std::set<std::string> FastOpType() const override;

  /// Finds a classic tree ensemble whose semantics are representable by opset 5.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the classic operator with TreeEnsemble and output adapter nodes.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
