// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Removes a Relu immediately followed by a Clip whose effective minimum is
 * non-negative.
 *
 * Clip attributes are supported for opsets 6 through 10 and constant minimum
 * inputs are supported from opset 11 onward. A missing, dynamic, negative, or
 * NaN minimum is rejected, as is a shared Relu output.
 *
 * @code
 * Before:
 *       ┌──────┐   ┌──────┐
 *   x → │ Relu │ → │ Clip │ → y
 *       └──────┘   └──────┘
 *
 * After:
 *       ┌──────┐
 *   x → │ Clip │ → y
 *       └──────┘
 * @endcode
 */
class ReluClipFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ReluClipFusionPattern(int priority = 1)
      : PatternOptimization(priority, "ReluClipFusion") {}

  /// Returns ``Clip`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a Clip with a redundant preceding Relu.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the pair by the Clip applied directly to the Relu input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Merges two consecutive Clip nodes when one defines the minimum and the other
 * the maximum.
 *
 * @code
 * Before:
 *               ┌──────┐
 *   x, min ────→│ Clip │────→ t
 *               └──────┘
 *                  │
 *                  ↓
 *               ┌──────┐
 *               │ Clip │←──── max
 *               └──────┘
 *                  │
 *                  ↓
 *                  y
 *
 * After:
 *                    ┌──────┐
 *   x, min, max ────→│ Clip │────→ y
 *                    └──────┘
 * @endcode
 *
 * The rewrite requires exactly one of the two Clip nodes to supply the minimum
 * bound and exactly one to supply the maximum bound, so the merged Clip carries
 * both bounds and retains the second Clip output. The first Clip output must
 * not have another consumer.
 */
class ClipClipPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ClipClipPattern(int priority = 1) : PatternOptimization(priority, "ClipClip") {}

  /// Returns ``Clip`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds two consecutive Clip nodes with complementary bounds rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Clip pair with a single Clip carrying both bounds.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
