// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Merges two consecutive ``Unsqueeze`` nodes into a single one.
 *
 * @code
 * Before:
 *                    ┌───────────┐
 *   x, axes=[1] ────→│ Unsqueeze │────→ t
 *                    └───────────┘
 *                          │
 *                          ↓
 *                   ┌───────────┐
 *                   │ Unsqueeze │←──── axes=[0]
 *                   └───────────┘
 *                          │
 *                          ↓
 *                          y
 *
 * After:
 *                      ┌───────────┐
 *   x, axes=[0,2] ────→│ Unsqueeze │────→ y
 *                      └───────────┘
 * @endcode
 *
 * Both ``axes`` inputs must be scalar or one-dimensional non-negative
 * constants. When one of the two nodes inserts more than one axis the rank of
 * the source tensor must be known so the combined axes can be computed. The
 * first ``Unsqueeze`` is preserved when its output is consumed elsewhere.
 */
class UnsqueezeUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit UnsqueezeUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeUnsqueeze") {}

  /// Returns ``Unsqueeze`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Unsqueeze`` feeding another default-domain ``Unsqueeze``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched pair with a single ``Unsqueeze``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Simplifies a ``Squeeze``/``Unsqueeze`` pair into ``Identity`` or ``Squeeze``.
 *
 * @code
 * Before:
 *                    ┌───────────┐
 *   x, axes=[1] ────→│ Unsqueeze │────→ t
 *                    └───────────┘
 *                          │
 *                          ↓
 *                     ┌─────────┐
 *                     │ Squeeze │←──── axes=[1]
 *                     └─────────┘
 *                          │
 *                          ↓
 *                          y
 *
 * After (matching axes):
 *          ┌──────────┐
 *   x ────→│ Identity │────→ y
 *          └──────────┘
 *
 * After (the following Squeeze removes additional axes):
 *                         ┌─────────┐
 *   x, adjusted axes ────→│ Squeeze │────→ y
 *                         └─────────┘
 *
 * The inverse Squeeze-then-Unsqueeze order is also matched.
 * @endcode
 *
 * The two nodes must be a ``Squeeze`` and an ``Unsqueeze`` (in either order).
 * Equal axes collapse to ``Identity`` (multiple axes must be contiguous). If
 * the first node is an ``Unsqueeze`` whose axes are a strict subset of the
 * following ``Squeeze``, the pair collapses to a single ``Squeeze`` on the
 * adjusted remaining axes; for example, ``Unsqueeze(x,[1])`` then
 * ``Squeeze([1,3])`` becomes ``Squeeze(x,[2])``. The first node is preserved
 * when its output is consumed elsewhere.
 */
class SqueezeUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SqueezeUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "SqueezeUnsqueeze") {}

  /// Returns ``Squeeze`` and ``Unsqueeze`` as the possible root operators.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Squeeze``/``Unsqueeze`` pair feeding an ``Unsqueeze``/``Squeeze``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched pair with an ``Identity`` or a single ``Squeeze``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
