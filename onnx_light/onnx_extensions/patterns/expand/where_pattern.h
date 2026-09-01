// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Rewrites ``Where(Not(c), x, y)`` into ``Where(c, y, x)``.
 *
 * @code
 * Before:
 *          ┌─────┐
 *   c ───▶ │ Not │ ───▶ condition
 *          └─────┘
 *
 *                       ┌───────┐
 *   condition, x, y ──▶ │ Where │ ───▶ z
 *                       └───────┘
 *
 * After:
 *              ┌───────┐
 *   c, y, x ─▶ │ Where │ ───▶ z
 *              └───────┘
 * @endcode
 *
 * The two branches are swapped. A shared ``Not`` is preserved for its other
 * consumers while the replacement ``Where`` bypasses it.
 */
class NotWherePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit NotWherePattern(int priority = 0) : PatternOptimization(priority, "NotWhere") {}

  /// Returns ``Where`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Where`` fed by a default-domain ``Not`` condition.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Swaps ``Where`` branches and removes or preserves the matched ``Not``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves ``Equal(x, c)`` after a compatible sibling ``Unsqueeze(x, a)`` and
 * preserves the rank of ``Equal(Unsqueeze(x, a), Unsqueeze(y, a))``.
 *
 * @code
 * Before (upstream form):
 *                    ┌───────────┐
 *   x, axes=a ─────▶ │ Unsqueeze │ ───▶ ux
 *                    └───────────┘
 *
 *             ┌───────┐
 *   x, c ───▶ │ Equal │ ───▶ q
 *             └───────┘
 *                  │
 *                  ▼
 *             ┌───────────┐
 *   axes=a ─▶ │ Unsqueeze │ ───▶ z
 *             └───────────┘
 *
 * After (upstream form):
 *                    ┌───────────┐
 *   x, axes=a ─────▶ │ Unsqueeze │ ───▶ ux
 *                    └───────────┘
 *
 *             ┌───────┐
 *   ux, c ──▶ │ Equal │ ───▶ z
 *             └───────┘
 *
 * Before (local form):
 *                    ┌───────────┐
 *   x, axes=a ─────▶ │ Unsqueeze │ ───▶ ux
 *                    └───────────┘
 *
 *                    ┌───────────┐
 *   y, axes=a ─────▶ │ Unsqueeze │ ───▶ uy
 *                    └───────────┘
 *
 *              ┌───────┐
 *   ux, uy ──▶ │ Equal │ ───▶ z
 *              └───────┘
 *
 * After (local form):
 *             ┌───────┐
 *   x, y ───▶ │ Equal │ ───▶ q
 *             └───────┘
 *
 *                  ┌───────────┐
 *   q, axes=a ───▶ │ Unsqueeze │ ───▶ z
 *                  └───────────┘
 * @endcode
 *
 * All axes are equal constant tensors. The upstream form requires a rank-zero
 * ``c`` unless inferred shapes prove that ``Equal(x,c)`` preserves ``x``'s
 * rank. The local form requires equal pre-``Unsqueeze`` ranks and unshared
 * expanded inputs.
 */
class UnsqueezeEqualPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit UnsqueezeEqualPattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeEqual") {}

  /// Returns ``Equal`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an upstream or local ``Equal``/``Unsqueeze`` topology.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the topology without changing the final output rank.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Rewrites an additive mask or factors a common term from ``Where`` branches.
 *
 * @code
 * Before (mask form):
 *                  ┌───────┐
 *   c, 0, -inf ──▶ │ Where │ ───▶ w
 *                  └───────┘
 *
 *             ┌─────┐
 *   w, k ───▶ │ Add │ ───▶ y
 *             └─────┘
 *
 * After (mask form):
 *                   ┌───────┐
 *   c, k, -inf ───▶ │ Where │ ───▶ y
 *                   └───────┘
 *
 * Before (factoring form):
 *             ┌─────┐
 *   a, z ───▶ │ Add │ ───▶ then
 *             └─────┘
 *
 *             ┌─────┐
 *   b, z ───▶ │ Add │ ───▶ else
 *             └─────┘
 *
 *                     ┌───────┐
 *   c, then, else ──▶ │ Where │ ───▶ y
 *                     └───────┘
 *
 * After (factoring form):
 *                ┌───────┐
 *   c, a, b ───▶ │ Where │ ───▶ w
 *                └───────┘
 *
 *             ┌─────┐
 *   w, z ───▶ │ Add │ ───▶ y
 *             └─────┘
 * @endcode
 *
 * The mask form requires an unshared ``Where`` output and accepts either
 * operand order for ``Add``. The factoring form requires both branch ``Add``
 * outputs to be unshared and recognizes the common input in either position.
 */
class WhereAddPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit WhereAddPattern(int priority = 0) : PatternOptimization(priority, "WhereAdd") {}

  /// Returns ``Where`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an additive mask or two ``Add`` branches sharing one input.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits the upstream ``Where`` or the local factored ``Where`` and ``Add``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
