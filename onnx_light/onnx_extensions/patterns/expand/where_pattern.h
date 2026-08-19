// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Rewrites ``Where(Not(c), x, y)`` into ``Where(c, y, x)``.
 *
 * The rewrite removes the local ``Not`` when its output is only consumed by the
 * matched ``Where``.
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
 * The local rewrite emits ``Unsqueeze(Equal(x, y), a)`` rather than dropping
 * the inserted dimensions. The upstream rewrite requires a rank-zero constant,
 * unless inferred shapes prove that ``Equal(x, c)`` preserves the rank of
 * ``x``.
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
 * It turns ``Add(Where(c, 0, -inf), x)`` into ``Where(c, x, -inf)`` when ``x``
 * is a provably finite scalar constant, and keeps the local
 * ``Where(c, Add(a, z), Add(b, z))`` to
 * ``Add(Where(c, a, b), z)`` rewrite.
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
