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
 * Rewrites ``Equal(Unsqueeze(x, a), Unsqueeze(y, a))`` into ``Equal(x, y)``.
 *
 * The rewrite applies when both ``Unsqueeze`` nodes use the same constant axes
 * and are consumed only by the matched ``Equal``.
 */
class UnsqueezeEqualPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit UnsqueezeEqualPattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeEqual") {}

  /// Returns ``Equal`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Equal`` comparing two compatible ``Unsqueeze`` outputs.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds ``Equal`` on the pre-unsqueezed inputs.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Factors a common additive term from ``Where(Add(a, z), Add(b, z))``.
 *
 * The rewrite turns ``Where(c, Add(a, z), Add(b, z))`` into
 * ``Add(Where(c, a, b), z)`` when both ``Add`` nodes are local to the matched
 * ``Where``.
 */
class WhereAddPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit WhereAddPattern(int priority = 0) : PatternOptimization(priority, "WhereAdd") {}

  /// Returns ``Where`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Where`` whose branches are ``Add`` nodes sharing one input.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits ``Where`` then ``Add`` with the factored common additive input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
