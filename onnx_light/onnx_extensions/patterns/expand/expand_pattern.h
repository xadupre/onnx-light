// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Removes an ``Expand`` that does not change the shape of its input.
 *
 * The rewrite replaces ``Expand(x, shape)`` with ``Identity(x)`` when the shape
 * of ``x`` is fully known and equals the constant ``shape`` fed to ``Expand``.
 */
class ExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandPattern(int priority = 0) : PatternOptimization(priority, "Expand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` whose constant target shape equals its input shape.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the redundant ``Expand`` with ``Identity``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Drops an ``Expand`` feeding an element-wise binary operator.
 *
 * ``Expand(x, shape)`` followed by ``Op(expanded, other)`` becomes
 * ``Op(x, other)`` when ``other`` already has the target ``shape`` and the two
 * operands broadcast together. The rewrite removes one allocation by letting the
 * binary operator broadcast ``x`` directly.
 */
class ExpandBroadcastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandBroadcastPattern(int priority = 0)
      : PatternOptimization(priority, "ExpandBroadcast") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` consumed by a single broadcasting binary operator.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the binary operator on the pre-expanded input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves an ``Expand`` past a following unary-like operator.
 *
 * ``Expand(x, shape)`` followed by a shape-preserving unary operator ``Op``
 * becomes ``Op(x)`` followed by ``Expand(..., shape)``. The unary operator then
 * runs on the smaller, pre-expansion tensor.
 */
class ExpandSwapPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandSwapPattern(int priority = 0) : PatternOptimization(priority, "ExpandSwap") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` consumed by a single unary-like operator.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits the unary operator first and re-expands its output.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
