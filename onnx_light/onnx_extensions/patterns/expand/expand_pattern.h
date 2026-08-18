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
 * Simplifies a dynamic ``Concat`` used as an ``Expand`` target shape.
 *
 * When the input and output shapes differ in exactly one dimension, every
 * other ``Concat`` input can be replaced by ``[1]``:
 *
 *      Shape(x)[0]   two
 *           \        /
 *            Concat
 *               |
 *        x --> Expand
 *
 * becomes:
 *
 *          [1]      two
 *           \        /
 *            Concat
 *               |
 *        x --> Expand
 *
 * This keeps the expanded dimension while avoiding unnecessary dynamic shape
 * computations for dimensions preserved by broadcasting.
 */
class ShapeBasedConcatExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedConcatExpandPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedConcatExpand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` whose dynamic Concat target changes one dimension.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the target shape with ``[1]`` for every unchanged dimension.
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

/**
 * Swaps an ``Expand`` and a following ``Unsqueeze``.
 *
 * ``Expand(x, shape)`` followed by ``Unsqueeze(expanded, axes)`` becomes
 * ``Unsqueeze(x, axes)`` followed by ``Expand(..., new_shape)`` where
 * ``new_shape`` inserts a ``1`` at every ``axes`` position of the original
 * ``shape``. The ``Unsqueeze`` then runs on the smaller, pre-expansion tensor.
 */
class SwapExpandUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SwapExpandUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "SwapExpandUnsqueeze") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` consumed by a single ``Unsqueeze`` with constant axes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits the ``Unsqueeze`` first and re-expands its output.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses ``Expand``, ``Unsqueeze`` and ``Expand`` into ``Unsqueeze`` then
 * ``Expand``.
 *
 * ``Expand`` does not change the rank of a tensor, so the ``Unsqueeze`` axes are
 * also valid for the original tensor. The first expansion and the new dimension
 * are absorbed by a single trailing ``Expand`` whose target shape is the
 * element-wise maximum of the first (unsqueezed) target shape and the second
 * target shape.
 */
class ExpandUnsqueezeExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandUnsqueezeExpandPattern(int priority = 0)
      : PatternOptimization(priority, "ExpandUnsqueezeExpand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds ``Expand`` then ``Unsqueeze`` then ``Expand`` with constant shapes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits ``Unsqueeze`` on the original tensor and a single trailing ``Expand``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
