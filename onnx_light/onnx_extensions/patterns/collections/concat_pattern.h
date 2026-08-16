// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Removes ``Concat`` inputs that are empty along the concatenation axis.
 *
 * An input is dropped when its inferred size along ``axis`` is exactly zero, so
 * it contributes nothing to the output. When a single input remains the
 * ``Concat`` becomes an ``Identity``; otherwise a narrower ``Concat`` is
 * emitted with the same ``axis``.
 */
class ConcatEmptyPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConcatEmptyPattern(int priority = 0) : PatternOptimization(priority, "ConcatEmpty") {}

  /// Returns ``Concat`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Concat`` with at least one empty input along its axis.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the ``Concat`` (or an ``Identity``) without the empty inputs.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Simplifies ``Gather(Concat(...), cst_index)`` when the index is a constant
 * single-element ``int64`` tensor and every ``Concat`` input is a 1-D tensor of
 * a statically known size.
 *
 * The pattern locates which ``Concat`` input holds the addressed element and
 * replaces the ``Gather`` with an ``Identity`` (single-element input), a
 * narrower ``Shape`` (input produced by a ranged ``Shape``), or a ``Gather``
 * with a locally adjusted index.
 */
class ConcatGatherPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConcatGatherPattern(int priority = 0) : PatternOptimization(priority, "ConcatGather") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a constant-index ``Gather`` over a fully-shaped ``Concat``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the ``Gather`` against the located ``Concat`` input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Pushes a shape-preserving unary operator through a ``Concat(x, x)`` so that
 * ``Unary(Concat(x, x))`` becomes ``Concat(Unary(x), Unary(x))``, exposing the
 * shared ``Unary(x)`` for common-subexpression elimination.
 *
 * The rewrite requires opset ≥ 18, a two-input ``Concat`` whose inputs are the
 * same value, and a consumer that is a shape-preserving unary op (or a
 * broadcasting ``Mul`` / ``Add`` / ``Div`` / ``Sub`` / ``Unsqueeze`` with a
 * constant scalar second input).
 */
class ConcatTwiceUnaryPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConcatTwiceUnaryPattern(int priority = 0)
      : PatternOptimization(priority, "ConcatTwiceUnary") {}

  /// Returns ``Concat`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Concat(x, x)`` consumed by a shape-preserving unary op.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Moves the unary op ahead of the ``Concat``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
