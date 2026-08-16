// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces ``SequenceConstruct(x0, x1, ...)`` followed by the matching
 * ``SequenceAt(seq, 0)``, ``SequenceAt(seq, 1)``, ... with one ``Identity`` per
 * ``SequenceAt``, forwarding the original tensor directly.
 *
 * The rewrite requires every consumer of the sequence to be a ``SequenceAt``
 * with a constant scalar index, and the indices to cover ``0 .. n-1`` exactly
 * once, so the sequence value becomes dead and is removed.
 */
class SequenceConstructAtPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SequenceConstructAtPattern(int priority = 0)
      : PatternOptimization(priority, "SequenceConstructAt") {}

  /// Returns ``SequenceConstruct`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``SequenceConstruct`` fully consumed by constant ``SequenceAt``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces each ``SequenceAt`` with an ``Identity`` on the built tensor.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces ``SplitToSequence(x, [split], axis=a)`` followed by the matching
 * ``SequenceAt(seq, 0)``, ``SequenceAt(seq, 1)``, ... with a single ``Split``.
 *
 * When ``keepdims=0`` (the default of ``SplitToSequence`` when no ``split``
 * input is provided), each split output keeps the split axis with size one, so
 * a ``Squeeze`` on that axis is inserted after the ``Split`` to restore the
 * scalar-reduced shape produced by ``SequenceAt``.
 *
 * The indices of the ``SequenceAt`` consumers must be constant scalars that
 * cover ``0 .. n-1`` exactly once.
 */
class SplitToSequenceSequenceAtPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SplitToSequenceSequenceAtPattern(int priority = 0)
      : PatternOptimization(priority, "SplitToSequenceSequenceAt") {}

  /// Returns ``SplitToSequence`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``SplitToSequence`` fully consumed by constant ``SequenceAt``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the sequence with a ``Split`` (and ``Squeeze`` when needed).
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
