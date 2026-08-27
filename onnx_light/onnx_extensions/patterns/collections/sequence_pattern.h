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
 *
 * @code
 * Before:
 *               +-------------------+
 *   x0, x1 ---> | SequenceConstruct | ---> s
 *               +-------------------+
 *                    |         |
 *                    v         v
 *           +------------+   +------------+
 *           | SequenceAt |   | SequenceAt |
 *           +------------+   +------------+
 *                | index=0        | index=1
 *                v                v
 *                y0               y1
 *
 * After:
 *           +----------+
 *   x0 ---> | Identity | ---> y0
 *           +----------+
 *
 *           +----------+
 *   x1 ---> | Identity | ---> y1
 *           +----------+
 * @endcode
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
 * When ``keepdims=0``, the replacement inserts a ``Squeeze`` on the split axis
 * after every ``Split`` output to reproduce the corresponding ``SequenceAt``
 * result.
 *
 * The indices of the ``SequenceAt`` consumers must be constant scalars that
 * cover ``0 .. n-1`` exactly once.
 *
 * @code
 * Before:
 *          +-----------------+
 *   x ---> | SplitToSequence | ---> s
 *          +-----------------+
 *               |       |
 *               v       v
 *        +------------+ +------------+
 *        | SequenceAt | | SequenceAt |
 *        +------------+ +------------+
 *           | index=0     | index=1
 *           v             v
 *           y0            y1
 *
 * After (keepdims=0):
 *          +-------+
 *   x ---> | Split | ---> t0, t1
 *          +-------+
 *
 *                     +---------+
 *   t0, axes=[1] ---> | Squeeze | ---> y0
 *                     +---------+
 *                     +---------+
 *   t1, axes=[1] ---> | Squeeze | ---> y1
 *                     +---------+
 *
 * After (keepdims=1):
 *          +-------+
 *   x ---> | Split | ---> y0, y1
 *          +-------+
 * @endcode
 *
 * With ``keepdims=1``, the ``Split`` writes the ``SequenceAt`` outputs
 * directly; an explicit ``split`` input is forwarded unchanged.
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
