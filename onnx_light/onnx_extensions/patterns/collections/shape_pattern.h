// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Simplifies ``Gather(Shape(X), indices)`` into ``Shape(X, start, end)`` when
 * ``indices`` is a constant ``int64`` scalar or a contiguous ascending 1-D
 * range.
 *
 * The ``start`` / ``end`` attributes possibly carried by the original ``Shape``
 * node are folded into the new bounds. A scalar index yields a
 * ``Shape`` followed by a ``Squeeze`` of the leading axis to recover the 0-D
 * result.
 *
 * @code
 * Before:
 *          +---------------------+
 *   x ---> | Shape start=1 end=5 | ---> s
 *          +---------------------+
 *                     |
 *                     v
 *              +--------------+
 *              | Gather axis0 | <--- indices=[1,2]
 *              +--------------+
 *                     |
 *                     v
 *                     y
 *
 * After:
 *          +---------------------+
 *   x ---> | Shape start=2 end=4 | ---> y
 *          +---------------------+
 *
 * After (scalar index):
 *          +---------------------+
 *   x ---> | Shape start=2 end=3 | ---> s
 *          +---------------------+
 *                    |
 *                    v
 *               +---------+
 *               | Squeeze | <--- axes=[0]
 *               +---------+
 *                    |
 *                    v
 *                    y
 * @endcode
 *
 * For scalar index ``1``, the result is
 * ``Squeeze(Shape(x,start=2,end=3),axes=[0])``. If ``s`` has another consumer,
 * the original ``Shape`` is retained.
 */
class GatherShapePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherShapePattern(int priority = 0) : PatternOptimization(priority, "GatherShape") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a contiguous constant-index ``Gather`` over a ``Shape``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the pair into a bounded ``Shape`` (plus ``Squeeze`` if scalar).
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces ``Shape(Transpose(X, perm))`` by ``Gather(Shape(X), perm)`` so the
 * expensive ``Transpose`` on the full data tensor is avoided.
 *
 * @code
 * Before:
 *          +------------------------+
 *   x ---> | Transpose perm=[2,0,1] | ---> t
 *          +------------------------+
 *                     |
 *                     v
 *              +---------------------+
 *              | Shape start=1 end=3 | ---> y
 *              +---------------------+
 *
 * After:
 *          +-------+
 *   x ---> | Shape | ---> sx
 *          +-------+
 *              |
 *              v
 *        +--------------+
 *        | Gather axis0 | <--- indices=[0,1]
 *        +--------------+
 *              |
 *              v
 *              y
 * @endcode
 *
 * The permutation is a compile-time attribute of the ``Transpose`` node, so the
 * requested dimensions are read directly from ``Shape(X)``. The optional
 * ``start`` / ``end`` attributes of the ``Shape`` node select the sub-range
 * ``perm[start:end]`` used as the ``Gather`` indices. The ``Transpose`` is
 * preserved when its output is consumed elsewhere.
 */
class ShapeTransposePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeTransposePattern(int priority = 0)
      : PatternOptimization(priority, "ShapeTranspose") {}

  /// Returns ``Shape`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Shape`` consuming the output of a ``Transpose`` with a perm.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the pair into ``Gather(Shape(X), perm)``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces ``Shape(Unsqueeze(X, axes))`` by a ``Concat`` of ``Shape(X)`` slices
 * interleaved with constant ``[1]`` tensors at the inserted axis positions.
 *
 * @code
 * Before:
 *   x, axes=[1]
 *        |
 *        v
 *   +-----------+
 *   | Unsqueeze | ---> u
 *   +-----------+
 *        |
 *        v
 *   +-------+
 *   | Shape | ---> y
 *   +-------+
 *
 * After:
 *          +---------------------+
 *   x ---> | Shape start=0 end=1 | ---> s0
 *          +---------------------+
 *
 *          +---------------------+
 *   x ---> | Shape start=1 end=3 | ---> s1
 *          +---------------------+
 *
 *   s0, [1], s1
 *        |
 *        v
 *   +--------+
 *   | Concat | ---> y
 *   +--------+
 * @endcode
 *
 * ``Shape(Unsqueeze(X, axes))`` equals ``Shape(X)`` with ``1`` entries inserted
 * at the ``axes`` positions, so the (potentially large) ``Unsqueeze`` on the
 * data tensor is avoided entirely while the shape vector stays identical. The
 * ``axes`` input must be a one-dimensional constant and the rank of ``X`` must
 * be known. The optional ``start`` / ``end`` attributes of the ``Shape`` node
 * are honoured. The ``Unsqueeze`` is preserved when its output is consumed
 * elsewhere.
 */
class UnsqueezeShapePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit UnsqueezeShapePattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeShape") {}

  /// Returns ``Unsqueeze`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Unsqueeze`` whose output feeds a ``Shape`` node.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the pair into a ``Concat`` of ranged ``Shape`` slices and ``[1]``s.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
