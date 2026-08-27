// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Simplifies ``Gather(Concat(..., X, ..., axis=0), cst)`` into
 * ``Gather(X, cst - offset)`` when exactly one ``Concat`` input ``X`` is
 * non-constant, all others are constants of known size, ``X`` is 1-D, and every
 * requested index falls inside ``X``'s slice of the concatenation.
 *
 * ``offset`` is the total size of the constant inputs preceding ``X``.
 *
 * @code
 * Before:
 *                  +--------------+
 *   c0, X, c1 ---> | Concat axis0 | ---> t
 *                  +--------------+
 *                         |
 *                         v
 *                    +--------------+
 *                    | Gather axis0 | <--- indices=[2,4]
 *                    +--------------+
 *                         |
 *                         v
 *                         y
 *
 * After:
 *                         +--------------+
 *   X, indices=[0,2] ---> | Gather axis0 | ---> y
 *                         +--------------+
 * @endcode
 *
 * If ``t`` has another consumer, the original ``Concat`` is retained.
 */
class GatherConcatPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherConcatPattern(int priority = 0) : PatternOptimization(priority, "GatherConcat") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``axis=0`` constant-index ``Gather`` over a ``Concat``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the ``Gather`` against the single non-constant ``Concat`` input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Composes two consecutive ``axis=0`` ``Gather`` nodes with constant indices
 * into a single ``Gather``.
 *
 * The inner indices ``cst1`` must be a 1-D constant array and the outer indices
 * ``cst2`` a constant integer tensor; the fused indices are ``cst1[cst2]``.
 * Scalar outer indices remain scalar, while non-scalar fused indices are
 * materialised as a vector.
 *
 * @code
 * Before:
 *   x, [4,1,7]
 *         |
 *         v
 *   +--------------+
 *   | Gather axis0 | ---> t
 *   +--------------+
 *                           |
 *                           v
 *                      +--------------+
 *                      | Gather axis0 | <--- [2,0]
 *                      +--------------+
 *                           |
 *                           v
 *                           y
 *
 * After:
 *   x, [7,4]
 *       |
 *       v
 *   +--------------+
 *   | Gather axis0 | ---> y
 *   +--------------+
 * @endcode
 *
 * If ``t`` has another consumer, the inner ``Gather`` is retained.
 */
class GatherGatherPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherGatherPattern(int priority = 0) : PatternOptimization(priority, "GatherGather") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds two consecutive constant-index ``axis=0`` ``Gather`` nodes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the pair with a single ``Gather`` on the composed indices.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
