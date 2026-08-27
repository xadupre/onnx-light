// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Fuses matching ReduceMin/ReduceMax and ArgMin/ArgMax nodes into TopK with K=1.
 *
 * @code
 * Before:
 *                    +---------------+
 *            +-----> | ReduceMax/Min | ---> values
 *            |       +---------------+
 *            |
 *   x -------+
 *            |
 *            |       +------------+
 *            +-----> | ArgMax/Min | ---> indices
 *                    +------------+
 *
 * After:
 *                 +------+
 *   x, K=[1] ---> | TopK |
 *                 +------+
 *                    |
 *              +-----+-----+
 *              |           |
 *              v           v
 *           values      indices
 *
 * After (keepdims=0):
 *                 +------+
 *   x, K=[1] ---> | TopK |
 *                 +------+
 *                    |
 *              +-----+-----+
 *              |           |
 *              v           v
 *      temporary values  temporary indices
 *              |           |
 *              v           v
 *          +---------+  +---------+
 *   axes -> | Squeeze |  | Squeeze | <- axes
 *          +---------+  +---------+
 *              |           |
 *              v           v
 *           values      indices
 * @endcode
 *
 * The Min variant sets ``largest=0``. When ``keepdims=0``, TopK produces
 * temporary singleton-axis outputs and two Squeeze nodes restore the original
 * ``values`` and ``indices`` output names.
 */
class ReduceArgTopKPattern final : public core::builder::PatternOptimization {
public:
  explicit ReduceArgTopKPattern(int priority = 0)
      : PatternOptimization(priority, "ReduceArgTopK") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves a Cast-ReduceSum-Mul-Sub-Cast normalization chain to the result type.
 *
 * @code
 * Before:
 *          +------+
 *   x ---> | Cast | ---> xc
 *          +------+
 *             |
 *             v
 *       +-----------+
 *       | ReduceSum | <--- axes
 *       +-----------+
 *             |
 *             v
 *          +-----+
 *          | Mul | <--- scale
 *          +-----+
 *             |
 *             v
 *          +-----+
 *          | Sub | <--- xc (or axes)
 *          +-----+
 *             |
 *             v
 *          +------+
 *          | Cast | ---> y
 *          +------+
 *
 * After:
 *          +-----------+
 *   x ---> | ReduceSum | <--- axes
 *          +-----------+
 *               |
 *               v
 *            +-----+      +------+
 *            | Mul | <--- | Cast | <--- scale
 *            +-----+      +------+
 *               |
 *               v
 *            +-----+
 *            | Sub | <--- x
 *            +-----+
 *               |
 *               v
 *               y
 * @endcode
 *
 * The reversed Sub operand order is retained. The five matched nodes become
 * ReduceSum, Cast, Mul, and Sub nodes in ``T`` while preserving ``axes``,
 * ``scale``, ReduceSum attributes, and the final output ``y``. The matcher also
 * accepts ``axes`` instead of ``xc`` as the reused Sub input; Apply replaces
 * that reused input with ``x`` in either case.
 */
class ReduceSumNormalizePattern final : public core::builder::PatternOptimization {
public:
  explicit ReduceSumNormalizePattern(int priority = 0)
      : PatternOptimization(priority, "ReduceSumNormalize") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
