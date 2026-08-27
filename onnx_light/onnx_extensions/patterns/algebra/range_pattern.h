// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Moves a shape-[1] second Add input from after Range into its start and limit inputs.
 *
 * @code
 * Before:
 *   start, limit, delta
 *           |
 *           v
 *       +-------+
 *       | Range | ---> r
 *       +-------+
 *           |
 *           v
 *        +-----+
 *        | Add | <--- offset
 *        +-----+
 *           |
 *           v
 *           y
 *
 * After:
 *               +---------+
 *   offset ---> | Squeeze | ---> s
 *               +---------+
 *
 *              +-----+
 *   start, s ->| Add | ---> start2
 *              +-----+
 *
 *              +-----+
 *   limit, s ->| Add | ---> limit2
 *              +-----+
 *
 *   start2, limit2, delta
 *              |
 *              v
 *          +-------+
 *          | Range | ---> y
 *          +-------+
 *
 * After (start is constant zero):
 *               +---------+
 *   offset ---> | Squeeze | ---> s
 *               +---------+
 *
 *              +-----+
 *   limit, s ->| Add | ---> limit2
 *              +-----+
 *
 *   s, limit2, delta
 *            |
 *            v
 *        +-------+
 *        | Range | ---> y
 *        +-------+
 * @endcode
 *
 * When ``start`` is the constant zero, ``start2`` is ``s`` and its Add is
 * omitted. The Range and Add are replaced while the Add output ``y``, the
 * offset, and any Range step input are retained.
 */
class SwapRangeAddScalarPattern final : public core::builder::PatternOptimization {
public:
  explicit SwapRangeAddScalarPattern(int priority = 0)
      : PatternOptimization(priority, "SwapRangeAddScalar") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
