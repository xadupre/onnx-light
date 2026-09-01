// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Combines the scalar constants from three nested Mul or Div nodes.
 *
 * @code
 * Before:
 *              ┌─────────┐
 *   a, c1 ────→│ LeftOp  │──┐
 *              └─────────┘  │
 *                           │
 *              ┌─────────┐  │    ┌─────────┐
 *   b, c2 ────→│ RightOp │──┴───→│ OuterOp │────→ y
 *              └─────────┘       └─────────┘
 *
 * After:
 *             ┌─────────┐
 *   a, b ────→│ OuterOp │────→ t
 *             └─────────┘
 *
 *                          ┌─────────┐
 *   t, combined scalar ───→│ Mul/Div │────→ y
 *                          └─────────┘
 * @endcode
 *
 * ``c1`` and ``c2`` are scalar or one-element constants. The replacement
 * retains the outer output and combines or reciprocates the constants as
 * required by the inner Mul/Div operators; two inner Div nodes produce the
 * final Div, while all other combinations produce the final Mul.
 */
class MulMulMulScalarPattern final : public core::builder::PatternOptimization {
public:
  explicit MulMulMulScalarPattern(int priority = 0)
      : PatternOptimization(priority, "MulMulMulScalar") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Reassociates nested Add or Mul nodes to reduce broadcasting rank.
 *
 * @code
 * Before:
 *             ┌─────────┐
 *   b, c ────→│ Add/Mul │────→ t
 *             └─────────┘
 *                  │
 *                  ↓
 *            ┌─────────┐
 *            │ Add/Mul │←──── a
 *            └─────────┘
 *                  │
 *                  ↓
 *                  y
 *
 * After:
 *             ┌─────────┐
 *   b, a ────→│ Add/Mul │────→ t2
 *             └─────────┘
 *                  │
 *                  ↓
 *            ┌─────────┐
 *            │ Add/Mul │←──── c
 *            └─────────┘
 *                  │
 *                  ↓
 *                  y
 *
 * Alternative matched forms may group (c, a), or place the inner node on the
 * right input of the outer node.
 * @endcode
 *
 * The symmetric right-nested form is also handled, and the alternative
 * ``Op(c, a)`` grouping may be selected. The inner output must be unshared;
 * the outer output name is retained.
 */
class SwitchOrderBinaryPattern final : public core::builder::PatternOptimization {
public:
  explicit SwitchOrderBinaryPattern(int priority = 0)
      : PatternOptimization(priority, "SwitchOrderBinary") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
