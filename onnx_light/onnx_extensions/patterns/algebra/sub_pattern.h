// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Rewrites multiplication by ``1 - x`` into a product followed by subtraction.
 *
 * @code
 * Before:
 *               ┌─────┐
 *   one, x ────▶│ Sub │────▶ d
 *               └─────┘
 *                  │
 *                  ▼
 *               ┌─────┐
 *               │ Mul │◀──── z
 *               └─────┘
 *                  │
 *                  ▼
 *                  y
 *
 * After:
 *             ┌─────┐
 *   x, z ────▶│ Mul │────▶ p
 *             └─────┘
 *                │
 *                ▼
 *             ┌─────┐
 *   z ───────▶│ Sub │────▶ y
 *             └─────┘
 * @endcode
 *
 * The symmetric Mul input order is supported. ``one`` may come from an
 * all-one ConstantOfShape when output-shape preservation is proven; ``z`` and
 * the final output ``y`` are retained.
 */
class Sub1MulPattern final : public core::builder::PatternOptimization {
public:
  explicit Sub1MulPattern(int priority = 0) : PatternOptimization(priority, "Sub1Mul") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
