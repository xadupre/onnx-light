// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Merges two consecutive ``Slice`` nodes acting on disjoint axes into a single
 * ``Slice`` whose ``starts`` / ``ends`` / ``axes`` (and ``steps`` when present)
 * are the concatenation of the two operand vectors.
 *
 * The intermediate result must be consumed by the second ``Slice`` alone and
 * both nodes must provide a constant ``axes`` input; a missing ``steps`` input
 * on either side is materialised as a vector of ones.
 *
 * @code
 * Before:
 *   x, s0, e0, a0, st0
 *            │
 *            ▼
 *       ┌───────┐
 *       │ Slice │ ───▶ t
 *       └───────┘
 *                                │
 *                                ▼
 *                          ┌───────┐
 *                          │ Slice │ ◀─── s1, e1, a1, st1
 *                          └───────┘
 *                                │
 *                                ▼
 *                                y
 *
 * After:
 *               ┌────────┐
 *   s0, s1 ───▶ │ Concat │ ───▶ s
 *               ├────────┤
 *               ├────────┤
 *   e0, e1 ───▶ │ Concat │ ───▶ e
 *               ├────────┤
 *               ├────────┤
 *   a0, a1 ───▶ │ Concat │ ───▶ a
 *               ├────────┤
 *               ├────────┤
 *   st0,st1 ──▶ │ Concat │ ───▶ st
 *               └────────┘
 *
 *   x, s, e, a, st
 *          │
 *          ▼
 *     ┌───────┐
 *     │ Slice │ ───▶ y
 *     └───────┘
 * @endcode
 */
class SliceSlicePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SliceSlicePattern(int priority = 0) : PatternOptimization(priority, "SliceSlice") {}

  /// Returns ``Slice`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Slice`` fed by another ``Slice`` operating on disjoint axes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Fuses the two slices into a single ``Slice`` with concatenated inputs.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
