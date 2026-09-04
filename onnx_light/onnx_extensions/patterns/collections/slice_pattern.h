// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces a full-range Slice by an Identity.
 *
 * The pattern supports the attribute form used before opset 10 and the input
 * form used from opset 10 onward. Starts must all be zero, ends must all be
 * ``INT64_MAX``, and optional steps must all be one. All dynamic parameters
 * and malformed parameter vectors are rejected.
 *
 * @code
 * Before:
 *              ┌───────┐
 *   x, ranges →│ Slice │→ y
 *              └───────┘
 *
 * After:
 *       ┌──────────┐
 *   x → │ Identity │→ y
 *       └──────────┘
 * @endcode
 */
class SliceEliminationPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SliceEliminationPattern(int priority = 1)
      : PatternOptimization(priority, "SliceElimination") {}

  /// Returns ``Slice`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a default-domain Slice that selects its complete input.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the redundant Slice with an Identity preserving its output.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

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
 *            ↓
 *        ┌───────┐
 *        │ Slice │──┐
 *        └───────┘  │
 *                   │ t
 *                   ↓
 *                 ┌───────┐
 *                 │ Slice │←──── s1, e1, a1, st1
 *                 └───────┘
 *                   │
 *                   ↓
 *                   y
 *
 * After:
 *                ┌────────┐
 *    s0, s1 ────→│ Concat │──────────────┐
 *                ├────────┤              │
 *                ├────────┤              │
 *    e0, e1 ────→│ Concat │──────────────┤
 *                ├────────┤              │
 *                ├────────┤              │
 *    a0, a1 ────→│ Concat │──────────────┤
 *                ├────────┤              │
 *                ├────────┤              │
 *    st0,st1 ───→│ Concat │──────────────┤
 *                └────────┘              │
 *                                        ↓
 *                                   ┌───────┐
 *    x ────────────────────────────→│ Slice │────→ y
 *                                   └───────┘
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
