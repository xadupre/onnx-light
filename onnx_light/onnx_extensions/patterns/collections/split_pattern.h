// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces a ``Split`` immediately followed by a ``Concat`` that re-joins all
 * of its outputs, in order and on the same axis, with a single ``Identity``.
 *
 * @code
 * Before:
 *          ┌───────┐
 *   x ────→│ Split │
 *          └───────┘
 *                  │
 *             ┌────┴────┐
 *             │         │
 *             ↓         ↓
 *             s0        s1
 *             │         │
 *             ↓         ↓
 *             └────┬────┘
 *                  ↓
 *              ┌────────┐
 *              │ Concat │────→ y
 *              └────────┘
 *
 * After:
 *          ┌──────────┐
 *   x ────→│ Identity │────→ y
 *          └──────────┘
 * @endcode
 *
 * The rewrite only fires when every ``Split`` output feeds the same ``Concat``
 * node, the concatenated inputs are exactly the split outputs in order, and the
 * two axes match (after normalising negative axes against the input rank).
 */
class SplitConcatPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SplitConcatPattern(int priority = 0) : PatternOptimization(priority, "SplitConcat") {}

  /// Returns ``Split`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Split`` whose outputs are re-joined by a single ``Concat``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched ``Split``/``Concat`` pair with an ``Identity``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Merges several sibling ``Gather`` nodes reading consecutive constant indices
 * ``0, 1, ..., n-1`` on the same axis of a common input into a single
 * ``Split`` (followed by ``Squeeze`` nodes when the indices are scalars).
 *
 * The axis dimension of the shared input must be statically equal to the number
 * of gathers, so the ``Split`` reproduces every original slice.
 *
 * @code
 * Before:
 *                   ┌──────────────┐
 *   x, index=0 ────→│ Gather axis1 │────→ y0
 *                   ├──────────────┤
 *                   ├──────────────┤
 *   x, index=1 ────→│ Gather axis1 │────→ y1
 *                   ├──────────────┤
 *                   ├──────────────┤
 *   x, index=2 ────→│ Gather axis1 │────→ y2
 *                   └──────────────┘
 *
 * After:
 *          ┌───────┐
 *   x ────→│ Split │
 *          └───────┘
 *              │
 *              │                   ┌─────────┐
 *              ├── t0, axes=[1] ──→│ Squeeze │────→ y0
 *              │                   ├─────────┤
 *              │                   ├─────────┤
 *              ├── t1, axes=[1] ──→│ Squeeze │────→ y1
 *              │                   ├─────────┤
 *              │                   ├─────────┤
 *              └── t2, axes=[1] ──→│ Squeeze │────→ y2
 *                                  └─────────┘
 *
 * After (single-element vector indices):
 *          ┌───────┐
 *   x ────→│ Split │────→ y0, y1, y2
 *          └───────┘
 * @endcode
 *
 * Single-element vector indices ``[i]`` retain the axis, so the ``Split``
 * writes ``yi`` directly without ``Squeeze`` nodes.
 */
class GathersSplitPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GathersSplitPattern(int priority = 0) : PatternOptimization(priority, "GathersSplit") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds sibling ``Gather`` nodes covering an axis with constant indices.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the sibling ``Gather`` nodes with a ``Split`` (plus ``Squeeze``).
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces compatible sibling scalar ``Gather`` and contiguous ``Slice``
 * ranges covering a complete static axis with one ``Split``. Scalar Gather
 * outputs are restored with ``Squeeze``.
 *
 * @code
 * Before:
 *   x --+--> Gather(index=0) ----------> y0
 *       +--> Slice(starts=1, ends=3) --> y1
 *
 * After:
 *   x --> Split(sizes=[1,2]) --+--> Squeeze --> y0
 *                              +-------------> y1
 * @endcode
 */
class GatherSliceToSplitPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherSliceToSplitPattern(int priority = 0)
      : PatternOptimization(priority, "GatherSliceToSplit") {}

  /// Returns Gather and Slice as possible roots.
  std::set<std::string> FastOpType() const override;

  /// Finds a complete, non-overlapping sibling partition on one axis.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits Split and any Squeeze nodes required by scalar Gather outputs.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Merges several sibling ``Slice`` nodes that partition a common input along a
 * single axis into contiguous, non-overlapping ranges into one ``Split``.
 *
 * Each slice must use a unit step, share the same axis, start where the
 * previous one ends, begin at zero and end at the axis dimension (or the
 * ``INT64_MAX`` sentinel used to denote "until the end").
 *
 * @code
 * Before:
 *                               ┌───────┐
 *   x, [0], [2], [0] ──────────→│ Slice │────→ y0
 *                               ├───────┤
 *                               ├───────┤
 *   x, [2], [5], [0] ──────────→│ Slice │────→ y1
 *                               ├───────┤
 *                               ├───────┤
 *   x, [5], [INT64_MAX], [0] ──→│ Slice │────→ y2
 *                               └───────┘
 *
 * After:
 *   x, splits=[2,3,1] ───→┌───────┐────→ y0
 *                         │ Split │────→ y1
 *                         └───────┘────→ y2
 * @endcode
 */
class SlicesSplitPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SlicesSplitPattern(int priority = 0) : PatternOptimization(priority, "SlicesSplit") {}

  /// Returns ``Slice`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds sibling ``Slice`` nodes partitioning an axis into contiguous ranges.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the sibling ``Slice`` nodes with a single ``Split``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
