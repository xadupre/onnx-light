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
 * Before:                     After:
 *
 *   x                           x
 *   |                           |
 *  Split(axis=a)              Identity
 *   |     |                     |
 *  Concat(axis=a)               y
 *   |
 *   y
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
