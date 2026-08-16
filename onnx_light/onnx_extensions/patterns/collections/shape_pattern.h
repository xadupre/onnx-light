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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
