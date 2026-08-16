// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Fuses two consecutive Not nodes into an Identity.
 *
 * @code
 * Before:             After:
 *
 *   x:bool              x:bool
 *    |                   |
 *   Not                Identity
 *    |                   |
 *   Not                 y:bool
 *    |
 *   y:bool
 * @endcode
 *
 * The first Not node is kept when its output is consumed elsewhere.
 */
class NotNotPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit NotNotPattern(int priority = 1) : PatternOptimization(priority, "NotNot") {}

  /// Returns ``Not`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds two consecutive Not nodes rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Not pair with an Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
