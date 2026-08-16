// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces an inference Dropout by an Identity node.
 *
 * @code
 * Before:             After:
 *
 *   x                   x
 *    |                   |
 *  Dropout            Identity
 *    |                   |
 *   y                   y
 * @endcode
 *
 * The rewrite applies only when the optional mask output is unused and the
 * ``training_mode`` input, when present, is a constant equal to ``0``.
 */
class DropoutPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit DropoutPattern(int priority = 1) : PatternOptimization(priority, "Dropout") {}

  /// Returns ``Dropout`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an inference Dropout rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Dropout with an Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
