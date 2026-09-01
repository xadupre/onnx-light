// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces a Constant node by an initializer and an Identity node.
 *
 * @code
 * Before:
 *                     ┌──────────┐
 *   value=tensor ────▶│ Constant │────▶ cst
 *                     └──────────┘
 *
 * After:
 *   tensor initializer
 *           │
 *           ▼
 *      ┌──────────┐
 *      │ Identity │────▶ cst
 *      └──────────┘
 * @endcode
 *
 * The rewrite applies only when the Constant value can be materialised as a
 * :cpp:class:`TensorProto` and retains the original Constant output name.
 */
class ConstantToInitializerPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConstantToInitializerPattern(int priority = 1)
      : PatternOptimization(priority, "ConstantToInitializer") {}

  /// Returns ``Constant`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a materialisable Constant node rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Constant with an initializer and an Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
