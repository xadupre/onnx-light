// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Merges two adjacent constant-mode Pad nodes with equal constant values.
 *
 * Both padding vectors must be constant, non-negative, and describe the same
 * rank. Attribute-form Pad nodes are supported before opset 11, input-form Pad
 * nodes from opset 11 onward, and constant axes inputs from opset 18 onward.
 * The fused padding is the element-wise sum of the two full-rank vectors.
 *
 * @code
 * Before:
 *       ┌─────┐   ┌─────┐
 *   x → │ Pad │ → │ Pad │ → y
 *       └─────┘   └─────┘
 *
 * After:
 *       ┌─────────────┐
 *   x → │ Pad(p0+p1) │ → y
 *       └─────────────┘
 * @endcode
 */
class PadPadFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit PadPadFusionPattern(int priority = 1) : PatternOptimization(priority, "PadPadFusion") {}

  /// Returns ``Pad`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds two compatible adjacent constant-mode Pad nodes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the pair with one Pad whose padding vectors are summed.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
