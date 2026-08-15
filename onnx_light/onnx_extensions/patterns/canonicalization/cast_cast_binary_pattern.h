// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/// Moves matching floating-point Cast nodes after a binary arithmetic operation.
class CastCastBinaryPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastCastBinaryPattern(int priority = 1)
      : PatternOptimization(priority, "CastCastBinary") {}

  /// Returns the supported binary arithmetic operator types.
  std::set<std::string> FastOpType() const override;

  /// Finds two compatible Cast nodes feeding a binary arithmetic operation.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Cast-Cast-binary sequence with binary-Cast.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
