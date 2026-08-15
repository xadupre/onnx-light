// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/// Moves an operation from a temporary floating-point type to its result type.
class CastOpCastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastOpCastPattern(int priority = 1) : PatternOptimization(priority, "CastOpCast") {}

  /// Returns the supported unary and binary operator types.
  std::set<std::string> FastOpType() const override;

  /// Finds an operation between compatible input and output Cast nodes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the operation in the output type and removes the surrounding Cast nodes.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
