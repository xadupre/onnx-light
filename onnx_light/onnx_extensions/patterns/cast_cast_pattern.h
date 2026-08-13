// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/// Collapses consecutive Cast nodes when a single conversion is equivalent.
class CastCastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastCastPattern(int priority = 1) : PatternOptimization(priority, "CastCast") {}

  /// Returns ``Cast`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a safe consecutive-Cast rewrite rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces a matched Cast pair with one Cast or Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  static core::symbolic::TensorType OneCastType(core::symbolic::TensorType input_type,
                                                core::symbolic::TensorType middle_type,
                                                core::symbolic::TensorType final_type);
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
