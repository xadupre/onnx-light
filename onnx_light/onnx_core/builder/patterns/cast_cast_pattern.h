// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

/// Collapses consecutive Cast nodes when a single conversion is equivalent.
class CastCastPattern final : public PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastCastPattern(int priority = 1) : PatternOptimization(priority) {}

  /// Returns ``Cast`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a safe consecutive-Cast rewrite rooted at ``candidate``.
  MatchResult Match(GraphBuilderPatternOptimization &opt,
                    const NodeProto &candidate) const override;

  /// Replaces a matched Cast pair with one Cast or Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(GraphBuilderPatternOptimization &opt,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  static symbolic::TensorType OneCastType(symbolic::TensorType input_type,
                                          symbolic::TensorType middle_type,
                                          symbolic::TensorType final_type);
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
