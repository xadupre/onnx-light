// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/// Canonicalizes a Cast to Identity when its input already has the target type.
class CastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastPattern(int priority = 0) : PatternOptimization(priority, "Cast") {}

  /// Returns ``Cast`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a redundant type-preserving Cast rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces one matched Cast with an Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
