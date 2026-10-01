// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Folds an initializer's Unsqueeze and Cast into the initializer consumed by Add.
 *
 * @code
 * Before:
 *   initializer ──→ ┌───────────┐ ──→ ┌──────┐ ──→ ┌─────┐ ──→ output
 *                   │ Unsqueeze │     │ Cast │     │ Add │
 *                   └───────────┘     └──────┘     └─────┘
 * After:
 *   folded initializer ──────────────────────────→ ┌─────┐ ──→ output
 *                                                   │ Add │
 *                                                   └─────┘
 * @endcode
 *
 * Only folds single-use intermediates and a non-overridable initializer when a
 * registered Cast kernel can evaluate the requested conversion.
 */
class InitializerUnsqueezeCastPattern final : public core::builder::PatternOptimization {
public:
  explicit InitializerUnsqueezeCastPattern(int priority = 1)
      : PatternOptimization(priority, "InitializerUnsqueezeCast") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
