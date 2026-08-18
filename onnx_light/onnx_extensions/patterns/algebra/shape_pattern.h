// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class ShapeBasedShapeShapeAddPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedShapeShapeAddPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedShapeShapeAdd") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
