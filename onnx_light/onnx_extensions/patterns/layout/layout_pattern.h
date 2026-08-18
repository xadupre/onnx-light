// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/// Moves matching Squeeze nodes after an Add node.
class SqueezeAddPattern final : public core::builder::PatternOptimization {
public:
  explicit SqueezeAddPattern(int priority = 0) : PatternOptimization(priority, "SqueezeAdd") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/// Moves equal Unsqueeze nodes after a Mul node.
class MulUnsqueezeUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit MulUnsqueezeUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "MulUnsqueezeUnsqueeze") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/// Moves an Unsqueeze before a scalar binary operation.
class SqueezeBinaryUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit SqueezeBinaryUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "SqueezeBinaryUnsqueeze") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/// Swaps an Unsqueeze followed by a Transpose.
class SwapUnsqueezeTransposePattern final : public core::builder::PatternOptimization {
public:
  explicit SwapUnsqueezeTransposePattern(int priority = 0)
      : PatternOptimization(priority, "SwapUnsqueezeTranspose") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/// Replaces a Transpose that only moves size-one axes with a Reshape.
class TransposeEqualReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeEqualReshapePattern(int priority = 0)
      : PatternOptimization(priority, "TransposeEqualReshape") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/// Moves a constant Reshape across one of its adjacent Transpose nodes.
class TransposeReshapeTransposePattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeReshapeTransposePattern(int priority = 0)
      : PatternOptimization(priority, "TransposeReshapeTranspose") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
