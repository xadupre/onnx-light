// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class LayerNormalizationPattern final : public core::builder::PatternOptimization {
public:
  explicit LayerNormalizationPattern(int priority = 1)
      : PatternOptimization(priority, "LayerNormalization") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class LayerNormalizationScalePattern final : public core::builder::PatternOptimization {
public:
  explicit LayerNormalizationScalePattern(int priority = 1)
      : PatternOptimization(priority, "LayerNormalizationScale") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class CastLayerNormalizationCastPattern final : public core::builder::PatternOptimization {
public:
  explicit CastLayerNormalizationCastPattern(int priority = 1)
      : PatternOptimization(priority, "CastLayerNormalizationCast") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class BatchNormalizationPattern final : public core::builder::PatternOptimization {
public:
  explicit BatchNormalizationPattern(int priority = 0)
      : PatternOptimization(priority, "BatchNormalization") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class BatchNormalizationTrainingPattern final : public core::builder::PatternOptimization {
public:
  explicit BatchNormalizationTrainingPattern(int priority = 0)
      : PatternOptimization(priority, "BatchNormalizationTraining") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class RMSNormalizationPattern final : public core::builder::PatternOptimization {
public:
  explicit RMSNormalizationPattern(int priority = 1)
      : PatternOptimization(priority, "RMSNormalization") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class RMSNormalizationMulPattern final : public core::builder::PatternOptimization {
public:
  explicit RMSNormalizationMulPattern(int priority = 1)
      : PatternOptimization(priority, "RMSNormalizationMul") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
