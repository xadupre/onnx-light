// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class RotaryEmbeddingPattern final : public core::builder::PatternOptimization {
public:
  explicit RotaryEmbeddingPattern(int priority = 1)
      : PatternOptimization(priority, "RotaryEmbedding") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class RotaryConcatPartPattern final : public core::builder::PatternOptimization {
public:
  explicit RotaryConcatPartPattern(int priority = 1)
      : PatternOptimization(priority, "RotaryConcatPart") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class FunctionCausalMaskPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionCausalMaskPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionCausalMask") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class FunctionCausalMaskMulAddPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionCausalMaskMulAddPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionCausalMaskMulAdd") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class FunctionCosSinCachePattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionCosSinCachePattern(int priority = 1)
      : PatternOptimization(priority, "FunctionCosSinCache") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class FunctionHalfRotaryEmbeddingPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionHalfRotaryEmbeddingPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionHalfRotaryEmbedding") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class FunctionAttentionPattern : public core::builder::PatternOptimization {
public:
  explicit FunctionAttentionPattern(int priority = 0)
      : PatternOptimization(priority, "FunctionAttention") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class FunctionAttentionGQAPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionAttentionGQAPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionAttentionGQA") {}
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class AttentionGQAPattern final : public core::builder::PatternOptimization {
public:
  explicit AttentionGQAPattern(int priority = 2) : PatternOptimization(priority, "AttentionGQA") {}
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
