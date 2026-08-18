// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class GeluPattern final : public core::builder::PatternOptimization {
public:
  explicit GeluPattern(int priority = 0, int min_opset = 20, std::string domain = "")
      : PatternOptimization(priority, "Gelu"), min_opset_(min_opset), domain_(std::move(domain)) {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  int min_opset_;
  std::string domain_;
};

class LeakyReluPattern final : public core::builder::PatternOptimization {
public:
  explicit LeakyReluPattern(int priority = 0, int min_opset = 6)
      : PatternOptimization(priority, "LeakyRelu"), min_opset_(min_opset) {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  int min_opset_;
};

class SoftmaxCrossEntropyLossCastPattern final : public core::builder::PatternOptimization {
public:
  explicit SoftmaxCrossEntropyLossCastPattern(int priority = 0, int min_opset = 14,
                                              std::string domain = "")
      : PatternOptimization(priority, "SoftmaxCrossEntropyLossCast"), min_opset_(min_opset),
        domain_(std::move(domain)) {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  int min_opset_;
  std::string domain_;
};

class MaxReluPattern final : public core::builder::PatternOptimization {
public:
  explicit MaxReluPattern(int priority = 1) : PatternOptimization(priority, "MaxRelu") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
