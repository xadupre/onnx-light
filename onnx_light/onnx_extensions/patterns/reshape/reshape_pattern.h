// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class ConcatReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ConcatReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ConcatReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapePattern(int priority = 0) : PatternOptimization(priority, "Reshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ReduceReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReduceReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ReduceReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class Reshape2Of3Pattern final : public core::builder::PatternOptimization {
public:
  explicit Reshape2Of3Pattern(int priority = 0) : PatternOptimization(priority, "Reshape2Of3") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ReshapeReshapeBinaryPattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeReshapeBinaryPattern(int priority = 0)
      : PatternOptimization(priority, "ReshapeReshapeBinary") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ReshapeReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ReshapeReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ReshapeSqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeSqueezePattern(int priority = 0)
      : PatternOptimization(priority, "ReshapeSqueeze") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ShapeBasedEditDistanceReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedEditDistanceReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedEditDistanceReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ShapeBasedReshapeIsSqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedReshapeIsSqueezePattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedReshapeIsSqueeze") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ShapedBasedReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ShapedBasedReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ShapedBasedReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class StaticConcatReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit StaticConcatReshapePattern(int priority = 0)
      : PatternOptimization(priority, "StaticConcatReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class UnsqueezeOrSqueezeReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit UnsqueezeOrSqueezeReshapePattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeOrSqueezeReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class UnsqueezeReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit UnsqueezeReshapePattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
