// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class GemmTransposePattern final : public core::builder::PatternOptimization {
public:
  explicit GemmTransposePattern(int priority = 1)
      : PatternOptimization(priority, "GemmTranspose") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class MatMulAddPattern final : public core::builder::PatternOptimization {
public:
  explicit MatMulAddPattern(int priority = 3, bool allow_reshape = false)
      : PatternOptimization(priority, "MatMulAdd"), allow_reshape_(allow_reshape) {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  bool allow_reshape_;
};

class MatMulReshape2Of3Pattern final : public core::builder::PatternOptimization {
public:
  explicit MatMulReshape2Of3Pattern(int priority = 1)
      : PatternOptimization(priority, "MatMulReshape2Of3") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class MulMulMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit MulMulMatMulPattern(int priority = 1) : PatternOptimization(priority, "MulMulMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ReshapeMatMulReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeMatMulReshapePattern(int priority = 1)
      : PatternOptimization(priority, "ReshapeMatMulReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ShapeBasedMatMulToMulPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedMatMulToMulPattern(int priority = 1)
      : PatternOptimization(priority, "ShapeBasedMatMulToMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class SwitchReshapeActivationPattern final : public core::builder::PatternOptimization {
public:
  explicit SwitchReshapeActivationPattern(int priority = 1)
      : PatternOptimization(priority, "SwitchReshapeActivation") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class TransposeMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeMatMulPattern(int priority = 1)
      : PatternOptimization(priority, "TransposeMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class TransposeReshapeMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeReshapeMatMulPattern(int priority = 1)
      : PatternOptimization(priority, "TransposeReshapeMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
