// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

class SameChildrenPattern : public core::builder::PatternOptimization {
public:
  explicit SameChildrenPattern(int priority = 0) : PatternOptimization(priority, "SameChildren") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

protected:
  SameChildrenPattern(int priority, std::string name)
      : PatternOptimization(priority, std::move(name)) {}
  static bool SameNode(const NodeProto &first, const NodeProto &second);
  static bool SameNodeWithAliases(
      const NodeProto &first, const NodeProto &second,
      const std::unordered_map<std::string, std::unordered_set<std::string>> &aliases);
  core::builder::MatchResult MatchWithNodes(core::builder::GraphGraph &graph,
                                            const NodeProto &candidate,
                                            const std::vector<const NodeProto *> &next_nodes) const;
};

class SameChildrenFromInputPattern final : public SameChildrenPattern {
public:
  explicit SameChildrenFromInputPattern(int priority = 0)
      : SameChildrenPattern(priority, "SameChildrenFromInput") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
};

class ShapeBasedSameChildrenPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedSameChildrenPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedSameChildren") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class ShapeBasedIdentityPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedIdentityPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedIdentity") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

class SwapUnaryPattern final : public core::builder::PatternOptimization {
public:
  explicit SwapUnaryPattern(int priority = 0) : PatternOptimization(priority, "SwapUnary") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
