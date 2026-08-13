// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

class GraphGraph;
class PatternOptimization;

/// Persistent description of one applied local graph rewrite.
struct LocalRewriting {
  /// Stable diagnostic name of the pattern that produced the rewrite.
  std::string pattern;
  /// One output name identifying each removed node.
  std::vector<std::string> removed_nodes;
  /// Replacement nodes owned by this record.
  utils::RepeatedProtoField<NodeProto> added_nodes;
  /// Initializers created while applying the pattern.
  utils::RepeatedProtoField<TensorProto> added_initializers;
  /// Position of the replacement in the graph before this iteration.
  std::size_t insert_at = 0;
  /// Optimization iteration in which this rewrite was applied.
  std::size_t iteration = 0;
};

/// Describes one subgraph recognized by an optimization pattern.
struct MatchResult {
  /// Pattern that produced this match.
  const PatternOptimization *pattern = nullptr;
  /// Nodes involved in the rewrite, in the order expected by Apply.
  std::vector<const NodeProto *> nodes;
  /// Optional node before which the replacement should be inserted.
  const NodeProto *insert_at = nullptr;
};

/// Stateless interface implemented by graph-rewriting patterns.
class PatternOptimization {
public:
  /// Creates a pattern with the given optimization priority.
  explicit PatternOptimization(int priority = 1, std::string name = {})
      : priority(priority), name_(std::move(name)) {}

  virtual ~PatternOptimization() = default;

  /// Returns the stable diagnostic name of this pattern.
  const std::string &Name() const noexcept { return name_; }

  /// Assigns the registry name, rejecting a conflicting intrinsic name.
  void SetRegisteredName(const std::string &name);

  /// Returns operator types from which this pattern can start.
  virtual std::set<std::string> FastOpType() const { return {}; }

  /// Returns the match rooted at ``candidate``. A null pattern means no match.
  virtual MatchResult Match(GraphGraph &graph, const NodeProto &candidate) const = 0;

  /// Builds the replacement nodes for one match.
  virtual utils::RepeatedProtoField<NodeProto>
  Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const = 0;

  /// Priority used by the optimization driver.
  int priority;

private:
  std::string name_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
