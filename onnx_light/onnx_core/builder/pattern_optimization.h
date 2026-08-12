// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>
#include <vector>

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

class GraphBuilderPatternOptimization;
class PatternOptimization;

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
  explicit PatternOptimization(int priority = 1) : priority(priority) {}

  virtual ~PatternOptimization() = default;

  /// Returns operator types from which this pattern can start.
  virtual std::set<std::string> FastOpType() const { return {}; }

  /// Returns the match rooted at ``candidate``. A null pattern means no match.
  virtual MatchResult Match(GraphBuilderPatternOptimization &opt,
                            const NodeProto &candidate) const = 0;

  /// Builds the replacement nodes for one match.
  virtual utils::RepeatedProtoField<NodeProto>
  Apply(GraphBuilderPatternOptimization &opt,
        const std::vector<const NodeProto *> &nodes) const = 0;

  /// Priority used by the optimization driver.
  int priority;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
