// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/graph/graph_partition.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Extracts low-transfer operator intervals as model-local ai.rt functions.
 *
 * Calls core::graph::ProposeOperatorGroups only at the first node of each
 * optimizer scan, using the builder's inferred tensor descriptors. The core
 * algorithm owns interval selection and transfer accounting; this opt-in
 * pattern only materializes its proposals. Constants remain explicit inputs.
 * Intervening Constant nodes stay before the calls that use them.
 *
 * Retains boundary names, side outputs, node metadata, source opsets and
 * lexical captures. Unknown boundary sizes prevent grouping. Device-configured
 * nodes and enclosing-function attribute references are not extracted.
 * Local-function bodies and their subgraphs are left unchanged, and generated
 * PartitionGroup calls are never wrapped again.
 *
 * @code
 * Before:
 *             ┌───────────┐       ┌───────────┐
 *   x, w ────→│ operators │───v──→│ operators │───→ y
 *             └───────────┘       └───────────┘
 *
 * After:
 *             ┌───────────────────────────────┐
 *   x, w ────→│ PartitionGroup(x, w) → y     │───→ y
 *             └───────────────────────────────┘
 * @endcode
 */
class PartitionGroupPattern final : public core::builder::PatternOptimization {
public:
  /// Creates an opt-in pattern with the core grouping limits.
  explicit PartitionGroupPattern(int priority = 0, uint64_t max_boundary_bytes = 1024,
                                 uint64_t min_group_nodes = 2, uint64_t max_group_nodes = 128);

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  core::graph::OperatorGroupOptions options_;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
