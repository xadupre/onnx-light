// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Finds repeated, connected, structurally corresponding regions around exact
 * attention or matrix-multiplication identities, including local functions.
 * Extracts every selected occurrence together, retaining weights as explicit
 * inputs and preserving side outputs and lexical subgraph dependencies.
 *
 * Algorithm, in execution order:
 *
 * - Indexes producers, consumers, declared types and lexical subgraph captures.
 *   Runs detection only at the first graph node of each optimizer scan.
 * - Finds exact attention/MatMul anchors, including recursively inspected local
 *   functions. Excludes existing layer-function calls and their bodies,
 *   stochastic operators, Dropout and opaque custom calls from candidate regions.
 * - Groups compatible anchors into repeated seed roles. Tries individual anchors,
 *   directed port-labelled one-hop neighborhoods, then bounded periodic strides
 *   to distinguish otherwise identical projections within a larger repeat.
 * - Grows all occurrences together through corresponding predecessors/successors.
 *   Requires matching attributes, port roles and topology, but not equal weights.
 *   Rejects incompatible known dtypes, ranks and feature dimensions; leading
 *   activation dimensions may vary. Each accepted region contains at least two nodes.
 * - Computes each candidate cut: entering values become formal inputs, including
 *   weights and captures; values used outside the region become formal outputs,
 *   including intermediate graph outputs. Rejects unsupported boundary types and
 *   nested outputs captured directly from an outer scope without a local Identity.
 *   Supports tensors, structured values, sequences of supported values and
 *   tensor-valued maps; keeps optional, sparse and opaque boundaries outside Layers.
 * - Contracts regions into scheduling units alongside untouched nodes. Rejects
 *   contraction cycles and preserves the relative order of unwrapped stochastic
 *   or opaque calls. Uses original topological position to break scheduling ties.
 * - Ranks cuts within a grown seed skeleton by crossing-edge multiplicity, then
 *   unknown-size crossings, saturating known byte size, decreasing region size,
 *   and original positions. Ranks seed families by total anchor coverage, total
 *   node coverage, occurrence count (smaller period), then region size.
 * - Revalidates the selected match in Apply and replaces every selected occurrence
 *   with uniquely named model-local functions. Copies nodes in topological order
 *   into closed function bodies, preserves caller boundary names/types/tags, and
 *   leaves weights as explicit inputs.
 *   Gives body inputs fresh names to avoid lexical SSA shadowing.
 *
 * Work limits:
 *
 * - Considers at most 32 seed groups and 4096 shared compatibility/growth attempts.
 * - Emits at most 32 occurrences per batch, with one un-emitted tail probe.
 *   Shortens batches when needed to leave at least two occurrences for the next batch.
 * - Limits each occurrence to 64 nodes and the rewritten interval to 4096 nodes.
 * - Detects repeated computations, not necessarily entire transformer/model layers.
 *
 * @code
 * Before:
 *             ┌─────────────────┐       ┌─────────────────┐
 *   x, w0 ───→│ repeated region │───y───→│ repeated region │───→ z
 *             └─────────────────┘       └─────────────────┘
 *                                               ↑
 *                                               w1
 *
 * After:
 *             ┌─────────────────┐       ┌─────────────────┐
 *   x, w0 ───→│ Layer(...)      │───y───→│ Layer_1(...)    │───→ z
 *             │ body(x, w0) → y │       │ body(y, w1) → z │
 *             └─────────────────┘       └─────────────────┘
 *                                               ↑
 *                                               w1
 * @endcode
 */
class RepeatedLayerPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit RepeatedLayerPattern(int priority = 0)
      : PatternOptimization(priority, "RepeatedLayer") {}

  /// Returns an empty filter because local-function names are unrestricted.
  std::set<std::string> FastOpType() const override;

  /// Finds occurrences from the first graph node and a dependency-safe rewrite interval.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rechecks the match and emits local functions with explicit formal interfaces.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
