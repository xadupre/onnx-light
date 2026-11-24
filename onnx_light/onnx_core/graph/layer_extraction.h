// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/graph/operator_groups.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/// Configures the work budget of one repeated-region search.
struct RepeatedLayerOptions {
  /// Caps seed-family visits and frontier-extension rounds; must be positive.
  std::size_t max_growth_attempts = 4096;
};

/**
 * Finds repeated, connected, structurally corresponding regions around
 * operator types containing "Attention" or standard MatMul, including local functions.
 * Extracts every selected occurrence together, retaining weights as explicit
 * inputs except for shared materialized constants with fewer than 32 elements.
 * Preserves side outputs and lexical subgraph dependencies.
 * Returns OperatorGroupProposal, shared with transfer-aware partitioning.
 * Optionally retains the index for immediate application on the unchanged graph.
 * Indices refer to the unchanged source graph; transfer estimates are not reported.
 *
 * Algorithm, in execution order:
 *
 * - Indexes producers, consumers, declared types, anchor signatures and lexical subgraph captures
 *   once for the detection scan.
 *   The pattern invokes detection only at the first graph node of each scan.
 * - Finds case-sensitive "Attention" substrings in operator types regardless of domain,
 *   or exact standard MatMul anchors, including recursively inspected local functions.
 *   Excludes existing layer-function calls and their bodies.
 *   Accepts operators independently of their domain or determinism when shape
 *   inference is available, including context-local shape callbacks. Otherwise
 *   requires complete output tensor annotations to preserve function inference.
 *   Does not change the conservative custom-operator constant-folding policy.
 *   Keeps nodes referencing enclosing-function attributes, including references
 *   inside graph attributes, outside extracted regions.
 * - Indexes connected port-labelled seeds from compatible anchors and their neighbors;
 *   the first frontier round extends these pairs to three-node motifs when possible.
 *   Only the structure
 *   of the dataflow graph matters, not the position of nodes in the node list.
 *   Different weight values are allowed.
 * - Grows all occurrences together through incremental predecessor/successor frontiers.
 *   Each newly included node contributes its incident edges once. Abstract value slots
 *   check connections and aliasing without copying or comparing whole growing regions.
 *   Splits repeated variants into independent families without searching the graph again.
 *   Keeps singleton differences on the boundary. Constant and weight preparations are
 *   ordinary nodes and join the motif when they correspond across occurrences.
 *   Rejects nonconvex terminal regions during contraction scheduling.
 *   Divergent occurrences remain available
 *   as seeds; marked nodes cannot be reused or crossed.
 *   Requires matching attributes, port roles and topology, but not equal weights.
 *   Rejects incompatible known dtypes, ranks and feature dimensions; leading
 *   activation dimensions may vary. Each accepted region contains at least two nodes.
 * - Computes boundaries only after growth stops: entering values become formal inputs, including
 *   weights and captures; values used outside the region become formal outputs,
 *   including intermediate graph outputs. Rejects unsupported boundary types and
 *   nested outputs captured directly from an outer scope without a local Identity.
 *   Supports tensors, structured values, sequences of supported values and
 *   tensor-valued maps; keeps optional, sparse and opaque boundaries outside Layers.
 * - Contracts regions into scheduling units alongside untouched nodes and
 *   topologically sorts them using data dependencies only; the original node order
 *   is just a deterministic tie-break. Rejects contraction cycles.
 * - Tries rarer anchor families first, then seeds with more occurrences.
 *   Does not enumerate or rank growing cuts. Interface size, transfer bytes and
 *   operator kinds inside a region are not considered.
 * - Reuses the prepared analysis while the graph is unchanged; otherwise revalidates.
 *   Direct replacement API calls always revalidate. Replaces every selected occurrence with
 *   calls to shared model-local functions when their bodies and interfaces
 *   agree up to local names, exporter provenance and recomputed tensor shapes.
 *   Retains shape annotations needed by operators without shape inference.
 *   Copies nodes in topological order
 *   into closed function bodies, preserves caller boundary names/types/tags, and
 *   leaves other weights as explicit inputs.
 *   Embeds materialized constants with fewer than 32 elements as Constant nodes
 *   only when their types, shapes and values agree at every corresponding input.
 *   Reconstructs external Shape results from existing corresponding inputs when
 *   start/end attributes agree, removing those result arguments from every call.
 *   Names each formal input from the longest prefix shared by a strict majority
 *   of corresponding arguments in the matched calls, falling back to input_N.
 *   Gives colliding names a port suffix to avoid lexical SSA shadowing.
 *   Names new functions Layer_<mainop>_<index>, using the first anchor's
 *   operator type in source order and an available nonnegative index.
 *
 * Work limits:
 *
 * - Visits at most 4096 nested nodes per graph node for lexical captures, with
 *   scope depth below 16. An incomplete node stays outside Layers; earlier values
 *   remain live and scheduling cannot move nodes across that node.
 *   Reports capture_analysis_limited even when safe groups elsewhere are found,
 *   or kResourceLimit instead of kNoMatch when no safe proposal remains.
 * - Shares max_growth_attempts seed-family visits and frontier-extension rounds
 *   (4096 by default). All disjoint occurrences grow together without capped batches.
 * - Limits each occurrence to 256 nodes and the rewritten interval to 4096 nodes.
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
 *   x, w0 ───→│ Layer(...)      │───y───→│ Layer(...)      │───→ z
 *             │ body(x, w0) → y │       │ body(y, w1) → z │
 *             └─────────────────┘       └─────────────────┘
 *                                               ↑
 *                                               w1
 * @endcode
 */
OperatorGroupProposal
ProposeRepeatedLayers(builder::GraphGraph &graph,
                      std::shared_ptr<const builder::PatternApplication> *application = nullptr);

/// Uses an explicit per-search budget; throws std::invalid_argument for a zero budget.
OperatorGroupProposal
ProposeRepeatedLayers(builder::GraphGraph &graph, const RepeatedLayerOptions &options,
                      std::shared_ptr<const builder::PatternApplication> *application = nullptr);

/// Revalidates the proposed interval and builds its replacement nodes.
/// Appends or reuses local functions and imports ai.rt opset 1, but does not
/// replace graph nodes; the caller must install the result at proposal.first.
/// Throws builder::BuilderError for an empty, inconsistent or ineligible interval.
utils::RepeatedProtoField<NodeProto>
MakeRepeatedLayerReplacements(builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes);

/// Revalidates with the same options used for the proposal.
utils::RepeatedProtoField<NodeProto>
MakeRepeatedLayerReplacements(builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes,
                              const RepeatedLayerOptions &options);

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
