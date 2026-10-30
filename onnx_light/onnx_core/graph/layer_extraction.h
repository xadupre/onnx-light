// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/graph/operator_groups.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/**
 * Finds repeated, connected, structurally corresponding regions around exact
 * attention or matrix-multiplication identities, including local functions.
 * Extracts every selected occurrence together, retaining weights as explicit
 * inputs except for shared materialized constants with fewer than 32 elements.
 * Preserves side outputs and lexical subgraph dependencies.
 * Returns OperatorGroupProposal, shared with transfer-aware partitioning.
 * Indices refer to the unchanged source graph; transfer estimates are not reported.
 *
 * Algorithm, in execution order:
 *
 * - Indexes producers, consumers, declared types and lexical subgraph captures.
 *   The pattern invokes detection only at the first graph node of each scan.
 * - Finds exact attention/MatMul anchors, including recursively inspected local
 *   functions. Excludes existing layer-function calls and their bodies.
 *   Starts with rarer anchor families so hybrid blocks can form a complete motif
 *   before their common projections are extracted separately.
 *   Accepts operators independently of their domain or determinism when shape
 *   inference is available, including context-local shape callbacks. Otherwise
 *   requires complete output tensor annotations to preserve function inference.
 *   Does not change the conservative custom-operator constant-folding policy.
 *   Keeps nodes referencing enclosing-function attributes, including references
 *   inside graph attributes, outside extracted regions.
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
 *   contraction cycles and preserves the original execution order except for
 *   literal Constants and constant Cast/CastLike/Identity preparations, which may
 *   move before a call that needs their values without evaluating any weights.
 *   Retains order for other operators and device-annotated or overridden conversions.
 * - Ranks cuts within a grown seed skeleton by crossing-edge multiplicity, then
 *   unknown-size crossings, saturating known byte size, decreasing region size,
 *   and original positions. Ranks seed families by total anchor coverage, total
 *   node coverage, occurrence count (smaller period), then region size.
 * - Revalidates the selected match and replaces every selected occurrence with
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
 * - Considers at most 32 seed groups and 4096 shared compatibility/growth attempts.
 * - Emits at most 32 occurrences per batch, with one un-emitted tail probe.
 *   Shortens batches when needed to leave at least two occurrences for the next batch.
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
OperatorGroupProposal ProposeRepeatedLayers(builder::GraphGraph &graph);

/// Revalidates the proposed interval and builds its replacement nodes.
/// Appends or reuses local functions and imports ai.rt opset 1, but does not
/// replace graph nodes; the caller must install the result at proposal.first.
/// Throws builder::BuilderError for an empty, inconsistent or ineligible interval.
utils::RepeatedProtoField<NodeProto>
MakeRepeatedLayerReplacements(builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes);

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
