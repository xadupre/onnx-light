// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_core/compute/inplace_reuse_types.h"
#include "onnx_core/compute/result_lifetime.h"
#include "onnx_core/expressions/dim_sum.h"
#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file inplace_reuse.h
 * @brief Heuristic that leverages the shapes inferred by
 *        :cpp:class:`core::shapes::ShapesContext` to guess, for
 *        every node of a graph, which output buffers may reuse which
 *        input buffers in place.
 *
 * The analysis is purely structural: it reports the reuse
 * *opportunities* implied by shape inference and value lifetimes, not
 * whether a particular kernel actually performs the reuse. A node's
 * output ``o`` may reuse the buffer of its input ``i`` when:
 *
 *   - both ``o`` and ``i`` carry a tensor descriptor in the populated
 *     :cpp:class:`ShapesContext` (shape inference succeeded for both);
 *   - ``i``'s buffer is large enough to hold ``o``: either ``i`` and
 *     ``o`` share the same element type and identical shape (an
 *     :cpp:enumerator:`InPlaceReuseKind::kEqual` match), or ``i``'s
 *     buffer is strictly larger in bytes than ``o``'s (an
 *     :cpp:enumerator:`InPlaceReuseKind::kGreater` match);
 *   - ``i`` is a graph intermediate (produced by an earlier node, not a
 *     declared graph input, initializer or output) so its buffer is not
 *     shared with the caller. Declared graph inputs are never overwritten
 *     in place unless ``allow_input_overwrite`` is set, in which case an
 *     input that is otherwise reusable (an intermediate-like lifetime, not
 *     also a graph output) may be aliased;
 *   - the node is the last consumer of ``i`` (``i`` is not read again by
 *     any later node, directly or through a subgraph capture), so
 *     overwriting it in place is safe;
 *   - ``i`` appears exactly once among the node's direct inputs, so the
 *     in-place write cannot clobber a second read of the same value.
 *
 * Each input is matched to at most one output and each output to at
 * most one input. ``kEqual`` matches are always preferred over
 * ``kGreater`` ones, since reusing a same-sized buffer wastes no space.
 * The runtime is expected to combine these structural guesses with the
 * kernel-level ``CanRunInPlace()`` capability before actually aliasing
 * buffers.
 *
 * The matching algorithm itself lives here, in
 * :cpp:func:`ComputeInPlaceReuseMatches`. :cpp:class:`ComputeContext`
 * (declared in ``compute_context.h``) calls into it and stores the
 * per-node result alongside its other graph-level annotations (mirroring
 * the way :cpp:class:`core::shapes::ShapesContext` stores inferred
 * descriptors). The free functions :cpp:func:`ComputeInPlaceReuse` and
 * :cpp:func:`WriteInPlaceReuseToMetadata` remain available as thin
 * convenience wrappers around :cpp:class:`ComputeContext`.
 */

namespace ONNX_LIGHT_NAMESPACE::core::compute {

using ::onnx_light::core::shapes::ShapesContext;

/// Returns (and memoizes in ``cache``) the packed byte-size expression of the
/// tensor named ``name`` in ``ctx``, or ``std::nullopt`` when its element type
/// has no fixed bit width (strings, sequences, maps, optionals, undefined).
const std::optional<expressions::DimType> &
GetCachedByteSizeExpr(const ShapesContext &ctx, const std::string &name,
                      std::unordered_map<std::string, std::optional<expressions::DimType>> &cache,
                      expressions::SimplifiedExpressionCache *simplification_cache = nullptr);

/**
 * Core matching algorithm: for every node of ``graph``, pairs each output
 * with at most one input whose buffer it may reuse in place, using the
 * shapes inferred into ``ctx`` and the per-value lifetime information already
 * computed in ``lifetime`` (see :cpp:func:`ComputeResultLifetimeInfo`).
 *
 * @param graph     Graph whose nodes are analysed, in topological order.
 * @param ctx       Shapes context already populated with the inferred
 *                  descriptors for ``graph``.
 * @param lifetime  Per-value lifetime information for ``graph`` (producer /
 *                  last-use maps and the set of names that must never be
 *                  reused in place), computed with the same
 *                  ``allow_input_overwrite`` setting the caller intends.
 * @return A vector with one entry per node of ``graph`` (same order as
 *         ``graph.node()``); each entry lists the reuse opportunities
 *         discovered for that node, ordered by output index. Nodes without
 *         any opportunity carry an empty list.
 */
std::vector<std::vector<InPlaceReuse>>
ComputeInPlaceReuseMatches(const GraphProto &graph, const ShapesContext &ctx,
                           const ResultLifetimeInfo &lifetime);

/**
 * Computes the in-place reuse opportunities for a single node ``node`` at index
 * ``i`` in a graph, given the shapes inferred into ``ctx`` and the lifetime maps
 * (``keep`` / ``producer`` / ``last_use``) accumulated for the nodes up to and
 * including ``node``. This is the per-node core shared by
 * :cpp:func:`ComputeInPlaceReuseMatches` (whole-graph) and the incremental
 * :cpp:func:`ComputeContext::AppendNodeReuse` path, so both produce identical
 * results for the same lifetime state.
 *
 * @param node                 Node to analyse.
 * @param i                    Index of ``node`` within its graph.
 * @param ctx                  Shapes context populated for the graph so far.
 * @param keep                 Names that must never be reused in place.
 * @param producer             Map from value name to its producing node index.
 * @param last_use             Map from value name to the index of its last use.
 * @param byte_size_expr_cache Memoization cache for symbolic byte-size exprs.
 * @param simplified_dim_cache Memoization cache for simplified dimensions.
 * @return The reuse opportunities for ``node``, ordered by output index.
 */
std::vector<InPlaceReuse> ComputeSingleNodeReuse(
    const NodeProto &node, int i, const ShapesContext &ctx,
    const std::unordered_set<std::string> &keep,
    const std::unordered_map<std::string, int> &producer,
    const std::unordered_map<std::string, int> &last_use,
    std::unordered_map<std::string, std::optional<expressions::DimType>> &byte_size_expr_cache,
    expressions::SimplifiedExpressionCache &simplified_dim_cache);

/**
 * Convenience wrapper around :cpp:func:`ComputeContext::ComputeInPlaceReuseGraph`:
 * computes and returns the per-node reuse opportunities for ``graph`` using the
 * shapes already inferred into ``ctx``.
 *
 * @param graph  Graph whose nodes are analysed, in topological order.
 * @param ctx    Shapes context already populated with the inferred
 *               descriptors for ``graph``.
 * @param allow_input_overwrite  See
 *               :cpp:func:`ComputeContext::ComputeInPlaceReuseGraph`.
 * @return A vector with one entry per node of ``graph`` (same order as
 *         ``graph.node()``); each entry lists the reuse opportunities
 *         discovered for that node. Nodes without any opportunity carry an
 *         empty list.
 */
std::vector<std::vector<InPlaceReuse>> ComputeInPlaceReuse(const GraphProto &graph,
                                                           const ShapesContext &ctx,
                                                           bool allow_input_overwrite = false);

/**
 * Convenience wrapper that computes the in-place reuse opportunities for
 * ``graph`` (via :cpp:class:`ComputeContext`) and records them in each node's
 * ``metadata_props`` under :cpp:var:`kInPlaceReuseMetadataKey`,
 * :cpp:var:`kReleaseAfterMetadataKey`, and (when ``value_tags`` is non-empty)
 * :cpp:var:`kReleaseAfterShapeTagMetadataKey`.
 *
 * @param graph       Graph whose nodes are analysed and mutated in place.
 * @param ctx         Shapes context already populated with the inferred
 *                    descriptors for ``graph``.
 * @param value_tags  Optional map from value name to tag string. When
 *                    non-empty, the shape-tagged subset of the release list is
 *                    also written under
 *                    :cpp:var:`kReleaseAfterShapeTagMetadataKey`. See
 *                    :cpp:func:`ComputeContext::ComputeInPlaceReuseGraph`.
 */
void WriteInPlaceReuseToMetadata(
    GraphProto &graph, const ShapesContext &ctx,
    const std::unordered_map<std::string, std::string> &value_tags = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::compute

// Backward-compatibility include: ComputeContext was previously declared in
// this header. Including compute_context.h here ensures that existing code
// that includes inplace_reuse.h continues to see the full ComputeContext API.
#include "onnx_core/compute/compute_context.h"
