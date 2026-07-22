// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_core/annotations/inplace_reuse_types.h"
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
 * The analysis is exposed through :cpp:class:`ComputeContext` (declared
 * in ``compute_context.h``), which stores the per-node result (mirroring
 * the way :cpp:class:`core::shapes::ShapesContext` stores inferred
 * descriptors). The free functions :cpp:func:`ComputeInPlaceReuse` and
 * :cpp:func:`WriteInPlaceReuseToMetadata` remain available as thin
 * convenience wrappers around it.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace annotations {

using ::onnx_light::core::shapes::ShapesContext;

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

} // namespace annotations
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE

// Backward-compatibility include: ComputeContext was previously declared in
// this header. Including compute_context.h here ensures that existing code
// that includes inplace_reuse.h continues to see the full ComputeContext API.
#include "onnx_core/annotations/compute_context.h"
