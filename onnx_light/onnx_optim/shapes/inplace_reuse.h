// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file inplace_reuse.h
 * @brief Heuristic that leverages the shapes inferred by
 *        :cpp:class:`onnx_optim::shapes::ShapesContext` to guess, for
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
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Classifies how an input buffer compares in size with the output that
 * reuses it:
 *
 *   - :cpp:enumerator:`kEqual`: the input and output have the same
 *     element type and identical shape, so the buffers have the same
 *     byte size. This is the preferred, space-optimal reuse.
 *   - :cpp:enumerator:`kGreater`: the input buffer is strictly larger in
 *     bytes than the output, so the output still fits but leaves part of
 *     the buffer unused.
 */
enum class InPlaceReuseKind {
  kEqual,
  kGreater,
};

/**
 * A single in-place reuse opportunity for one node: the output at
 * position :cpp:var:`output_index` may reuse the buffer of the input at
 * position :cpp:var:`input_index` (both indices refer to the node's
 * ``output()`` / ``input()`` lists). :cpp:var:`kind` records whether the
 * input buffer has the same size as the output
 * (:cpp:enumerator:`InPlaceReuseKind::kEqual`) or is strictly larger
 * (:cpp:enumerator:`InPlaceReuseKind::kGreater`).
 */
struct InPlaceReuse {
  int64_t output_index = -1;
  int64_t input_index = -1;
  InPlaceReuseKind kind = InPlaceReuseKind::kEqual;

  bool operator==(const InPlaceReuse &other) const noexcept {
    return output_index == other.output_index && input_index == other.input_index &&
           kind == other.kind;
  }
  bool operator!=(const InPlaceReuse &other) const noexcept { return !(*this == other); }
};

/**
 * Guesses, for every node of ``graph``, which outputs may reuse which
 * input buffers in place, using the shapes and element types already
 * inferred into ``ctx`` (typically by
 * :cpp:func:`ShapesContext::ComputeShapeGraph` or
 * :cpp:func:`ShapesContext::ComputeShapeModel`).
 *
 * @param graph  Graph whose nodes are analysed, in topological order.
 * @param ctx    Shapes context already populated with the inferred
 *               descriptors for ``graph`` (graph inputs, initializers,
 *               intermediates and outputs).
 * @param allow_input_overwrite  When ``false`` (the default), declared
 *               graph inputs are never offered as reusable buffers, so a
 *               caller's input is never overwritten in place. When
 *               ``true``, a declared graph input may be reused like an
 *               intermediate (subject to the same lifetime and shape
 *               checks), allowing kernels to overwrite it.
 * @return A vector with one entry per node of ``graph`` (same order as
 *         ``graph.node()``); each entry lists the reuse opportunities
 *         discovered for that node. Nodes without any opportunity carry
 *         an empty list.
 */
std::vector<std::vector<InPlaceReuse>> ComputeInPlaceReuse(const GraphProto &graph,
                                                           const ShapesContext &ctx,
                                                           bool allow_input_overwrite = false);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
