// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
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
 *
 * The analysis is exposed through :cpp:class:`InplaceContext`, which
 * stores the per-node result (mirroring the way
 * :cpp:class:`onnx_optim::shapes::ShapesContext` stores inferred
 * descriptors). The free functions :cpp:func:`ComputeInPlaceReuse` and
 * :cpp:func:`WriteInPlaceReuseToMetadata` remain available as thin
 * convenience wrappers around it.
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
 * Metadata key under which :cpp:func:`InplaceContext::WriteToMetadata`
 * records a node's in-place reuse opportunities. The associated value is a
 * string with one ``output_index:input_index:kind`` triplet per opportunity
 * (``kind`` is ``equal`` or ``greater``), triplets separated by ``;``.
 */
constexpr const char *kInPlaceReuseMetadataKey = "onnx_light.inplace_reuse";

/**
 * Holds the in-place reuse opportunities computed for a graph, mirroring the
 * way :cpp:class:`onnx_optim::shapes::ShapesContext` holds the inferred
 * descriptors.
 *
 * The reuse guess is purely structural: it reports the opportunities implied
 * by shape inference and value lifetimes, not whether a particular kernel
 * actually performs the reuse. Populate the context with
 * :cpp:func:`ComputeInPlaceReuseGraph` (consuming a :cpp:class:`ShapesContext`
 * already filled by :cpp:func:`ShapesContext::ComputeShapeGraph` or
 * :cpp:func:`ShapesContext::ComputeShapeModel`), then read the result through
 * :cpp:func:`Reuse` / :cpp:func:`NodeReuse` or persist it into the graph with
 * :cpp:func:`WriteToMetadata`.
 */
class InplaceContext {
public:
  InplaceContext() = default;

  /**
   * Guesses, for every node of ``graph``, which outputs may reuse which input
   * buffers in place, using the shapes and element types already inferred
   * into ``ctx``, and stores the result in ``*this`` (replacing any
   * previously computed result).
   *
   * @param graph  Graph whose nodes are analysed, in topological order.
   * @param ctx    Shapes context already populated with the inferred
   *               descriptors for ``graph`` (graph inputs, initializers,
   *               intermediates and outputs).
   * @param allow_input_overwrite  When ``false`` (the default), declared
   *               graph inputs are never offered as reusable buffers, so a
   *               caller's input is never overwritten in place. When ``true``,
   *               a declared graph input may be reused like an intermediate
   *               (subject to the same lifetime and shape checks), allowing
   *               kernels to overwrite it.
   */
  void ComputeInPlaceReuseGraph(const GraphProto &graph, const ShapesContext &ctx,
                                bool allow_input_overwrite = false);

  /// Number of nodes for which reuse has been computed (one entry per node of
  /// the analysed graph, in ``graph.node()`` order). Zero before
  /// :cpp:func:`ComputeInPlaceReuseGraph` has been called.
  std::size_t Size() const noexcept { return reuse_.size(); }

  /// ``true`` when no reuse has been computed yet.
  bool Empty() const noexcept { return reuse_.empty(); }

  /// Read-only access to the per-node reuse opportunities. Entry ``i`` lists
  /// the opportunities discovered for ``graph.node()[i]``; nodes without any
  /// opportunity carry an empty list.
  const std::vector<std::vector<InPlaceReuse>> &Reuse() const noexcept { return reuse_; }

  /// Reuse opportunities discovered for the node at ``node_index``.
  ///
  /// @throws std::out_of_range when ``node_index`` is out of bounds.
  const std::vector<InPlaceReuse> &NodeReuse(std::size_t node_index) const {
    return reuse_.at(node_index);
  }

  /**
   * Records the computed opportunities into each node's ``metadata_props`` of
   * ``graph`` under :cpp:var:`kInPlaceReuseMetadataKey`.
   *
   * For every node that has at least one opportunity, a single metadata entry
   * is added (or updated in place if the key already exists) whose value lists
   * the opportunities as ``output_index:input_index:kind`` triplets separated
   * by ``;`` (``kind`` being ``equal`` or ``greater``). Nodes without any
   * opportunity are left untouched.
   *
   * ``graph`` must be the same graph passed to
   * :cpp:func:`ComputeInPlaceReuseGraph`, so that node indices line up with
   * the stored result.
   *
   * @param graph  Graph whose nodes are mutated in place.
   * @throws std::invalid_argument when ``graph`` has a different number of
   *         nodes than the result stored in ``*this``.
   */
  void WriteToMetadata(GraphProto &graph) const;

  /// Empties the stored result.
  void Clear() noexcept { reuse_.clear(); }

private:
  std::vector<std::vector<InPlaceReuse>> reuse_;
};

/**
 * Convenience wrapper around :cpp:func:`InplaceContext::ComputeInPlaceReuseGraph`:
 * computes and returns the per-node reuse opportunities for ``graph`` using the
 * shapes already inferred into ``ctx``.
 *
 * @param graph  Graph whose nodes are analysed, in topological order.
 * @param ctx    Shapes context already populated with the inferred
 *               descriptors for ``graph``.
 * @param allow_input_overwrite  See
 *               :cpp:func:`InplaceContext::ComputeInPlaceReuseGraph`.
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
 * ``graph`` (via :cpp:class:`InplaceContext`) and records them in each node's
 * ``metadata_props`` under :cpp:var:`kInPlaceReuseMetadataKey`.
 *
 * @param graph  Graph whose nodes are analysed and mutated in place.
 * @param ctx    Shapes context already populated with the inferred descriptors
 *               for ``graph``.
 */
void WriteInPlaceReuseToMetadata(GraphProto &graph, const ShapesContext &ctx);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
