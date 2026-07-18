// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "onnx_optim/expressions.h"
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
 * The analysis is exposed through :cpp:class:`ComputeContext`, which
 * stores the per-node result (mirroring the way
 * :cpp:class:`onnx_optim::shapes::ShapesContext` stores inferred
 * descriptors). The free functions :cpp:func:`ComputeInPlaceReuse` and
 * :cpp:func:`WriteInPlaceReuseToMetadata` remain available as thin
 * convenience wrappers around it.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace annotations {

using ::onnx_light::onnx_optim::shapes::ShapesContext;

/**
 * Classifies how an input buffer compares in size with the output that
 * reuses it:
 *
 *   - :cpp:enumerator:`kEqual`: the input and output buffers have the same
 *     byte size (e.g. a Transpose or a same-total-size Reshape).
 *     This is the preferred, space-optimal reuse.
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
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records a node's in-place reuse opportunities. The associated value is a
 * string with one ``output_index:input_index:kind`` triplet per opportunity
 * (``kind`` is ``equal`` or ``greater``), triplets separated by ``;``.
 */
constexpr const char *kInPlaceReuseMetadataKey = "onnx_light.inplace_reuse";

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records, for every node, which referenced values reach their last use at
 * this node and can therefore be released after the node runs. The associated
 * value is a ``;``-separated list of value names.
 */
constexpr const char *kReleaseAfterMetadataKey = "onnx_light.release_after";

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records, for every node, which declared graph inputs or initializers reach
 * their last use at that node. Those names are not released by the runtime
 * (their lifetime is owned by the caller / model) but this key still exposes
 * the "last read" information. The associated value is a ``;``-separated list
 * of value names.
 */
constexpr const char *kNotUsedAfterMetadataKey = "onnx_light.not_used_after";

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records, for every node, the subset of releasable values that carry the
 * ``"shape"`` value tag (i.e. tensors that represent tensor-shape metadata
 * rather than activation data). The associated value is a ``;``-separated
 * list of value names — a strict subset of the
 * :cpp:var:`kReleaseAfterMetadataKey` entry for the same node. The key is
 * omitted for nodes that have no shape-tagged releasable values.
 */
constexpr const char *kReleaseAfterShapeTagMetadataKey = "onnx_light.release_after_shape_tag";

/**
 * Kind of decision recorded in the optional :cpp:class:`ComputeContext`
 * decision log.
 *
 *  * ``kInPlace`` — one output was matched to one input for in-place reuse.
 *  * ``kRelease`` — one value reached its last use at a node and can be
 *                   released after that node.
 *  * ``kReleaseShapeTag`` — one released value was also classified as
 *                           ``"shape"`` by value tagging.
 */
enum class ComputeEventAction : int32_t {
  kInPlace = 0,
  kRelease = 1,
  kReleaseShapeTag = 2,
};

/// Returns the short lowercase label for ``action``.
const char *ComputeEventActionName(ComputeEventAction action);

/**
 * One entry of the optional :cpp:class:`ComputeContext` decision log.
 */
struct ComputeEvent {
  /// Decision kind.
  ComputeEventAction action = ComputeEventAction::kInPlace;
  /// Node index in ``graph.node()`` where the decision was made.
  int64_t node_index = -1;
  /// Value name for ``kRelease`` / ``kReleaseShapeTag`` decisions.
  std::string name;
  /// Output index for ``kInPlace`` decisions; ``-1`` otherwise.
  int64_t output_index = -1;
  /// Input index for ``kInPlace`` decisions; ``-1`` otherwise.
  int64_t input_index = -1;
  /// Match kind for ``kInPlace`` decisions.
  InPlaceReuseKind kind = InPlaceReuseKind::kEqual;
};

using ComputeEventLog = std::vector<ComputeEvent>;

/**
 * Represents a per-node memory snapshot computed by
 * :cpp:class:`ComputeContext`.
 *
 * The snapshot represents the memory footprint visible while one node runs:
 *
 *   - ``already_allocated_bytes`` is the sum of the buffers already alive before
 *     the node starts (declared inputs, initializers and still-live
 *     intermediates), after shape inference and lifetime analysis;
 *   - ``output_allocation_bytes`` is the additional memory that must be
 *     allocated for the node's outputs because no eligible in-place reuse
 *     opportunity covers them;
 *   - ``total_bytes`` is their sum.
 *
 * The profile is stored as a ``std::map`` with seven well-known keys:
 *
 *   - ``"total_bytes"``
 *   - ``"already_allocated_bytes"``
 *   - ``"output_allocation_bytes"``
 *   - ``"inputs"``
 *   - ``"initializers"``
 *   - ``"intermediates"``
 *   - ``"outputs"``
 *
 * The first three keys map to scalar :cpp:type:`onnx_optim::expressions::DimType`
 * values. The other four keys map to ``std::map<ShapeTag, DimType>`` buckets
 * split by value tag (``"shape"``, ``"axes"``, ``"weight"``, or the empty string
 * for untagged values). ``"already_allocated_bytes"`` is the sum of the
 * ``"inputs"``, ``"initializers"`` and ``"intermediates"`` maps;
 * ``"output_allocation_bytes"`` is the sum of ``"outputs"``; and
 * ``"total_bytes"`` is the sum of those two scalar entries. Every amount is
 * represented as a :cpp:type:`onnx_optim::expressions::DimType`, so symbolic
 * shapes retain their expression form instead of being dropped. The ``"outputs"``
 * map only counts the extra allocations performed at this node; outputs that
 * reuse an existing input buffer in place contribute no additional bytes there.
 */
using ShapeTag = std::string;
using TaggedMemory = std::map<ShapeTag, expressions::DimType>;
using NodeMemoryProfileValue = std::variant<expressions::DimType, TaggedMemory>;
using NodeMemoryProfile = std::map<std::string, NodeMemoryProfileValue>;

constexpr const char *kNodeMemoryTotalBytesKey = "total_bytes";
constexpr const char *kNodeMemoryAlreadyAllocatedBytesKey = "already_allocated_bytes";
constexpr const char *kNodeMemoryOutputAllocationBytesKey = "output_allocation_bytes";
constexpr const char *kNodeMemoryInputsKey = "inputs";
constexpr const char *kNodeMemoryInitializersKey = "initializers";
constexpr const char *kNodeMemoryIntermediatesKey = "intermediates";
constexpr const char *kNodeMemoryOutputsKey = "outputs";

inline expressions::DimType &NodeMemoryProfileScalar(NodeMemoryProfile &profile,
                                                     const std::string &key) {
  auto it = profile.find(key);
  if (it == profile.end()) {
    it = profile.emplace(key, expressions::DimType{int64_t{0}}).first;
  }
  return std::get<expressions::DimType>(it->second);
}

inline const expressions::DimType &NodeMemoryProfileScalar(const NodeMemoryProfile &profile,
                                                           const std::string &key) {
  static const expressions::DimType zero = int64_t{0};
  auto it = profile.find(key);
  if (it == profile.end()) {
    return zero;
  }
  return std::get<expressions::DimType>(it->second);
}

inline TaggedMemory &NodeMemoryProfileBucket(NodeMemoryProfile &profile, const std::string &key) {
  auto it = profile.find(key);
  if (it == profile.end()) {
    it = profile.emplace(key, TaggedMemory{}).first;
  }
  return std::get<TaggedMemory>(it->second);
}

inline const TaggedMemory &NodeMemoryProfileBucket(const NodeMemoryProfile &profile,
                                                   const std::string &key) {
  static const TaggedMemory empty;
  auto it = profile.find(key);
  if (it == profile.end()) {
    return empty;
  }
  return std::get<TaggedMemory>(it->second);
}

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
class ComputeContext {
public:
  ComputeContext() = default;

  /**
   * Infers semantic ``shape`` / ``axes`` / ``weight`` tags for the values and
   * nodes in ``graph`` and stores the result in ``*this`` (replacing any
   * previously computed tags).
   *
   * @return A pair ``(value_tags, node_tags)`` where ``value_tags`` maps value
   *         names to their inferred tag and ``node_tags`` follows the order of
   *         ``graph.node()``.
   */
  std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
  ComputeValueAndNodeTags(const GraphProto &graph);

  /**
   * Same as :cpp:func:`ComputeValueAndNodeTags(const GraphProto&)` but for a
   * function body.
   */
  std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
  ComputeValueAndNodeTags(const FunctionProto &function);

  /**
   * Same as :cpp:func:`ComputeValueAndNodeTags(const GraphProto&)` but for an
   * arbitrary node list.
   */
  std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
  ComputeValueAndNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes);

  /**
   * Same as :cpp:func:`ComputeValueAndNodeTags(const utils::RepeatedProtoField<NodeProto>&)`
   * but for callers that still materialize node vectors.
   */
  std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
  ComputeValueAndNodeTags(const std::vector<NodeProto> &nodes);

  /// Read-only access to the last value-tag map computed through
  /// :cpp:func:`ComputeValueAndNodeTags`.
  const std::unordered_map<std::string, std::string> &ValueTags() const noexcept {
    return value_tags_;
  }

  /// Read-only access to the last per-node tag list computed through
  /// :cpp:func:`ComputeValueAndNodeTags`.
  const std::vector<std::string> &NodeTags() const noexcept { return node_tags_; }

  /// Tag inferred for the node at ``node_index``.
  ///
  /// @throws std::out_of_range when ``node_index`` is out of bounds.
  const std::string &NodeTag(std::size_t node_index) const { return node_tags_.at(node_index); }

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
   * @param value_tags  Optional map from value name to tag string (``"shape"``,
   *               ``"axes"``, ``"weight"``). When non-empty, values in the
   *               release list that carry the ``"shape"`` tag are also stored
   *               separately and exposed through
   *               :cpp:func:`ReleaseAfterShapeTagged` /
   *               :cpp:func:`NodeReleaseAfterShapeTagged`, and written to
   *               :cpp:var:`kReleaseAfterShapeTagMetadataKey` by
   *               :cpp:func:`WriteToMetadata`.
   */
  void
  ComputeInPlaceReuseGraph(const GraphProto &graph, const ShapesContext &ctx,
                           bool allow_input_overwrite = false,
                           const std::unordered_map<std::string, std::string> &value_tags = {});

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

  /// Read-only access to the per-node shape-tagged releasable values. When
  /// :cpp:func:`ComputeInPlaceReuseGraph` was called with a non-empty
  /// ``value_tags`` map, this vector has one entry per node (same order as
  /// ``graph.node()``), and entry ``i`` lists the names from the
  /// ``release_after`` list that carry the ``"shape"`` value tag. When
  /// ``ComputeInPlaceReuseGraph`` was called without ``value_tags`` (or with
  /// an empty map), this vector is itself empty.
  const std::vector<std::vector<std::string>> &ReleaseAfterShapeTagged() const noexcept {
    return release_after_shape_tagged_;
  }

  /// Shape-tagged releasable values for the node at ``node_index``.
  ///
  /// @throws std::out_of_range when ``node_index`` is out of bounds, or when
  ///         :cpp:func:`ComputeInPlaceReuseGraph` was called without value tags
  ///         (in which case the vector is empty and every access is out of
  ///         bounds).
  const std::vector<std::string> &NodeReleaseAfterShapeTagged(std::size_t node_index) const {
    return release_after_shape_tagged_.at(node_index);
  }

  /// Read-only access to the per-node memory snapshots. Entry ``i`` describes
  /// the memory footprint observed while running ``graph.node()[i]``.
  const std::vector<NodeMemoryProfile> &Memory() const noexcept { return memory_; }

  /// Memory snapshot for the node at ``node_index``.
  ///
  /// @throws std::out_of_range when ``node_index`` is out of bounds.
  const NodeMemoryProfile &NodeMemory(std::size_t node_index) const {
    return memory_.at(node_index);
  }

  // ── Optional decision logging ────────────────────────────────────────
  //
  // Mirrors the opt-in event logs of RuntimeContext and ShapesContext.
  // When disabled (the default), no ComputeEvent object is constructed.
  void set_events_enabled(bool enabled) noexcept { events_enabled_ = enabled; }
  bool events_enabled() const noexcept { return events_enabled_; }

  /// Append-only log of decisions made by :cpp:func:`ComputeInPlaceReuseGraph`.
  const ComputeEventLog &Events() const noexcept { return events_; }
  ComputeEventLog &Events() noexcept { return events_; }

  /// Empties the decision log without touching computed results.
  void ClearEvents() noexcept { events_.clear(); }

  /**
   * Records the computed opportunities into each node's ``metadata_props`` of
   * ``graph`` under :cpp:var:`kInPlaceReuseMetadataKey`,
   * :cpp:var:`kReleaseAfterMetadataKey`,
   * :cpp:var:`kNotUsedAfterMetadataKey`, and (when
   * :cpp:func:`ComputeInPlaceReuseGraph` was called with value tags)
   * :cpp:var:`kReleaseAfterShapeTagMetadataKey`.
   *
   * For every node that has at least one in-place opportunity, a single
   * metadata entry is added (or updated in place if the key already exists)
   * whose value lists the opportunities as
   * ``output_index:input_index:kind`` triplets separated by ``;`` (``kind``
   * being ``equal`` or ``greater``).
   *
   * For every node that has releasable last-use inputs, one metadata entry is
   * added (or updated in place) under :cpp:var:`kReleaseAfterMetadataKey`; the
   * value is a ``;``-separated list of releasable names.
   *
   * For every node that has declared graph inputs / initializers reaching
   * their last use, one metadata entry is added (or updated in place) under
   * :cpp:var:`kNotUsedAfterMetadataKey`; the value is a ``;``-separated list
   * of those names.
   *
   * When shape-tag information was provided to
   * :cpp:func:`ComputeInPlaceReuseGraph`, a further metadata entry is added
   * under :cpp:var:`kReleaseAfterShapeTagMetadataKey` for every node that has
   * at least one shape-tagged releasable value; the value is a
   * ``;``-separated list of those names.
   *
   * Nodes without in-place opportunities, without releasable names, and
   * without last-use input/initializer names are left untouched.
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
  void Clear() noexcept {
    value_tags_.clear();
    node_tags_.clear();
    reuse_.clear();
    release_after_.clear();
    not_used_after_.clear();
    release_after_shape_tagged_.clear();
    memory_.clear();
  }

private:
  std::unordered_map<std::string, std::string> value_tags_;
  std::vector<std::string> node_tags_;
  std::vector<std::vector<InPlaceReuse>> reuse_;
  std::vector<std::vector<std::string>> release_after_;
  std::vector<std::vector<std::string>> not_used_after_;
  std::vector<std::vector<std::string>> release_after_shape_tagged_;
  std::vector<NodeMemoryProfile> memory_;
  ComputeEventLog events_;
  bool events_enabled_ = false;
};

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
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
