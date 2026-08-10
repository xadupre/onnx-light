// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/compute/inplace_reuse_types.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/expressions/dim_sum.h"
#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file compute_context.h
 * @brief Graph-level annotation context that combines value-tag inference,
 *        in-place reuse analysis, and per-node memory profiling.
 *
 * :cpp:class:`ComputeContext` stores the results of all three analyses and
 * exposes them in a single object, mirroring the way
 * :cpp:class:`core::shapes::ShapesContext` stores inferred descriptors.
 */

namespace ONNX_LIGHT_NAMESPACE::core::compute {

using ::onnx_light::core::shapes::ShapesContext;

// The symbolic value descriptors live in ``core::symbolic``; bring them
// into ``core::compute`` so this file can keep referring to them
// unqualified.
using ::onnx_light::core::symbolic::Device;
using ::onnx_light::core::symbolic::SymDim;
using ::onnx_light::core::symbolic::SymShape;
using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;

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
inline constexpr const char *ComputeEventActionName(ComputeEventAction action) {
  switch (action) {
  case ComputeEventAction::kInPlace:
    return "inplace";
  case ComputeEventAction::kRelease:
    return "release";
  case ComputeEventAction::kReleaseShapeTag:
    return "release_shape_tag";
  }
  throw std::invalid_argument("ComputeEventActionName: unexpected action value " +
                              std::to_string(static_cast<int32_t>(action)));
}

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
 * The first three keys map to scalar :cpp:type:`core::expressions::DimType`
 * values. The other four keys map to ``std::map<ShapeTag, DimType>`` buckets
 * split by value tag (``"shape"``, ``"axes"``, ``"weight"``, or the empty string
 * for untagged values). ``"already_allocated_bytes"`` is the sum of the
 * ``"inputs"``, ``"initializers"`` and ``"intermediates"`` maps;
 * ``"output_allocation_bytes"`` is the sum of ``"outputs"``; and
 * ``"total_bytes"`` is the sum of those two scalar entries. Every amount is
 * represented as a :cpp:type:`core::expressions::DimType`, so symbolic
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

/**
 * Holds the in-place reuse opportunities computed for a graph, mirroring the
 * way :cpp:class:`core::shapes::ShapesContext` holds the inferred
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
  /// Callback signature for custom value-tag behavior.
  /// Receives ``(ctx, node, node_index)`` where ``ctx`` can be mutated through
  /// :cpp:func:`TrySetValueTag` / :cpp:func:`SetNodeTag`.
  using CustomValueTagFn =
      std::function<void(ComputeContext &, const NodeProto &, std::size_t node_index)>;
  using CustomValueTagMap = std::unordered_map<std::string, CustomValueTagFn>;

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

  /// Seeds an initial value tag for a graph input, initializer, value_info or
  /// output so it participates in the incremental tag inference driven by
  /// :cpp:func:`AppendNodeTags`. Mirrors the seeding performed whole-graph by
  /// :cpp:func:`CollectGraphSeedTags`.
  void SeedValueTag(const std::string &name, const std::string &tag);

  /// Incrementally updates the value/node tags after the node at ``node_index``
  /// (the last node of ``nodes``) has been appended. Only that node and the
  /// nodes whose values it changes are (re)processed through a monotone
  /// worklist — no whole-graph loop — so appending ``N`` nodes stays linear in
  /// the graph size instead of quadratic. The built-in inference rules converge
  /// to the same least fixed point as :cpp:func:`ComputeValueAndNodeTags`.
  void AppendNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes, std::size_t node_index);

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

  // ── Constant analysis (incremental) ───────────────────────────────────
  //
  // A value is *constant* when its content is known before inference (graph
  // initializers, ``Constant`` outputs, or outputs of a deterministic node
  // whose inputs are all constant). These helpers maintain that information
  // incrementally so :cpp:class:`core::builder::GraphBuilder` can track it node
  // by node, mirroring :cpp:func:`SeedValueTag` / :cpp:func:`AppendNodeTags`.
  // The whole-graph metadata writer lives in ``constant_info.h``
  // (:cpp:func:`WriteConstantInfoToMetadata`).

  /// Seeds ``name`` as a constant value (e.g. an initializer) so it
  /// participates in the incremental constant analysis driven by
  /// :cpp:func:`AppendNodeConstant`.
  void SeedConstant(const std::string &name);

  /// Incrementally updates the constant analysis after the node at
  /// ``node_index`` has been appended: records whether the node is constant and,
  /// when it is, marks its outputs as constant values. Appends exactly one entry
  /// to the per-node constant flag list.
  void AppendNodeConstant(const NodeProto &node, std::size_t node_index);

  /// Whether ``name`` is currently known to be a constant value.
  bool IsConstantValue(const std::string &name) const noexcept {
    return constant_values_.count(name) != 0;
  }

  /// Read-only access to the incremental per-node constant flags (``1`` when the
  /// node is constant). One entry per appended node, in graph order.
  const std::vector<char> &NodeConstant() const noexcept { return node_constant_; }

  /// Whether the node at ``node_index`` produces constant outputs.
  ///
  /// @throws std::out_of_range when ``node_index`` is out of bounds.
  bool NodeConstant(std::size_t node_index) const { return node_constant_.at(node_index) != 0; }

  /// Sets or updates a value tag and returns ``true`` when the internal map changed.
  /// Returns ``false`` when ``name`` is empty, when ``tag`` is invalid/empty,
  /// or when setting it would not change the map.
  bool TrySetValueTag(const std::string &name, const std::string &tag);

  /// Sets or updates a per-node tag and returns ``true`` when the internal list changed.
  /// Returns ``false`` when ``tag`` is invalid/empty or does not change the
  /// current value.
  /// @throws std::out_of_range when ``node_index`` is out of bounds.
  bool SetNodeTag(std::size_t node_index, const std::string &tag);

  /// Internal flag helpers used by value-tag inference around custom callbacks.
  void ClearCustomValueTagChangedFlag() noexcept { custom_value_tags_changed_ = false; }
  bool ConsumeCustomValueTagChangedFlag() noexcept {
    const bool changed = custom_value_tags_changed_;
    custom_value_tags_changed_ = false;
    return changed;
  }

  /// Registers or replaces a custom value-tag callback for ``(domain, op_type)``.
  /// ``domain == ""`` is normalized to ``ai.onnx``.
  void SetCustomValueTagFunction(const std::string &domain, const std::string &op_type,
                                 CustomValueTagFn fn) {
    EXT_ENFORCE_INVALID(
        !op_type.empty(),
        "SetCustomValueTagFunction: op_type cannot be empty when registering a custom callback.");
    EXT_ENFORCE_INVALID(static_cast<bool>(fn),
                        "SetCustomValueTagFunction: callback function cannot be null.");
    custom_value_tags_[MakeCustomValueTagKey(domain, op_type)] = std::move(fn);
  }

  /// Returns a pointer to the custom value-tag callback registered for
  /// ``(domain, op_type)``, or ``nullptr`` if none is registered.
  const CustomValueTagFn *GetCustomValueTagFunction(const std::string &domain,
                                                    const std::string &op_type) const {
    auto it = custom_value_tags_.find(MakeCustomValueTagKey(domain, op_type));
    return it == custom_value_tags_.end() ? nullptr : &it->second;
  }

  /// Removes the custom value-tag callback registered for ``(domain, op_type)``.
  bool RemoveCustomValueTagFunction(const std::string &domain, const std::string &op_type) {
    return custom_value_tags_.erase(MakeCustomValueTagKey(domain, op_type)) > 0;
  }

  /// Removes every custom value-tag callback.
  void ClearCustomValueTagFunctions() { custom_value_tags_.clear(); }

  /// Read-only access to all registered custom value-tag callbacks.
  const CustomValueTagMap &CustomValueTagFunctions() const noexcept { return custom_value_tags_; }

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

  /// Seeds the incremental in-place-reuse lifetime state for a declared graph
  /// input (``is_graph_input``) or initializer (``is_initializer``). When
  /// ``allow_input_overwrite`` is ``false`` the value is protected (kept) from
  /// reuse; when ``true`` a graph input is instead made available before the
  /// first node (producer index ``-1``) so it can be reused at its last use.
  /// Mirrors the seeding performed whole-graph by
  /// :cpp:func:`ComputeResultLifetimeInfo`.
  void SeedReuseInput(const std::string &name, bool is_graph_input, bool is_initializer,
                      bool allow_input_overwrite);

  /// Seeds the incremental in-place-reuse lifetime state for a declared graph
  /// output: the value is kept alive and removed from the release / not-used
  /// lists of any earlier node that had treated it as releasable.
  void SeedReuseOutput(const std::string &name);

  /// Incrementally updates the in-place reuse and release-after annotations
  /// after the node ``node`` at ``node_index`` has been appended, using the
  /// shapes already inferred into ``ctx``. Only this node and the previous
  /// last-users of its inputs are touched — no whole-graph loop — via
  /// :cpp:func:`ComputeSingleNodeReuse`. Appends exactly one entry to each of
  /// the per-node result vectors so they stay aligned with :cpp:func:`Size`.
  void AppendNodeReuse(const NodeProto &node, std::size_t node_index, const ShapesContext &ctx);

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

  // ── Shape inference ──────────────────────────────────────────────────
  //
  // ComputeContext owns the ShapesContext driving every downstream
  // analysis (in-place reuse, peak memory) so the inferred descriptors stay
  // alive alongside the reuse / release results.

  /// Runs shape inference on ``graph`` and stores the resulting descriptors in
  /// the :cpp:class:`ShapesContext` owned by ``*this``.
  ///
  /// @param graph  Graph whose nodes are analysed, in topological order.
  /// @return A reference to the owned :cpp:class:`ShapesContext`, now populated.
  const ShapesContext &ComputeShapes(const GraphProto &graph);

  /// Runs shape inference on ``model.graph()`` (also recording opset versions
  /// and local functions from ``model``) and stores the resulting descriptors
  /// in the :cpp:class:`ShapesContext` owned by ``*this``.
  ///
  /// @param model  Model whose main graph is analysed.
  /// @param prefill_with_value_info_output  Forwarded to
  ///        :cpp:func:`ShapesContext::ComputeShapeModel`.
  /// @return A reference to the owned :cpp:class:`ShapesContext`, now populated.
  const ShapesContext &ComputeShapes(const ModelProto &model,
                                     bool prefill_with_value_info_output = false);

  /// Map from value name to the :cpp:class:`SymTensor` describing its shape and
  /// element type, used to seed shape inference for a :cpp:class:`FunctionProto`
  /// or a bare node list (neither of which carries declared input types).
  using InputShapes = std::unordered_map<std::string, SymTensor>;

  /// Runs shape inference on the body of ``function`` and stores the resulting
  /// descriptors in the :cpp:class:`ShapesContext` owned by ``*this``.
  ///
  /// A :cpp:class:`FunctionProto` only names its inputs, so their shapes and
  /// element types must be supplied through ``input_shapes``. Any input consumed
  /// by a node but absent from ``input_shapes`` (and not produced by an earlier
  /// node) makes shape inference throw ``std::invalid_argument``.
  ///
  /// @param function  Function whose nodes are analysed, in topological order.
  /// @param input_shapes  Shapes/types for the function inputs, seeded into the
  ///        owned :cpp:class:`ShapesContext` before inference.
  /// @return A reference to the owned :cpp:class:`ShapesContext`, now populated.
  const ShapesContext &ComputeShapes(const FunctionProto &function,
                                     const InputShapes &input_shapes = {});

  /// Runs shape inference on the node list ``nodes`` and stores the resulting
  /// descriptors in the :cpp:class:`ShapesContext` owned by ``*this``.
  ///
  /// A bare node list has no declared inputs, so the shapes and element types of
  /// every value not produced by the list itself must be supplied through
  /// ``input_shapes``. Any input consumed by a node but absent from
  /// ``input_shapes`` (and not produced by an earlier node) makes shape inference
  /// throw ``std::invalid_argument``.
  ///
  /// @param nodes  Nodes analysed in topological order.
  /// @param input_shapes  Shapes/types for the values consumed by ``nodes`` but
  ///        not produced by them, seeded into the owned
  ///        :cpp:class:`ShapesContext` before inference.
  /// @return A reference to the owned :cpp:class:`ShapesContext`, now populated.
  const ShapesContext &ComputeShapes(const utils::RepeatedProtoField<NodeProto> &nodes,
                                     const InputShapes &input_shapes = {});

  /// Read-only access to the :cpp:class:`ShapesContext` owned by ``*this``.
  /// Empty until :cpp:func:`ComputeShapes` (or :cpp:func:`Compute`) has run.
  const ShapesContext &Shapes() const noexcept { return shapes_; }

  /// Mutable access to the owned :cpp:class:`ShapesContext`, allowing callers to
  /// seed it (opset versions, custom inference functions, ...) before
  /// :cpp:func:`ComputeShapes`.
  ShapesContext &Shapes() noexcept { return shapes_; }

  // ── Peak memory ──────────────────────────────────────────────────────

  /// Computes the estimated peak scratch memory for every node of ``graph``
  /// using the shapes already inferred into the owned :cpp:class:`ShapesContext`
  /// (via :cpp:func:`ComputeShapes`), storing one entry per node in
  /// ``graph.node()`` order. Nodes whose estimate is zero keep a ``0`` entry.
  ///
  /// @param graph   Graph whose nodes are analysed, in topological order.
  /// @param device  Logical device passed to the peak-memory dispatch function.
  /// @return A reference to the stored per-node peak-memory vector.
  const std::vector<int64_t> &ComputePeakMemory(const GraphProto &graph,
                                                Device device = Device::kUndefined);

  /// Read-only access to the per-node peak-memory estimates computed by
  /// :cpp:func:`ComputePeakMemory`. Empty before it has been called.
  const std::vector<int64_t> &PeakMemory() const noexcept { return peak_memory_; }

  /// Peak-memory estimate for the node at ``node_index``.
  ///
  /// @throws std::out_of_range when ``node_index`` is out of bounds.
  int64_t NodePeakMemory(std::size_t node_index) const { return peak_memory_.at(node_index); }

  // ── High-level orchestration ─────────────────────────────────────────

  /// Runs every analysis on ``graph`` in order and stores all results in
  /// ``*this``: shape inference (:cpp:func:`ComputeShapes`), value / node
  /// tagging (:cpp:func:`ComputeValueAndNodeTags`), in-place reuse together
  /// with the release-after and shape-tag classification
  /// (:cpp:func:`ComputeInPlaceReuseGraph`) and per-node peak memory
  /// (:cpp:func:`ComputePeakMemory`).
  ///
  /// @param graph  Graph whose nodes are analysed, in topological order.
  /// @param device  Logical device passed to the peak-memory dispatch function.
  /// @param allow_input_overwrite  Forwarded to
  ///        :cpp:func:`ComputeInPlaceReuseGraph`.
  void Compute(const GraphProto &graph, Device device = Device::kUndefined,
               bool allow_input_overwrite = false);

  /// Same as :cpp:func:`Compute(const GraphProto&, ...)` but seeds shape
  /// inference from ``model`` (opset versions and local functions) before
  /// analysing ``model.graph()``.
  void Compute(const ModelProto &model, Device device = Device::kUndefined,
               bool allow_input_overwrite = false, bool prefill_with_value_info_output = false);

  /// Pushes every computed result into ``graph``: the inferred shapes are
  /// written into ``graph`` (value_info / outputs, via
  /// :cpp:func:`ShapesContext::ApplyInferredShapesToGraph`), the in-place /
  /// release / shape-tag information into node ``metadata_props`` (via
  /// :cpp:func:`WriteToMetadata`) and the per-node peak-memory estimates under
  /// :cpp:var:`kNodePeakMemoryMetadataKey`.
  ///
  /// @param graph  Graph whose value_info and node metadata are mutated in
  ///        place; must be the same graph passed to :cpp:func:`Compute` /
  ///        :cpp:func:`ComputeInPlaceReuseGraph`.
  void WriteToGraph(GraphProto &graph) const;

  /// Same as :cpp:func:`WriteToGraph(GraphProto&)` applied to ``model.graph()``.
  void WriteToModel(ModelProto &model) const;

  /// Creates the :cpp:class:`runtime::ExecutionPlan` derived from every result
  /// stored in ``*this``. It first pushes the information into ``graph`` (see
  /// :cpp:func:`WriteToGraph`) and then builds the plan from the annotated
  /// graph, so the schedule is driven entirely by the analyses held by this
  /// context.
  runtime::ExecutionPlan BuildExecutionPlan(GraphProto &graph) const;

  /// Same as :cpp:func:`BuildExecutionPlan(GraphProto&, ...)` applied to
  /// ``model.graph()``.
  runtime::ExecutionPlan BuildExecutionPlan(ModelProto &model) const;

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
    constant_values_.clear();
    node_constant_.clear();
    reuse_.clear();
    release_after_.clear();
    not_used_after_.clear();
    release_after_shape_tagged_.clear();
    memory_.clear();
    peak_memory_.clear();
    shapes_.Clear();
    node_tag_custom_override_.clear();
    tag_producer_node_.clear();
    tag_consumers_.clear();
    incr_producer_.clear();
    incr_last_use_.clear();
    incr_keep_.clear();
    incr_graph_inputs_.clear();
    incr_graph_initializers_.clear();
    incr_graph_outputs_.clear();
    incr_byte_size_expr_cache_.clear();
    incr_simplified_dim_cache_ = expressions::SimplifiedExpressionCache{};
  }

private:
  static std::string NormalizeDomain(const std::string &domain) {
    return domain.empty() ? std::string(::onnx_light::core::shapes::kOnnxDomain) : domain;
  }

  static std::string MakeCustomValueTagKey(const std::string &domain, const std::string &op_type) {
    return NormalizeDomain(domain) + ":" + op_type;
  }

  std::unordered_map<std::string, std::string> value_tags_;
  std::vector<std::string> node_tags_;
  // Incremental constant analysis state (see SeedConstant / AppendNodeConstant).
  std::unordered_set<std::string> constant_values_;
  std::vector<char> node_constant_;
  bool custom_value_tags_changed_ = false;
  CustomValueTagMap custom_value_tags_;
  std::vector<std::vector<InPlaceReuse>> reuse_;
  std::vector<std::vector<std::string>> release_after_;
  std::vector<std::vector<std::string>> not_used_after_;
  std::vector<std::vector<std::string>> release_after_shape_tagged_;
  std::vector<NodeMemoryProfile> memory_;
  ShapesContext shapes_;
  std::vector<int64_t> peak_memory_;
  ComputeEventLog events_;
  bool events_enabled_ = false;

  // ── Incremental (per-node) annotation state ──────────────────────────
  // Maintained by AppendNodeTags / AppendNodeReuse so each appended node only
  // touches itself and its immediate dependents (no whole-graph rescan).

  // Value/node tag worklist state.
  std::vector<char> node_tag_custom_override_;
  std::unordered_map<std::string, int> tag_producer_node_;
  std::unordered_map<std::string, std::vector<int>> tag_consumers_;

  // In-place reuse lifetime state.
  std::unordered_map<std::string, int> incr_producer_;
  std::unordered_map<std::string, int> incr_last_use_;
  std::unordered_set<std::string> incr_keep_;
  std::unordered_set<std::string> incr_graph_inputs_;
  std::unordered_set<std::string> incr_graph_initializers_;
  std::unordered_set<std::string> incr_graph_outputs_;
  std::unordered_map<std::string, std::optional<expressions::DimType>> incr_byte_size_expr_cache_;
  expressions::SimplifiedExpressionCache incr_simplified_dim_cache_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
