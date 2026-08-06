// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_core/symbolic/sym_map.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/simple_string.h"

/**
 * @file shapes_context.h
 * @brief Name → :cpp:class:`SymTensor` map shared by all
 *        ``onnx_shapes`` shape-inference functions.
 *
 * ``ShapesContext`` is the in/out parameter consumed and produced by
 * the per-operator ``ComputeShape*`` functions (for example
 * :cpp:func:`ComputeShapeAbs`). It holds the
 * :cpp:class:`SymTensor` descriptors of every named value (graph
 * input, initializer or intermediate result) currently known to a
 * shape-inference pass. ``ComputeShape*`` functions read the entries
 * corresponding to a node's inputs and insert new entries for the
 * node's outputs.
 */

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

// The symbolic value descriptors (SymDim, SymShape, SymTensor, SymSequence,
// TensorType, ...) live in ``onnx_core::symbolic`` so both ``onnx_op`` and
// ``onnx_shapes`` can share them. Bring them into ``onnx_shapes::shapes`` so
// the whole shape-inference stack can keep referring to them unqualified.
using ::onnx_light::core::symbolic::DataTypeToTensorType;
using ::onnx_light::core::symbolic::Device;
using ::onnx_light::core::symbolic::GPUIndex;
using ::onnx_light::core::symbolic::IsGPU;
using ::onnx_light::core::symbolic::IsIntegerTensorType;
using ::onnx_light::core::symbolic::kMaxGPUIndex;
using ::onnx_light::core::symbolic::kMaxOptimRank;
using ::onnx_light::core::symbolic::kOptimValueAsShapeMaxElements;
using ::onnx_light::core::symbolic::kValueInfoDeviceMetadataKey;
using ::onnx_light::core::symbolic::kValueInfoMaxMetadataKey;
using ::onnx_light::core::symbolic::kValueInfoMinMetadataKey;
using ::onnx_light::core::symbolic::ShapeFromTensorProtoDims;
using ::onnx_light::core::symbolic::SymCmpResult;
using ::onnx_light::core::symbolic::SymDim;
using ::onnx_light::core::symbolic::SymMap;
using ::onnx_light::core::symbolic::SymSequence;
using ::onnx_light::core::symbolic::SymShape;
using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

/// Sentinel value returned by :cpp:func:`ShapesContext::OpsetVersion`
/// when no opset version has been recorded for the requested domain.
inline constexpr int kUnknownOpsetVersion = -1;

/// Canonical domain string used for the standard ONNX operator set
/// (``ai.onnx``). An empty domain on a ``NodeProto`` is treated as
/// equivalent to this value.
inline constexpr const char *kOnnxDomain = "ai.onnx";

/// Canonical domain string for the traditional ML operator set
/// (``ai.onnx.ml``), shared by :cpp:class:`ShapesContext` domain
/// validation and the ``onnx_shapes::shapes::traditionalml`` shape
/// functions, which alias this constant instead of redefining it.
inline constexpr const char *kOnnxMlDomain = "ai.onnx.ml";

/// Canonical domain string for the preview operator set
/// (``ai.onnx.preview``). See :cpp:var:`kOnnxMlDomain` for rationale.
inline constexpr const char *kOnnxPreviewDomain = "ai.onnx.preview";

/// Canonical domain string for the training-preview operator set
/// (``ai.onnx.preview.training``). See :cpp:var:`kOnnxMlDomain` for
/// rationale.
inline constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

/// Canonical domain string for the runtime-only operator set
/// (``ai.rt``). See :cpp:var:`kOnnxMlDomain` for rationale.
inline constexpr const char *kAiRtDomain = "ai.rt";

/**
 * Kind of shape-inference event recorded in the
 * :cpp:class:`ShapesContext` event log. Mirrors the runtime
 * :cpp:enum:`onnx_kernels::RuntimeEventAction` design.
 *
 *  * ``kAdd``         — a new tensor descriptor was inserted via
 *                       :cpp:func:`ShapesContext::Set` on a previously
 *                       absent name.
 *  * ``kReplace``     — an existing tensor descriptor was overwritten
 *                       via :cpp:func:`ShapesContext::Set`.
 *  * ``kComputeNode`` — shape inference was dispatched for a single
 *                       :cpp:class:`NodeProto`. The event records the
 *                       node's ``op_domain`` / ``op_type`` and the list
 *                       of ``inputs`` it consumed. It does not mutate the
 *                       tensor map by itself.
 *  * ``kConstraint``  — a new symbolic-dimension equality constraint
 *                       (``a == b``) was recorded via
 *                       :cpp:func:`ShapesContext::AddConstraint`. The two
 *                       operands are stored in ``inputs``.
 *  * ``kConstraintMax`` — a new symbolic-dimension upper-bound constraint
 *                       (``lhs <= rhs``) was recorded via
 *                       :cpp:func:`ShapesContext::AddLessEqualConstraint`.
 *                       The two operands are stored in ``inputs``.
 */
enum class ShapeEventAction : int32_t {
  kAdd = 0,
  kReplace = 1,
  kComputeNode = 2,
  kConstraint = 3,
  kConstraintMax = 4
};

/**
 * Returns a short lowercase label for ``action`` (``"add"``,
 * ``"replace"``, ``"compute_node"``). Useful for human-readable
 * rendering of the event log.
 */
inline constexpr const char *ShapeEventActionName(ShapeEventAction action) noexcept {
  switch (action) {
  case ShapeEventAction::kAdd:
    return "add";
  case ShapeEventAction::kReplace:
    return "replace";
  case ShapeEventAction::kComputeNode:
    return "compute_node";
  case ShapeEventAction::kConstraint:
    return "constraint";
  case ShapeEventAction::kConstraintMax:
    return "constraint_max";
  }
  return "unknown";
}

/**
 * Single entry of the :cpp:class:`ShapesContext` event log.
 *
 * Each insertion or replacement of a tensor descriptor performed
 * through :cpp:func:`ShapesContext::Set` produces one ``ShapeEvent``
 * capturing the action, the name of the value and a snapshot of the
 * descriptor's element type and shape. Because shape inference works on
 * descriptors rather than data, no element values are captured; the
 * ``shape`` is recorded as a list of per-dimension strings so symbolic
 * dimensions (``"N"``, ``"2*N"``) are preserved alongside concrete
 * integer dimensions.
 *
 * :cpp:enumerator:`ShapeEventAction::kComputeNode` events instead
 * summarise the dispatch of a single ``NodeProto``: they carry the
 * node's ``op_domain`` / ``op_type`` and the list of ``inputs``
 * consumed; ``data_type`` is set to ``DataType::UNDEFINED`` and
 * ``shape`` is left empty.
 *
 * :cpp:enumerator:`ShapeEventAction::kConstraint` /
 * :cpp:enumerator:`ShapeEventAction::kConstraintMax` events record a
 * newly inserted symbolic-dimension constraint; the two operands are
 * stored in ``inputs`` (``{a, b}`` for an equality ``a == b``,
 * ``{lhs, rhs}`` for an upper bound ``lhs <= rhs``).
 */
struct ShapeEvent {
  /// Kind of event recorded by this entry.
  ShapeEventAction action = ShapeEventAction::kAdd;
  /// Name of the value targeted by the event.
  std::string name;
  /// Element data type of the descriptor at the moment of the event,
  /// encoded as a ``TensorProto::DataType`` integer value. Set to
  /// ``DataType::UNDEFINED`` for ``kComputeNode`` / ``kConstraint`` /
  /// ``kConstraintMax`` events.
  int32_t data_type = 0;
  /// Descriptor shape at the moment of the event, with each dimension
  /// rendered as a string (a decimal integer for concrete dims, the
  /// symbolic expression otherwise). Empty for ``kComputeNode`` /
  /// ``kConstraint`` / ``kConstraintMax`` events.
  std::vector<std::string> shape;
  /// For ``kComputeNode`` events: ONNX op domain of the dispatched
  /// node, normalised so the default domain is reported as
  /// ``"ai.onnx"``. Empty for other events.
  std::string op_domain;
  /// For ``kComputeNode`` events: ONNX ``op_type`` of the dispatched
  /// node. Empty for other events.
  std::string op_type;
  /// For ``kComputeNode`` events: ordered list of input names consumed
  /// by the node, matching ``NodeProto::input``. For ``kConstraint`` /
  /// ``kConstraintMax`` events: the two constraint operands (``{a, b}``
  /// or ``{lhs, rhs}``). Empty for ``kAdd`` / ``kReplace`` events.
  std::vector<std::string> inputs;
  /// Index of the node this event is associated with. For graph inputs it is
  /// ``-1`` and for initializers it is ``-2``. For intermediate / output
  /// descriptors and for ``kComputeNode`` / ``kConstraint`` /
  /// ``kConstraintMax`` events it is the position (``>= 0``) of the producing
  /// / dispatched node in its graph node list. ``-1`` when no producing node
  /// is known.
  int64_t node_index = -1;
  /// Index of the control-flow node in the **parent** graph whose attribute
  /// subgraph produced this event. ``-1`` for events from the top-level graph.
  /// Combined with :cpp:var:`subgraph_attr_name` this uniquely identifies
  /// which operator and which attribute subgraph an event originated from.
  int64_t subgraph_node_index = -1;
  /// Attribute name of the subgraph within the control-flow node identified
  /// by :cpp:var:`subgraph_node_index`: ``"body"`` for :onnx:`Loop` /
  /// :onnx:`Scan`, ``"then_branch"`` or ``"else_branch"`` for :onnx:`If`.
  /// Empty for top-level-graph events.
  std::string subgraph_attr_name;
};

/**
 * Append-only log of shape-inference events recorded by
 * :cpp:class:`ShapesContext`.
 */
using ShapeEventLog = std::vector<ShapeEvent>;

/**
 * Lightweight container shared by the per-operator ``ComputeShape*``
 * shape-inference functions. ``ShapesContext`` carries two pieces of
 * information:
 *
 *   - a ``name → SymTensor`` map describing every named value
 *     (graph input, initializer or intermediate result) currently
 *     known to the shape-inference pass;
 *   - a ``name → SymSequence`` map describing every named
 *     sequence-typed value (the output of ``SequenceConstruct``,
 *     ``SequenceEmpty``, ``SplitToSequence``, ...);
 *   - a ``domain → opset_version`` map mirroring the ``opset_import``
 *     entries of the surrounding ``ModelProto``, so that
 *     ``ComputeShape*`` functions can pick the correct schema
 *     revision when shape inference depends on the operator's opset
 *     version.
 *
 * The context is a thin wrapper and does not own any tensor data: the
 * :cpp:class:`SymTensor` values stored here are themselves
 * non-owning views.
 */
/**
 * @brief Hash functor for ``std::pair<std::string, std::string>``.
 *
 * Used by the ``std::unordered_set`` members that store symbolic-dimension
 * equality and upper-bound constraints.  The hash combines the individual
 * ``std::hash<std::string>`` values using the boost-style hash_combine
 * formula to avoid the trivial collision risk of a plain XOR.
 */
struct PairStringHash {
  /// Combines two string hashes using bit-mixing with the fractional part of
  /// the golden ratio (``2^32 / φ ≈ 0x9e3779b9``) to spread bits uniformly
  /// and reduce collisions compared to a plain XOR.
  static constexpr std::size_t kHashCombineMul = 0x9e3779b9ULL;

  std::size_t operator()(const std::pair<std::string, std::string> &p) const noexcept {
    std::size_t h = std::hash<std::string>{}(p.first);
    h ^= std::hash<std::string>{}(p.second) + kHashCombineMul + (h << 6) + (h >> 2);
    return h;
  }
};

class ShapesContext {
public:
  using CustomComputeShapeFn = std::function<void(ShapesContext &, const NodeProto &)>;
  using CustomShapeInferenceMap = std::unordered_map<std::string, CustomComputeShapeFn>;

  ShapesContext() = default;

  // ── Tensor descriptors ──────────────────────────────────────────────

  /// Inserts or replaces the descriptor for ``name``. ``tensor`` is
  /// consumed; callers must pass an rvalue (use ``std::move``). When
  /// event logging is enabled (see :cpp:func:`set_events_enabled`) a
  /// :cpp:class:`ShapeEvent` describing the new state is appended to the
  /// event log — with action :cpp:enumerator:`ShapeEventAction::kAdd`
  /// when ``name`` was absent and
  /// :cpp:enumerator:`ShapeEventAction::kReplace` otherwise.
  void Set(const std::string &name, SymTensor &&tensor) {
    if (events_enabled_) {
      LogSetEvent(name, tensor);
    }
    tensors_[name] = std::move(tensor);
  }

  /// Overload: ``name`` given as a null-terminated C string.
  void Set(const char *name, SymTensor &&tensor) { Set(std::string(name), std::move(tensor)); }

  /// Returns ``true`` when an entry exists for ``name``.
  bool Has(const std::string &name) const { return tensors_.find(name) != tensors_.end(); }

  /// Returns the descriptor for ``name``. Throws ``std::out_of_range``
  /// if no such entry exists.
  const SymTensor &Get(const std::string &name) const { return tensors_.at(name); }

  /// Number of named entries currently stored.
  std::size_t Size() const noexcept { return tensors_.size(); }

  /// ``true`` when no entries are stored.
  bool Empty() const noexcept { return tensors_.empty(); }

  /// Removes every entry (both tensor descriptors and opset versions).
  void Clear() noexcept {
    tensors_.clear();
    sequences_.clear();
    opsets_.clear();
    local_functions_.clear();
    custom_shape_inference_.clear();
    constraints_.clear();
    subgraph_contexts_.clear();
    topk_k_dims_.clear();
    dim_values_.clear();
  }

  /// Returns (and records) the symbolic dimension name to use for the TopK
  /// output axis driven by the K input named ``k_input_name``. The first
  /// unique K input seen returns ``"TopK_k"``; each subsequent distinct K
  /// input gets ``"TopK_k_2"``, ``"TopK_k_3"``, and so on. Calling this
  /// method twice with the same ``k_input_name`` always returns the same
  /// string.
  const std::string &TopKKDimName(const std::string &k_input_name) {
    auto it = topk_k_dims_.find(k_input_name);
    if (it != topk_k_dims_.end()) {
      return it->second;
    }
    const std::size_t count = topk_k_dims_.size();
    std::string dim_name =
        count == 0 ? std::string("TopK_k") : "TopK_k_" + std::to_string(count + 1);
    return topk_k_dims_.emplace(k_input_name, std::move(dim_name)).first->second;
  }

  /// Read-only access to the underlying map (useful for iteration).
  const std::unordered_map<std::string, SymTensor> &Tensors() const noexcept { return tensors_; }

  // ── Resolved symbolic-dimension values ──────────────────────────────
  //
  // While validating concrete tensor shapes against declared (possibly
  // symbolic) shapes, each symbolic dimension expression (``dim_param``)
  // is bound to the concrete value it first resolves to; every later
  // occurrence of the same expression must agree with that binding. These
  // accessors expose the expression → value store consumed and populated
  // by :cpp:func:`core::symbolic::SymShape::FitsConcreteShape`.

  /// Returns ``true`` when the symbolic dimension expression ``expr`` has
  /// been bound to a concrete value.
  bool HasDimValue(const std::string &expr) const {
    return dim_values_.find(expr) != dim_values_.end();
  }

  /// Returns the concrete value bound to the symbolic dimension expression
  /// ``expr``. Callers should first confirm the binding exists with
  /// :cpp:func:`HasDimValue`; this throws ``std::out_of_range`` otherwise.
  int64_t DimValue(const std::string &expr) const { return dim_values_.at(expr); }

  /// Binds the symbolic dimension expression ``expr`` to the concrete
  /// value ``value``. An existing binding for ``expr`` is overwritten, so
  /// callers that want to detect (rather than replace) a conflicting value
  /// check :cpp:func:`HasDimValue` / :cpp:func:`DimValue` first.
  void SetDimValue(const std::string &expr, int64_t value) { dim_values_[expr] = value; }

  /// Read-only access to the underlying expression → value map (useful
  /// for iteration).
  const std::unordered_map<std::string, int64_t> &DimValues() const noexcept { return dim_values_; }

  // ── Sequence descriptors ────────────────────────────────────────────

  /// Inserts or replaces the descriptor for a sequence-typed value
  /// named ``name``. ``sequence`` is consumed; callers must pass an
  /// rvalue (use ``std::move``).
  void SetSequence(const std::string &name, SymSequence &&sequence) {
    sequences_[name] = std::move(sequence);
  }

  /// Overload: ``name`` given as a null-terminated C string.
  void SetSequence(const char *name, SymSequence &&sequence) {
    sequences_[std::string(name)] = std::move(sequence);
  }

  /// Returns ``true`` when a sequence-typed entry exists for ``name``.
  bool HasSequence(const std::string &name) const {
    return sequences_.find(name) != sequences_.end();
  }

  /// Returns the sequence descriptor for ``name``. Throws
  /// ``std::out_of_range`` if no such entry exists.
  const SymSequence &GetSequence(const std::string &name) const { return sequences_.at(name); }

  /// Number of sequence-typed entries currently stored.
  std::size_t SequencesSize() const noexcept { return sequences_.size(); }

  /// Read-only access to the underlying sequence map (useful for iteration).
  const std::unordered_map<std::string, SymSequence> &Sequences() const noexcept {
    return sequences_;
  }

  // ── Child contexts for control-flow subgraphs ───────────────────────

  /// Key identifying a child :cpp:class:`ShapesContext` retained while
  /// inferring a control-flow node's attribute subgraph: the index of
  /// the control-flow node in this context's graph paired with the name
  /// of the attribute carrying the subgraph (``"body"`` for
  /// :onnx:`Loop` / :onnx:`Scan`, ``"then_branch"`` / ``"else_branch"``
  /// for :onnx:`If`).
  using SubgraphContextKey = std::pair<int64_t, std::string>;

  /// Retains the child context ``context`` produced while inferring the
  /// subgraph ``attr_name`` of the control-flow node at ``node_index`` so
  /// that the subgraph's internal descriptors stay inspectable once the
  /// parent inference has completed. ``context`` is consumed (moved into
  /// the store). Any context previously registered for the same key is
  /// replaced.
  void RegisterSubgraphContext(int64_t node_index, const std::string &attr_name,
                               ShapesContext context);

  /// Returns ``true`` when a child context was registered for the
  /// subgraph ``attr_name`` of the control-flow node at ``node_index``.
  bool HasSubgraphContext(int64_t node_index, const std::string &attr_name) const {
    return subgraph_contexts_.find(SubgraphContextKey(node_index, attr_name)) !=
           subgraph_contexts_.end();
  }

  /// Returns the child context registered for the subgraph ``attr_name``
  /// of the control-flow node at ``node_index``. Throws
  /// ``std::out_of_range`` if no such context exists.
  const ShapesContext &GetSubgraphContext(int64_t node_index, const std::string &attr_name) const {
    return *subgraph_contexts_.at(SubgraphContextKey(node_index, attr_name));
  }

  /// Number of retained child contexts.
  std::size_t SubgraphContextsSize() const noexcept { return subgraph_contexts_.size(); }

  /// Read-only access to the retained child-context map (useful for iteration).
  const std::map<SubgraphContextKey, std::shared_ptr<ShapesContext>> &
  SubgraphContexts() const noexcept {
    return subgraph_contexts_;
  }

  /// Empties the retained child-context map without modifying other
  /// context state (tensor / sequence descriptors, opsets, constraints,
  /// events, ...).
  void ClearSubgraphContexts() noexcept { subgraph_contexts_.clear(); }

  // ── Opset versions ──────────────────────────────────────────────────

  /**
   * Records the opset version of ``domain``. An empty ``domain`` is
   * normalised to :cpp:var:`kOnnxDomain`. Replaces any previous entry
   * for the same domain.
   */
  void SetOpsetVersion(const std::string &domain, int opset_version);

  /// ``true`` when an opset version has been recorded for ``domain``
  /// (after normalising the empty domain to :cpp:var:`kOnnxDomain`).
  bool HasOpsetVersion(const std::string &domain) const;

  /**
   * Returns the recorded opset version of ``domain``, or
   * :cpp:var:`kUnknownOpsetVersion` if none was recorded. An empty
   * ``domain`` is normalised to :cpp:var:`kOnnxDomain`.
   */
  int OpsetVersion(const std::string &domain) const;

  /// Read-only access to the underlying ``domain → opset_version`` map.
  const std::unordered_map<std::string, int> &Opsets() const noexcept { return opsets_; }

  // ── Model-local functions ──────────────────────────────────────────
  //
  // ``onnx_shapes`` shape inference dispatches node-level inference via
  // the registered op schemas (see ``DispatchTable``). A node whose
  // ``op_type`` is not registered may still be valid if it calls a
  // model-local :cpp:class:`FunctionProto` declared in
  // ``ModelProto::functions``. Local functions are addressed by a
  // ``"<domain>:<name>"`` key (matching :cpp:func:`GetFunctionIdentifier`
  // in ``onnx_lib``); :cpp:func:`ComputeShapeModel` registers every
  // entry of ``model.functions()`` here so that
  // :cpp:func:`ComputeShapeNode` can detect and expand the call.

  /// Registers a non-owning pointer to a model-local
  /// :cpp:class:`FunctionProto`. The pointer must remain valid for the
  /// lifetime of the shape-inference pass. Replaces any previous entry
  /// registered under the same ``"<domain>:<name>"`` key. ``func`` must
  /// not be ``nullptr``.
  void SetLocalFunction(const FunctionProto *func) {
    EXT_ENFORCE_INVALID(func != nullptr, "SetLocalFunction: func must not be nullptr.");
    std::string key = func->domain();
    key += ":";
    key += func->name();
    local_functions_[key] = func;
  }

  /// ``true`` when a model-local function is registered for ``key``.
  /// ``key`` is expected to be the ``"<domain>:<name>"`` identifier of
  /// the function.
  bool HasLocalFunction(const std::string &key) const {
    return local_functions_.find(key) != local_functions_.end();
  }

  /// Returns the registered ``FunctionProto`` pointer for ``key``, or
  /// ``nullptr`` when none is registered. ``key`` is expected to be the
  /// ``"<domain>:<name>"`` identifier of the function.
  const FunctionProto *GetLocalFunction(const std::string &key) const {
    auto it = local_functions_.find(key);
    return it == local_functions_.end() ? nullptr : it->second;
  }

  /// Read-only access to the underlying ``"<domain>:<name>" →
  /// FunctionProto*`` map.
  const std::unordered_map<std::string, const FunctionProto *> &LocalFunctions() const noexcept {
    return local_functions_;
  }

  // ── Custom shape-inference hooks ────────────────────────────────────
  //
  // In addition to built-in dispatch-table entries and model-local
  // functions, callers may register a per-(domain, op_type) callback
  // to infer shapes for custom operators.

  /// Registers or replaces a custom shape-inference callback for the
  /// ``(domain, op_type)`` pair. ``domain == ""`` is normalized to
  /// :cpp:var:`kOnnxDomain`.
  void SetCustomShapeInferenceFunction(const std::string &domain, const std::string &op_type,
                                       CustomComputeShapeFn fn);

  /// Returns a pointer to the custom shape-inference callback
  /// registered for ``(domain, op_type)``, or ``nullptr`` if none is
  /// registered. ``domain == ""`` is normalized to :cpp:var:`kOnnxDomain`.
  const CustomComputeShapeFn *GetCustomShapeInferenceFunction(const std::string &domain,
                                                              const std::string &op_type) const;

  /// Removes the custom shape-inference callback registered for
  /// ``(domain, op_type)``. ``domain == ""`` is normalized to
  /// :cpp:var:`kOnnxDomain`. Returns ``true`` when an entry was removed
  /// and ``false`` when no callback matched that key.
  bool RemoveCustomShapeInferenceFunction(const std::string &domain, const std::string &op_type);

  /// Removes every custom shape-inference callback.
  void ClearCustomShapeInferenceFunctions() { custom_shape_inference_.clear(); }

  /// Read-only access to all registered custom shape-inference callbacks.
  const CustomShapeInferenceMap &CustomShapeInferenceFunctions() const noexcept {
    return custom_shape_inference_;
  }

  // ── Symbolic-dimension constraints ──────────────────────────────────
  //
  // Shape inference may discover that two symbolic dimensions must be
  // equal — for example when a graph output declares ``shape=["ANCHOR",
  // 4]`` but node-level inference produces ``shape=["N", 4]`` for the
  // same value. Storing such constraints lets downstream passes
  // unify ``"ANCHOR"`` and ``"N"`` instead of silently picking one of
  // the two names. Constraints are recorded as unordered pairs of
  // strings: a canonical ordering (smaller first lexicographically) is
  // used internally so that ``(a, b)`` and ``(b, a)`` are deduplicated.
  // Self-constraints (``a == a``) are dropped.

  /// Type used to store a single symbolic equality constraint.
  using Constraint = std::pair<std::string, std::string>;

  /// Records that two symbolic dimension names are equal. The pair is
  /// canonicalised so that ``(a, b)`` and ``(b, a)`` are stored only
  /// once, and the trivial self-equality is dropped. Returns ``true``
  /// when a new constraint was inserted, ``false`` otherwise (either
  /// a duplicate or a self-constraint).
  bool AddConstraint(const std::string &a, const std::string &b) {
    if (a == b) {
      return false;
    }
    Constraint c = (a < b) ? Constraint(a, b) : Constraint(b, a);
    const bool inserted = constraints_.insert(c).second;
    if (inserted && events_enabled_) {
      LogConstraintEvent(ShapeEventAction::kConstraint, c.first, c.second);
    }
    return inserted;
  }

  /// ``true`` when an equality constraint between ``a`` and ``b`` is
  /// recorded (canonical order is applied before lookup).
  bool HasConstraint(const std::string &a, const std::string &b) const {
    if (a == b) {
      return true;
    }
    Constraint c = (a < b) ? Constraint(a, b) : Constraint(b, a);
    return constraints_.find(c) != constraints_.end();
  }

  /// Number of recorded constraints.
  std::size_t ConstraintsSize() const noexcept { return constraints_.size(); }

  /// Read-only access to the underlying set of equality constraints.
  /// Each element is a ``(lhs, rhs)`` pair with ``lhs < rhs``.
  const std::unordered_set<Constraint, PairStringHash> &Constraints() const noexcept {
    return constraints_;
  }

  // ── Symbolic-dimension upper-bound constraints ──────────────────────
  //
  // Some operators produce a symbolic output dimension whose runtime
  // value is unknown but is guaranteed to be **less than or equal to**
  // an expression of other dimensions. Two examples:
  //
  //   - ``NonZero(X)`` produces ``Y`` with shape ``(rank(X), nnz)``
  //     where ``nnz`` is bounded above by ``prod(shape(X))``.
  //   - ``Compress(X, cond, axis=k)`` produces an output whose ``k``-th
  //     dimension ``count`` is bounded above by ``X.shape[k]``.
  //   - ``If(...)`` merges two branches; when the matching dims of the
  //     two branches differ the merged dim is bounded above by the
  //     ``max`` of the two branch expressions.
  //
  // Such inequalities are recorded as ordered ``(lhs, rhs)`` pairs
  // meaning ``lhs <= rhs`` where ``lhs`` is a symbolic dimension name
  // and ``rhs`` is an arbitrary (integer-string or symbolic) expression.
  // The trivial case ``lhs == rhs`` is dropped.

  /// Type used to store a single symbolic ``<=`` (less-or-equal)
  /// upper-bound constraint. The first element is the bounded symbol,
  /// the second is an arbitrary dimension expression that upper-bounds
  /// it.
  using LessEqualConstraint = std::pair<std::string, std::string>;

  /// Records that the symbolic dimension named ``lhs`` is
  /// less-than-or-equal-to the expression ``rhs``. The trivial
  /// self-bound (``lhs == rhs``) is dropped, and empty operands are
  /// rejected (they cannot designate a valid dimension name nor a
  /// well-formed bound expression). Returns ``true`` when a new
  /// constraint was inserted, ``false`` otherwise (duplicate,
  /// self-bound, or empty operand).
  bool AddLessEqualConstraint(const std::string &lhs, const std::string &rhs) {
    if (lhs == rhs || lhs.empty() || rhs.empty()) {
      return false;
    }
    const bool inserted = le_constraints_.insert(LessEqualConstraint(lhs, rhs)).second;
    if (inserted && events_enabled_) {
      LogConstraintEvent(ShapeEventAction::kConstraintMax, lhs, rhs);
    }
    return inserted;
  }

  /// ``true`` when a ``lhs <= rhs`` constraint is recorded.
  /// ``lhs == rhs`` always returns ``true``.
  bool HasLessEqualConstraint(const std::string &lhs, const std::string &rhs) const {
    if (lhs == rhs) {
      return true;
    }
    return le_constraints_.find(LessEqualConstraint(lhs, rhs)) != le_constraints_.end();
  }

  /// Number of recorded ``<=`` constraints.
  std::size_t LessEqualConstraintsSize() const noexcept { return le_constraints_.size(); }

  /// Read-only access to the underlying set of ``<=`` constraints. Each
  /// element is an ordered ``(lhs, rhs)`` pair meaning ``lhs <= rhs``.
  const std::unordered_set<LessEqualConstraint, PairStringHash> &
  LessEqualConstraints() const noexcept {
    return le_constraints_;
  }

  // ── Shape-inference entry points ────────────────────────────────────
  //
  // The methods below run shape inference on a single ``NodeProto``, a
  // sequence of nodes, a ``GraphProto`` or an entire ``ModelProto``,
  // writing the inferred descriptors back into this context.

  /// Dispatches a single ``NodeProto`` to the matching per-operator
  /// ``ComputeShape*`` function and stores the resulting output
  /// :cpp:class:`SymTensor` descriptors in ``*this``.
  void ComputeShapeNode(const NodeProto &node);

  /// Throws ``std::invalid_argument`` if any non-empty input name
  /// declared by ``node`` is missing from ``*this``.
  void CheckInputsAvailable(const NodeProto &node) const;

  /// Throws ``std::invalid_argument`` if any non-empty output name
  /// declared by ``node`` already has an entry in ``*this``.
  void CheckOutputsNotAvailable(const NodeProto &node) const;

  /// Runs :cpp:func:`ComputeShapeNode` on every node of ``nodes`` in
  /// order.
  void ComputeShapes(const utils::RepeatedProtoField<NodeProto> &nodes);

  /// Seeds ``*this`` from the initializers and inputs of ``graph`` and
  /// then runs :cpp:func:`ComputeShapes` on its nodes.
  void ComputeShapeGraph(const GraphProto &graph);

  /// Runs shape inference on ``model.graph()``, also recording opset
  /// versions and local functions from ``model``.
  void ComputeShapeModel(const ModelProto &model, bool prefill_with_value_info_output = false);

  /// Writes the shape and element-type descriptors stored in ``*this``
  /// back into ``graph``.
  void ApplyInferredShapesToGraph(GraphProto &graph) const;

  /// Writes the shape and element-type descriptors stored in ``*this``
  /// back into ``model.graph()``.
  void ApplyInferredShapesToModel(ModelProto &model) const;

  // ── Event logging ───────────────────────────────────────────────────
  //
  // Mirrors the opt-in event log of :cpp:class:`core::runtime::RuntimeContext`.
  // When disabled (the default), :cpp:func:`Set`,
  // :cpp:func:`ComputeShapeNode`, :cpp:func:`AddConstraint` and
  // :cpp:func:`AddLessEqualConstraint` skip all event construction,
  // eliminating the bookkeeping overhead from the hot path.

  /// Enables or disables event logging. When disabled (the default),
  /// :cpp:func:`Set`, :cpp:func:`ComputeShapeNode`,
  /// :cpp:func:`AddConstraint` and :cpp:func:`AddLessEqualConstraint`
  /// skip all event construction. Call ``set_events_enabled(true)``
  /// before running shape inference if descriptor or constraint tracing
  /// is required.
  void set_events_enabled(bool enabled) noexcept { events_enabled_ = enabled; }
  bool events_enabled() const noexcept { return events_enabled_; }

  /// Index of the control-flow node in the parent graph currently being
  /// inferred. Set before running subgraph shape inference so that events
  /// recorded inside carry :cpp:var:`ShapeEvent::subgraph_node_index` and
  /// :cpp:var:`ShapeEvent::subgraph_attr_name`. ``-1`` for the top-level
  /// graph. Use :cpp:func:`set_current_subgraph` to update both the index
  /// and the attribute name atomically.
  void set_current_subgraph(int64_t node_index, const std::string &attr_name) {
    current_subgraph_node_index_ = node_index;
    current_subgraph_attr_name_ = attr_name;
  }
  int64_t current_subgraph_node_index() const noexcept { return current_subgraph_node_index_; }
  const std::string &current_subgraph_attr_name() const noexcept {
    return current_subgraph_attr_name_;
  }

  /// Index of the node currently being processed, used to tag the
  /// :cpp:var:`ShapeEvent::node_index` of descriptors and events recorded
  /// during its shape-inference dispatch. Set by :cpp:func:`ComputeShapes`
  /// before each :cpp:func:`ComputeShapeNode` call, ``-2`` while seeding
  /// initializers and ``-1`` while seeding graph inputs (or when no node is
  /// being processed).
  void set_current_node_index(int64_t index) noexcept { current_node_index_ = index; }
  int64_t current_node_index() const noexcept { return current_node_index_; }

  /// Append-only log of every tensor descriptor mutation performed
  /// through :cpp:func:`Set`, every node dispatched through
  /// :cpp:func:`ComputeShapeNode` and every constraint recorded through
  /// :cpp:func:`AddConstraint` / :cpp:func:`AddLessEqualConstraint`. See
  /// :cpp:class:`ShapeEvent` for the captured fields.
  const ShapeEventLog &Events() const noexcept { return events_; }
  ShapeEventLog &Events() noexcept { return events_; }

  /// Empties the event log without otherwise touching the context.
  void ClearEvents() noexcept { events_.clear(); }

  /// Registers a callback invoked synchronously for every newly appended
  /// :cpp:class:`ShapeEvent`. Used by callers that want to stream shape
  /// inference progress while computation is still running.
  void set_event_callback(std::function<void(const ShapeEvent &)> callback) {
    event_callback_ = std::move(callback);
  }
  void clear_event_callback() noexcept { event_callback_ = nullptr; }
  bool has_event_callback() const noexcept { return static_cast<bool>(event_callback_); }

  /// Appends a :cpp:class:`ShapeEvent` with action
  /// :cpp:enumerator:`ShapeEventAction::kComputeNode` summarising the
  /// shape-inference dispatch of a single ``NodeProto`` (its
  /// ``op_domain`` / ``op_type`` and the ``inputs`` it consumed).
  /// Appended by :cpp:func:`ComputeShapeNode` for every dispatched node
  /// when event logging is enabled.
  void AppendComputeNodeEvent(const std::string &op_domain, const std::string &op_type,
                              std::vector<std::string> inputs);

private:
  /// Appends a :cpp:enumerator:`ShapeEventAction::kAdd` /
  /// :cpp:enumerator:`ShapeEventAction::kReplace` event for ``name``
  /// describing ``tensor`` (the descriptor about to be stored). Only
  /// called by :cpp:func:`Set` when event logging is enabled.
  void LogSetEvent(const std::string &name, const SymTensor &tensor);

  /// Appends a :cpp:enumerator:`ShapeEventAction::kConstraint` /
  /// :cpp:enumerator:`ShapeEventAction::kConstraintMax` event recording
  /// the two operands of a newly inserted constraint in ``inputs``. Only
  /// called by :cpp:func:`AddConstraint` /
  /// :cpp:func:`AddLessEqualConstraint` when event logging is enabled.
  void LogConstraintEvent(ShapeEventAction action, const std::string &lhs, const std::string &rhs);

  void EmitEvent(const ShapeEvent &ev) const {
    if (event_callback_) {
      event_callback_(ev);
    }
  }

  std::unordered_map<std::string, SymTensor> tensors_;
  std::unordered_map<std::string, SymSequence> sequences_;
  std::unordered_map<std::string, int> opsets_;
  /// Concrete values bound to symbolic dimension expressions while
  /// validating concrete shapes against declared symbolic shapes (see
  /// :cpp:func:`HasDimValue` / :cpp:func:`SetDimValue`).
  std::unordered_map<std::string, int64_t> dim_values_;
  std::unordered_map<std::string, const FunctionProto *> local_functions_;
  CustomShapeInferenceMap custom_shape_inference_;
  std::unordered_set<Constraint, PairStringHash> constraints_;
  std::unordered_set<LessEqualConstraint, PairStringHash> le_constraints_;
  std::map<SubgraphContextKey, std::shared_ptr<ShapesContext>> subgraph_contexts_;
  ShapeEventLog events_;
  std::function<void(const ShapeEvent &)> event_callback_;
  bool events_enabled_ = false;
  int64_t current_node_index_ = -1;
  /// Index of the control-flow node in the parent graph currently being
  /// inferred (see :cpp:func:`set_current_subgraph`). ``-1`` for the
  /// top-level graph.
  int64_t current_subgraph_node_index_ = -1;
  /// Attribute name of the subgraph currently being inferred (see
  /// :cpp:func:`set_current_subgraph`). Empty for the top-level graph;
  /// set to ``"body"``, ``"then_branch"``, ``"else_branch"``, etc. when
  /// running shape inference for a control-flow body subgraph.
  std::string current_subgraph_attr_name_;
  /// Tracks the mapping from TopK K-input names to the symbolic dimension
  /// name assigned by :cpp:func:`TopKKDimName`. Populated lazily.
  std::unordered_map<std::string, std::string> topk_k_dims_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
