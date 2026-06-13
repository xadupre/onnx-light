// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/simple_string.h"

/**
 * @file shapes_context.h
 * @brief Name → :cpp:class:`OptimTensor` map shared by all
 *        ``onnx_optim`` shape-inference functions.
 *
 * ``ShapesContext`` is the in/out parameter consumed and produced by
 * the per-operator ``ComputeShape*`` functions (for example
 * :cpp:func:`ComputeShapeAbs`). It holds the
 * :cpp:class:`OptimTensor` descriptors of every named value (graph
 * input, initializer or intermediate result) currently known to a
 * shape-inference pass. ``ComputeShape*`` functions read the entries
 * corresponding to a node's inputs and insert new entries for the
 * node's outputs.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/// Sentinel value returned by :cpp:func:`ShapesContext::OpsetVersion`
/// when no opset version has been recorded for the requested domain.
inline constexpr int kUnknownOpsetVersion = -1;

/// Canonical domain string used for the standard ONNX operator set
/// (``ai.onnx``). An empty domain on a ``NodeProto`` is treated as
/// equivalent to this value.
inline constexpr const char *kOnnxDomain = "ai.onnx";

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
const char *ShapeEventActionName(ShapeEventAction action) noexcept;

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
 *   - a ``name → OptimTensor`` map describing every named value
 *     (graph input, initializer or intermediate result) currently
 *     known to the shape-inference pass;
 *   - a ``name → OptimSequence`` map describing every named
 *     sequence-typed value (the output of ``SequenceConstruct``,
 *     ``SequenceEmpty``, ``SplitToSequence``, ...);
 *   - a ``domain → opset_version`` map mirroring the ``opset_import``
 *     entries of the surrounding ``ModelProto``, so that
 *     ``ComputeShape*`` functions can pick the correct schema
 *     revision when shape inference depends on the operator's opset
 *     version.
 *
 * The context is a thin wrapper and does not own any tensor data: the
 * :cpp:class:`OptimTensor` values stored here are themselves
 * non-owning views.
 */
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
  void Set(const std::string &name, OptimTensor &&tensor) {
    if (events_enabled_) {
      LogSetEvent(name, tensor);
    }
    tensors_[name] = std::move(tensor);
  }

  /// Overload: ``name`` given as a null-terminated C string.
  void Set(const char *name, OptimTensor &&tensor) { Set(std::string(name), std::move(tensor)); }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  void Set(const utils::String &name, OptimTensor &&tensor) {
    Set(std::string(name.data(), name.size()), std::move(tensor));
  }

  /// Returns ``true`` when an entry exists for ``name``.
  bool Has(const std::string &name) const { return tensors_.find(name) != tensors_.end(); }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  bool Has(const utils::String &name) const { return Has(std::string(name.data(), name.size())); }

  /// Returns the descriptor for ``name``. Throws ``std::out_of_range``
  /// if no such entry exists.
  const OptimTensor &Get(const std::string &name) const { return tensors_.at(name); }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  const OptimTensor &Get(const utils::String &name) const {
    return Get(std::string(name.data(), name.size()));
  }

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
  }

  /// Read-only access to the underlying map (useful for iteration).
  const std::unordered_map<std::string, OptimTensor> &Tensors() const noexcept { return tensors_; }

  // ── Sequence descriptors ────────────────────────────────────────────

  /// Inserts or replaces the descriptor for a sequence-typed value
  /// named ``name``. ``sequence`` is consumed; callers must pass an
  /// rvalue (use ``std::move``).
  void SetSequence(const std::string &name, OptimSequence &&sequence) {
    sequences_[name] = std::move(sequence);
  }

  /// Overload: ``name`` given as a null-terminated C string.
  void SetSequence(const char *name, OptimSequence &&sequence) {
    sequences_[std::string(name)] = std::move(sequence);
  }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  void SetSequence(const utils::String &name, OptimSequence &&sequence) {
    sequences_[std::string(name.data(), name.size())] = std::move(sequence);
  }

  /// Returns ``true`` when a sequence-typed entry exists for ``name``.
  bool HasSequence(const std::string &name) const {
    return sequences_.find(name) != sequences_.end();
  }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  bool HasSequence(const utils::String &name) const {
    return HasSequence(std::string(name.data(), name.size()));
  }

  /// Returns the sequence descriptor for ``name``. Throws
  /// ``std::out_of_range`` if no such entry exists.
  const OptimSequence &GetSequence(const std::string &name) const { return sequences_.at(name); }

  /// Overload: ``name`` given as a :cpp:class:`utils::String`.
  const OptimSequence &GetSequence(const utils::String &name) const {
    return GetSequence(std::string(name.data(), name.size()));
  }

  /// Number of sequence-typed entries currently stored.
  std::size_t SequencesSize() const noexcept { return sequences_.size(); }

  /// Read-only access to the underlying sequence map (useful for iteration).
  const std::unordered_map<std::string, OptimSequence> &Sequences() const noexcept {
    return sequences_;
  }

  // ── Opset versions ──────────────────────────────────────────────────

  /**
   * Records the opset version of ``domain``. An empty ``domain`` is
   * normalised to :cpp:var:`kOnnxDomain`. Replaces any previous entry
   * for the same domain.
   */
  void SetOpsetVersion(const std::string &domain, int opset_version) {
    opsets_[NormaliseDomain(domain)] = opset_version;
  }

  /// ``true`` when an opset version has been recorded for ``domain``
  /// (after normalising the empty domain to :cpp:var:`kOnnxDomain`).
  bool HasOpsetVersion(const std::string &domain) const {
    return opsets_.find(NormaliseDomain(domain)) != opsets_.end();
  }

  /**
   * Returns the recorded opset version of ``domain``, or
   * :cpp:var:`kUnknownOpsetVersion` if none was recorded. An empty
   * ``domain`` is normalised to :cpp:var:`kOnnxDomain`.
   */
  int OpsetVersion(const std::string &domain) const {
    auto it = opsets_.find(NormaliseDomain(domain));
    return it == opsets_.end() ? kUnknownOpsetVersion : it->second;
  }

  /// Read-only access to the underlying ``domain → opset_version`` map.
  const std::unordered_map<std::string, int> &Opsets() const noexcept { return opsets_; }

  // ── Model-local functions ──────────────────────────────────────────
  //
  // ``onnx_optim`` shape inference dispatches node-level inference via
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
    const std::string key = func->domain().as_string() + ":" + func->name().as_string();
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
                                       CustomComputeShapeFn fn) {
    EXT_ENFORCE_INVALID(!op_type.empty(),
                        "SetCustomShapeInferenceFunction: op_type must not be empty.");
    EXT_ENFORCE_INVALID(static_cast<bool>(fn),
                        "SetCustomShapeInferenceFunction: fn must not be empty.");
    custom_shape_inference_[NormaliseDomain(domain) + ":" + op_type] = std::move(fn);
  }

  /// Returns a pointer to the custom shape-inference callback
  /// registered for ``(domain, op_type)``, or ``nullptr`` if none is
  /// registered. ``domain == ""`` is normalized to :cpp:var:`kOnnxDomain`.
  const CustomComputeShapeFn *GetCustomShapeInferenceFunction(const std::string &domain,
                                                              const std::string &op_type) const {
    auto it = custom_shape_inference_.find(NormaliseDomain(domain) + ":" + op_type);
    return it == custom_shape_inference_.end() ? nullptr : &it->second;
  }

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
  const std::set<Constraint> &Constraints() const noexcept { return constraints_; }

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
  const std::set<LessEqualConstraint> &LessEqualConstraints() const noexcept {
    return le_constraints_;
  }

  // ── Shape-inference entry points ────────────────────────────────────
  //
  // The methods below run shape inference on a single ``NodeProto``, a
  // sequence of nodes, a ``GraphProto`` or an entire ``ModelProto``,
  // writing the inferred descriptors back into this context. See
  // ``onnx_optim/shapes/shape_inference.h`` for the full per-method
  // documentation.

  /// Dispatches a single ``NodeProto`` to the matching per-operator
  /// ``ComputeShape*`` function and stores the resulting output
  /// :cpp:class:`OptimTensor` descriptors in ``*this``.
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
  // Mirrors the opt-in event log of :cpp:class:`onnx_kernels::RuntimeContext`.
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

  /// Append-only log of every tensor descriptor mutation performed
  /// through :cpp:func:`Set`, every node dispatched through
  /// :cpp:func:`ComputeShapeNode` and every constraint recorded through
  /// :cpp:func:`AddConstraint` / :cpp:func:`AddLessEqualConstraint`. See
  /// :cpp:class:`ShapeEvent` for the captured fields.
  const ShapeEventLog &Events() const noexcept { return events_; }
  ShapeEventLog &Events() noexcept { return events_; }

  /// Empties the event log without otherwise touching the context.
  void ClearEvents() noexcept { events_.clear(); }

  /// Appends a :cpp:class:`ShapeEvent` with action
  /// :cpp:enumerator:`ShapeEventAction::kComputeNode` summarising the
  /// shape-inference dispatch of a single ``NodeProto`` (its
  /// ``op_domain`` / ``op_type`` and the ``inputs`` it consumed).
  /// Appended by :cpp:func:`ComputeShapeNode` for every dispatched node
  /// when event logging is enabled.
  void AppendComputeNodeEvent(const std::string &op_domain, const std::string &op_type,
                              std::vector<std::string> inputs);

private:
  static std::string NormaliseDomain(const std::string &domain) {
    return domain.empty() ? std::string(kOnnxDomain) : domain;
  }

  /// Appends a :cpp:enumerator:`ShapeEventAction::kAdd` /
  /// :cpp:enumerator:`ShapeEventAction::kReplace` event for ``name``
  /// describing ``tensor`` (the descriptor about to be stored). Only
  /// called by :cpp:func:`Set` when event logging is enabled.
  void LogSetEvent(const std::string &name, const OptimTensor &tensor);

  /// Appends a :cpp:enumerator:`ShapeEventAction::kConstraint` /
  /// :cpp:enumerator:`ShapeEventAction::kConstraintMax` event recording
  /// the two operands of a newly inserted constraint in ``inputs``. Only
  /// called by :cpp:func:`AddConstraint` /
  /// :cpp:func:`AddLessEqualConstraint` when event logging is enabled.
  void LogConstraintEvent(ShapeEventAction action, const std::string &lhs, const std::string &rhs);

  std::unordered_map<std::string, OptimTensor> tensors_;
  std::unordered_map<std::string, OptimSequence> sequences_;
  std::unordered_map<std::string, int> opsets_;
  std::unordered_map<std::string, const FunctionProto *> local_functions_;
  CustomShapeInferenceMap custom_shape_inference_;
  std::set<Constraint> constraints_;
  std::set<LessEqualConstraint> le_constraints_;
  ShapeEventLog events_;
  bool events_enabled_ = false;
};

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
