// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/raw_buffer_allocator.h"
#include "onnx_kernels/simple_map.h"
#include "onnx_kernels/simple_sequence.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_light_helpers.h"
#include "onnx_proto/onnx.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @file runtime_context.h
 * @brief Per-invocation runtime state shared across the nodes of a
 *        graph evaluated through :cpp:func:`RunNode` /
 *        :cpp:func:`RunNodes`.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

/**
 * Name-keyed map of tensors carrying both the graph inputs/initializers
 * and the intermediate values produced by previously executed nodes.
 * Owned by :cpp:class:`RuntimeContext`; the dispatcher reads a node's
 * inputs from this map by name (matching ``node.input(i)``) and inserts
 * every produced output under the name declared by ``node.output(i)``.
 */
using TensorMap = std::unordered_map<std::string, Tensor>;

/**
 * Name-keyed map of sequences carrying the sequence-typed graph
 * values produced or consumed by sequence operators
 * (``SequenceConstruct``, ``SequenceEmpty``, ``SequenceInsert``,
 * ``SequenceErase``, ``SequenceAt``, ``SequenceLength``,
 * ``ConcatFromSequence``, ``SplitToSequence``, ``SequenceMap``).
 *
 * Sequences are stored separately from tensors because their runtime
 * representation (:cpp:struct:`Sequence`) is a list of tensors and not
 * a single tensor: the dispatcher therefore keeps a sibling map of
 * sequence-typed edges, looked up by the same ``NodeProto::input`` /
 * ``NodeProto::output`` names.
 */
using SequenceMap = std::unordered_map<std::string, Sequence>;

/**
 * Name-keyed map of maps carrying the map-typed graph values produced or
 * consumed by map operators (``CastMap``, ``DictVectorizer``, ``ZipMap``).
 *
 * Maps are stored separately from tensors because their runtime
 * representation (:cpp:struct:`Map`) is a keys+values tensor pair and not
 * a single tensor: the dispatcher therefore keeps a sibling map of
 * map-typed edges, looked up by the same ``NodeProto::input`` /
 * ``NodeProto::output`` names.
 */
using OnnxMapMap = std::unordered_map<std::string, Map>;

/**
 * Name-keyed map of model-local :cpp:type:`FunctionProto` definitions
 * known to the runtime. Populated by :cpp:func:`RunModel` from
 * ``ModelProto::functions()`` so the dispatcher in :cpp:func:`RunNode`
 * can transparently invoke :cpp:func:`RunFunction` whenever a node
 * references a model-local function instead of a built-in kernel.
 *
 * Keys are the canonical ``"<domain>:<op_type>:<overload>"`` triple
 * (the default ONNX domain — the empty ``NodeProto::domain()`` — is
 * normalised to ``"ai.onnx"`` and the overload defaults to the empty
 * string). Values are non-owning pointers into the caller-owned
 * ``ModelProto``; the entries are valid only as long as the model
 * outlives the runtime context.
 */
using FunctionMap = std::unordered_map<std::string, const FunctionProto *>;

/**
 * Signature of a user-provided custom kernel callback. Mirrors
 * :cpp:type:`onnx_kernels::NodeKernelFn`: implementations read their
 * inputs from ``rt.tensors()`` (or ``rt.sequences()``) by name and
 * insert produced outputs under the names declared by ``node.output(i)``.
 *
 * Custom kernels are looked up by the canonical
 * ``"<domain>:<op_type>"`` key (the default ONNX domain — the empty
 * ``NodeProto::domain()`` — is normalised to ``"ai.onnx"``).
 */
using CustomKernelFn = std::function<void(const NodeProto &, class RuntimeContext &)>;

/**
 * Name-keyed map of user-provided custom kernels consulted by
 * :cpp:func:`RunNode` before the built-in
 * :cpp:func:`KernelDispatchTable`. Allows callers to extend the
 * runtime with operators implemented either in C++ (any callable
 * compatible with :cpp:type:`CustomKernelFn`) or in Python (through
 * the ``RuntimeContext.register_custom_kernel`` binding) without
 * touching the static dispatch table. Keys are
 * ``"<domain>:<op_type>"``; a custom registration overrides any
 * built-in entry with the same key.
 */
using CustomKernelMap = std::unordered_map<std::string, CustomKernelFn>;

/**
 * Maximum number of element values captured inline by
 * :cpp:class:`RuntimeEvent`. The event always carries a fixed-size buffer
 * of ``kRuntimeEventValueLimit`` entries; for tensors with more elements
 * the buffer holds only the first ``kRuntimeEventValueLimit`` values
 * (the remainder is truncated). When the element count exceeds the
 * limit the event's ``data_type`` is also set to ``-1`` to signal the
 * truncation, and ``shape`` is left empty so the log stays bounded for
 * large activations.
 */
inline constexpr int64_t kRuntimeEventValueLimit = 8;

/**
 * Kind of tensor map mutation recorded in the :cpp:class:`RuntimeContext`
 * event log.
 *
 *  * ``kAdd``     — a new entry was inserted (e.g. via :cpp:func:`RuntimeContext::Set`
 *                   or :cpp:func:`RuntimeContext::Put` on a previously absent name).
 *  * ``kReplace`` — an existing entry was overwritten via
 *                   :cpp:func:`RuntimeContext::Put`.
 *  * ``kRemove``  — an entry was erased via :cpp:func:`RuntimeContext::Remove`.
 *  * ``kRunNode`` — a kernel was dispatched for a single
 *                   :cpp:class:`NodeProto`. The event records the node's
 *                   ``op_domain`` / ``op_type``, the list of ``inputs``
 *                   it consumed, and the wall-clock ``duration_ns`` of
 *                   the dispatch (start time stored in ``timestamp_ns``).
 *                   Does not mutate the tensor map by itself.
 */
enum class RuntimeEventAction : int32_t { kAdd = 0, kReplace = 1, kRemove = 2, kRunNode = 3 };

/**
 * Role of the tensor at the moment the event was recorded. Set by the
 * call site that performs the mutation; not derived from the tensor map
 * itself.
 *
 *  * ``kUnknown``      — origin not specified.
 *  * ``kInitializer``  — a graph initializer seeded by :cpp:func:`RunGraph`.
 *  * ``kInput``        — a graph / function / subgraph input binding, or a
 *                        value injected by the caller before running.
 *  * ``kIntermediate`` — an intermediate value produced by a node kernel.
 *  * ``kOutput``       — a subgraph / function output propagated back to
 *                        the caller's tensor map.
 */
enum class RuntimeEventKind : int32_t {
  kUnknown = 0,
  kInitializer = 1,
  kInput = 2,
  kIntermediate = 3,
  kOutput = 4,
};

/**
 * Returns a short lowercase label for ``action`` (``"add"``, ``"replace"``,
 * ``"remove"``, ``"run_node"``). Useful for human-readable rendering of the
 * event log.
 */
inline constexpr const char *RuntimeEventActionName(RuntimeEventAction action) noexcept {
  switch (action) {
  case RuntimeEventAction::kAdd:
    return "add";
  case RuntimeEventAction::kReplace:
    return "replace";
  case RuntimeEventAction::kRemove:
    return "remove";
  case RuntimeEventAction::kRunNode:
    return "run_node";
  }
  return "unknown";
}

/**
 * Returns a short lowercase label for ``kind`` (``"unknown"``,
 * ``"initializer"``, ``"input"``, ``"intermediate"``, ``"output"``).
 */
inline constexpr const char *RuntimeEventKindName(RuntimeEventKind kind) noexcept {
  switch (kind) {
  case RuntimeEventKind::kUnknown:
    return "unknown";
  case RuntimeEventKind::kInitializer:
    return "initializer";
  case RuntimeEventKind::kInput:
    return "input";
  case RuntimeEventKind::kIntermediate:
    return "intermediate";
  case RuntimeEventKind::kOutput:
    return "output";
  }
  return "unknown";
}

/**
 * Single entry of the :cpp:class:`RuntimeContext` event log.
 *
 * Each mutation of the underlying ``TensorMap`` performed through
 * :cpp:func:`RuntimeContext::Set`, :cpp:func:`RuntimeContext::Put` or
 * :cpp:func:`RuntimeContext::Remove` produces one ``RuntimeEvent`` capturing
 * the action, the role (``kind``), the name of the tensor, the wall-clock
 * timestamp (nanoseconds since the Unix epoch), and a snapshot of the
 * tensor's type and shape.
 *
 * The element values are captured into a fixed-size buffer of
 * :cpp:var:`kRuntimeEventValueLimit` entries (``values`` for numeric dtypes,
 * ``string_values`` for ``DataType::STRING``); ``value_count`` records how
 * many slots are populated (``min(element_count, kRuntimeEventValueLimit)``).
 * When the tensor has more than :cpp:var:`kRuntimeEventValueLimit` elements
 * the buffer holds only the first ``kRuntimeEventValueLimit`` values
 * (the remainder is truncated), ``data_type`` is set to ``-1`` to signal
 * the truncation and ``shape`` is left empty.
 *
 * ``kRemove`` events always set ``data_type = DataType::UNDEFINED``,
 * ``value_count = 0``, leave ``shape`` empty and do not populate
 * ``values`` / ``string_values``; they only record the name, kind and
 * timestamp of the removal.
 */
struct RuntimeEvent {
  /// Kind of mutation recorded by this entry.
  RuntimeEventAction action = RuntimeEventAction::kAdd;
  /// Role of the tensor at the moment of the event (see
  /// :cpp:enum:`RuntimeEventKind`).
  RuntimeEventKind kind = RuntimeEventKind::kUnknown;
  /// Wall-clock timestamp of the event, in nanoseconds since the Unix
  /// epoch (``std::chrono::system_clock``).
  int64_t timestamp_ns = 0;
  /// Name under which the tensor is (or was) stored in the
  /// :cpp:class:`RuntimeContext` tensor map.
  std::string name;
  /// Element data type of the tensor at the moment of the event, encoded
  /// as a ``TensorProto::DataType`` integer value. Set to
  /// ``DataType::UNDEFINED`` for ``kRemove`` events, and to ``-1`` for
  /// ``kAdd`` / ``kReplace`` events whose tensor has more than
  /// :cpp:var:`kRuntimeEventValueLimit` elements (the values buffer is
  /// then truncated to the first ``kRuntimeEventValueLimit`` entries and
  /// ``shape`` is left empty).
  int32_t data_type = 0;
  /// Tensor shape at the moment of the event. Empty for ``kRemove``, for
  /// scalar tensors (``element_count == 1``), and for ``kAdd`` /
  /// ``kReplace`` events whose tensor exceeds
  /// :cpp:var:`kRuntimeEventValueLimit` elements (truncated payload).
  std::vector<int64_t> shape;
  /// Number of populated entries in ``values`` / ``string_values``
  /// (``min(element_count, kRuntimeEventValueLimit)``). Zero for
  /// ``kRemove`` events.
  int32_t value_count = 0;
  /// Fixed-size buffer holding the first ``value_count`` numeric values
  /// of the tensor (coerced to ``double``). Boolean values are recorded
  /// as ``0.0`` / ``1.0``. Unused slots are zero-initialised. Always
  /// empty for ``DataType::STRING`` and ``kRemove`` events.
  std::array<double, kRuntimeEventValueLimit> values{};
  /// Fixed-size buffer holding the first ``value_count`` string values
  /// of the tensor when ``data_type`` is ``DataType::STRING``. Unused
  /// slots are empty strings.
  std::array<std::string, kRuntimeEventValueLimit> string_values{};
  /// For ``kRunNode`` events: ONNX op domain of the node that was
  /// dispatched, normalised so the default domain is reported as
  /// ``"ai.onnx"``. Empty for all other event actions.
  std::string op_domain;
  /// For ``kRunNode`` events: ONNX ``op_type`` of the node that was
  /// dispatched. Empty for all other event actions.
  std::string op_type;
  /// For ``kRunNode`` events: ordered list of input names consumed by
  /// the node, matching ``NodeProto::input``. Empty for all other event
  /// actions.
  std::vector<std::string> inputs;
  /// For ``kRunNode`` events: wall-clock duration of the kernel
  /// dispatch in nanoseconds (``std::chrono::steady_clock``). Zero for
  /// all other event actions.
  int64_t duration_ns = 0;
  /// Index of the node this event is associated with. For
  /// :cpp:enumerator:`RuntimeEventKind::kInput` values it is ``-1`` and for
  /// :cpp:enumerator:`RuntimeEventKind::kInitializer` values it is ``-2``.
  /// For intermediate / output tensors and for ``kRunNode`` events it is the
  /// position (``>= 0``) of the producing / dispatched node in its graph (or
  /// function / subgraph) node list. ``-1`` when no producing node is known.
  int64_t node_index = -1;
  /// Device the tensor lives on at the moment of the event: ``-1`` for the
  /// CPU and ``0``–``8192`` for a GPU device index. The CPU reference runtime
  /// always reports ``-1``.
  int32_t device = -1;
  /// Index of the control-flow node in the **parent** graph whose attribute
  /// subgraph produced this event. ``-1`` for events from the top-level graph.
  /// Combined with :cpp:var:`subgraph_attr_name` this uniquely identifies
  /// which operator and which attribute subgraph an event originated from.
  int64_t subgraph_node_index = -1;
  /// Attribute name of the subgraph within the control-flow node identified
  /// by :cpp:var:`subgraph_node_index`: ``"body"`` for :onnx:`Loop` /
  /// :onnx:`Scan` / :onnx:`SequenceMap`, ``"then_branch"`` or
  /// ``"else_branch"`` for :onnx:`If`. Empty for top-level-graph events.
  std::string subgraph_attr_name;
};

/**
 * Append-only log of tensor map mutations recorded by
 * :cpp:class:`RuntimeContext`.
 */
using RuntimeEventLog = std::vector<RuntimeEvent>;

// Forward declaration so ExecutionPlan::ReleaseAfter can reference
// RuntimeContext by reference without a full definition at this point.
class RuntimeContext;

/**
 * Precomputed per-graph release schedule used by
 * :cpp:func:`RunGraph` / :cpp:func:`RunFunction` /
 * :cpp:func:`RunNodes` when
 * :cpp:func:`RuntimeContext::release_intermediates` is enabled.
 *
 * An :cpp:class:`ExecutionPlan` captures two complementary pieces of
 * information for a given node sequence:
 *
 *  * ``keep`` — the *structural* set of names that must never be
 *    released by the per-node release loop. For a :cpp:class:`GraphProto`
 *    this is the union of declared inputs, initializers, and declared
 *    outputs; for a :cpp:class:`FunctionProto` it is the union of
 *    declared inputs and outputs.
 *  * ``releasable[i]`` — the list of names whose last reference (per
 *    :cpp:func:`RuntimeContext::CollectNodeInputs`) falls at node ``i``
 *    and that are not in ``keep`` — i.e. the intermediates that may be
 *    removed from the tensor / sequence map right after node ``i``
 *    finishes.
 *
 * The analysis depends only on the graph topology and not on any
 * runtime value, so a single plan can be reused across every
 * invocation of the same model. :cpp:func:`RuntimeContext::GetExecutionPlan`
 * builds and caches one plan per graph / function for that reason.
 */
class ExecutionPlan {
public:
  ExecutionPlan() = default;

  /// Builds the plan for ``graph``. ``keep`` is seeded with the graph's
  /// declared inputs, initializers and declared outputs; ``releasable``
  /// is computed by :cpp:func:`RuntimeContext::ComputeReleasableInputs`.
  explicit ExecutionPlan(const GraphProto &graph);

  /// Builds the plan for ``func``. ``keep`` is seeded with the
  /// function's declared inputs and outputs.
  explicit ExecutionPlan(const FunctionProto &func);

  /// Builds the plan for a free-standing node range. ``keep`` is the
  /// user-supplied set of names that must never be released (typically
  /// the names already populated in the runtime context at run start
  /// plus any graph / function outputs).
  ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                std::unordered_set<std::string> keep);

  /// Structural set of names that must never be released. See the
  /// class-level documentation for the exact contents.
  const std::unordered_set<std::string> &keep() const noexcept { return keep_; }

  /// For each node ``i`` in the underlying node range, the list of
  /// names whose last reference falls at ``i`` and that are not in
  /// :cpp:func:`keep`.
  const std::vector<std::vector<std::string>> &releasable() const noexcept { return releasable_; }

  /// Number of nodes covered by this plan (``releasable().size()``).
  size_t num_nodes() const noexcept { return releasable_.size(); }

  /// Releases from ``rt`` every name in the ``releasable()`` slot
  /// associated with ``node``. ``node`` must be one of the
  /// :cpp:class:`NodeProto` instances the plan was built from
  /// (lookup is by address); if it is not, this is a no-op. Each
  /// removal is performed on both the tensor map and the sequence
  /// map: :cpp:func:`RuntimeContext::Remove` is a no-op if the name
  /// is absent and emits a :cpp:enumerator:`RuntimeEventAction::kRemove`
  /// event when event logging is on; sequence removals do not emit
  /// events (sequence values live outside the tensor event stream).
  void ReleaseAfter(const NodeProto &node, RuntimeContext &rt) const;

private:
  std::unordered_set<std::string> keep_;
  std::vector<std::vector<std::string>> releasable_;
  std::unordered_map<const NodeProto *, size_t> node_index_;
};

/**
 * Per-invocation runtime state passed to :cpp:func:`RunNode` /
 * :cpp:func:`RunNodes`.
 *
 * Bundles together everything a chain of nodes needs to execute:
 *  * a :cpp:type:`TensorMap` carrying the graph inputs / initializers
 *    and every intermediate value produced by previously executed
 *    nodes (accessed through :cpp:func:`tensors`);
 *  * the construction-time :cpp:class:`kernel::KernelContext` (opset
 *    and any future construction-time inputs) used to instantiate
 *    each per-operator kernel (accessed through :cpp:func:`kernel_ctx`).
 *
 * Grouping them in a single object keeps the dispatcher signatures
 * stable as more per-invocation state (allocators, device descriptors,
 * profiling hooks, …) is added in the future without forcing every
 * trampoline or call site to take an extra argument.
 *
 * Convenience accessors (:cpp:func:`Set`, :cpp:func:`Get`,
 * :cpp:func:`Has`, :cpp:func:`Remove`) wrap the underlying map so
 * callers do not have to reach for ``rt.tensors()[name]`` directly.
 */
class RuntimeContext {
public:
  RuntimeContext() = default;
  ~RuntimeContext();
  explicit RuntimeContext(kernel::KernelContext kernel_ctx) : kernel_ctx_(std::move(kernel_ctx)) {}
  RuntimeContext(kernel::KernelContext kernel_ctx, TensorMap tensors)
      : tensors_(std::move(tensors)), kernel_ctx_(std::move(kernel_ctx)) {}

  /// Enables or disables event logging. When disabled (the default),
  /// :cpp:func:`Set`, :cpp:func:`Put`, :cpp:func:`Remove` and
  /// :cpp:func:`RunNode` skip all event construction, clock reads, and
  /// value decoding — eliminating the profiling overhead from the hot
  /// path. Call ``set_events_enabled(true)`` before running if per-node
  /// profiling is required.
  void set_events_enabled(bool enabled) noexcept { events_enabled_ = enabled; }
  bool events_enabled() const noexcept { return events_enabled_; }

  /// Verbosity level used by :cpp:func:`RunNode` to print execution
  /// progress to ``stdout`` while the graph is running. ``0`` disables
  /// printing.
  void set_verbose(int verbose) noexcept { verbose_ = verbose; }
  int verbose() const noexcept { return verbose_; }

  /// Index of the control-flow node in the parent graph currently being
  /// executed. Set before running a subgraph so that events recorded inside
  /// carry :cpp:var:`RuntimeEvent::subgraph_node_index` and
  /// :cpp:var:`RuntimeEvent::subgraph_attr_name`. ``-1`` for the top-level
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

  /// Index of the node currently being executed, used to tag the
  /// :cpp:var:`RuntimeEvent::node_index` of intermediate / output tensors
  /// produced during its dispatch. Set by :cpp:func:`RunNodes` before each
  /// :cpp:func:`RunNode` call and ``-1`` when no node is executing.
  void set_current_node_index(int64_t index) noexcept { current_node_index_ = index; }
  int64_t current_node_index() const noexcept { return current_node_index_; }

  /// In/out tensor map shared across every node in a chain.
  TensorMap &tensors() noexcept { return tensors_; }
  const TensorMap &tensors() const noexcept { return tensors_; }

  /// Kernel construction context (opset).
  kernel::KernelContext &kernel_ctx() noexcept {
    // Keep the construction-time context in sync with the currently attached
    // allocator so kernels built via ``rt.kernel_ctx()`` route their result
    // storage through it.
    kernel_ctx_.allocator = allocator_;
    return kernel_ctx_;
  }
  const kernel::KernelContext &kernel_ctx() const noexcept { return kernel_ctx_; }

  /// Optional allocator used by kernels to acquire and release
  /// :cpp:struct:`RawBuffer` instances. ``nullptr`` when no allocator has
  /// been attached (the default). The caller retains ownership; the lifetime
  /// of the allocator must exceed the lifetime of this context.
  void set_allocator(RawBufferAllocator *allocator) noexcept { allocator_ = allocator; }
  RawBufferAllocator *allocator() noexcept { return allocator_; }
  const RawBufferAllocator *allocator() const noexcept { return allocator_; }

  /// Model-local function registry consulted by :cpp:func:`RunNode`
  /// before falling back to the built-in kernel dispatch table.
  FunctionMap &functions() noexcept { return functions_; }
  const FunctionMap &functions() const noexcept { return functions_; }

  /// User-provided custom kernel registry consulted by
  /// :cpp:func:`RunNode` before the built-in :cpp:func:`KernelDispatchTable`.
  /// Keys are the canonical ``"<domain>:<op_type>"`` pair (the default
  /// ONNX domain — the empty ``NodeProto::domain()`` — is normalised
  /// to ``"ai.onnx"``). A custom registration overrides any built-in
  /// entry with the same key, but model-local functions and the
  /// built-in control-flow operators (``If``, ``Loop``, ``Scan``,
  /// ``SequenceMap``) still take precedence.
  CustomKernelMap &custom_kernels() noexcept { return custom_kernels_; }
  const CustomKernelMap &custom_kernels() const noexcept { return custom_kernels_; }

  /// Registers or replaces a custom kernel for ``(domain, op_type)``.
  /// The empty domain is normalized to ``"ai.onnx"``.
  void RegisterCustomKernel(const std::string &domain, const std::string &op_type,
                            CustomKernelFn fn) {
    const std::string d = domain.empty() ? std::string("ai.onnx") : domain;
    custom_kernels_[d + ":" + op_type] = std::move(fn);
  }

  /// Removes the custom kernel registered for ``(domain, op_type)``.
  /// The empty domain is normalised to ``"ai.onnx"``.
  /// Returns ``true`` when an entry was removed, ``false`` otherwise.
  bool UnregisterCustomKernel(const std::string &domain, const std::string &op_type) {
    const std::string d = domain.empty() ? std::string("ai.onnx") : domain;
    return custom_kernels_.erase(d + ":" + op_type) > 0;
  }

  /// Removes every registered custom kernel.
  void ClearCustomKernels() { custom_kernels_.clear(); }

  /// Returns ``true`` if a tensor named ``name`` is currently held.
  bool Has(const std::string &name) const { return tensors_.find(name) != tensors_.end(); }

  /// Removes the tensor stored under ``name`` if present. Returns
  /// ``true`` if an entry was erased, ``false`` otherwise. When an entry
  /// is erased a :cpp:class:`RuntimeEvent` with action
  /// :cpp:enumerator:`RuntimeEventAction::kRemove` is appended to the
  /// event log; nothing is logged when ``name`` is not present.
  bool Remove(const std::string &name);

  /// Inserts the tensor under ``name``. The name must not already
  /// be present in the map; use :cpp:func:`Put` (or ``tensors()``
  /// directly) to overwrite. A :cpp:class:`RuntimeEvent` with action
  /// :cpp:enumerator:`RuntimeEventAction::kAdd` and the supplied ``kind``
  /// is appended to the event log on successful insertion. ``kind``
  /// defaults to :cpp:enumerator:`RuntimeEventKind::kInput`, which is
  /// the typical role of values seeded by the caller before running.
  void Set(const std::string &name, Tensor tensor,
           RuntimeEventKind kind = RuntimeEventKind::kInput);

  /// Inserts or overwrites the tensor stored under ``name``. Appends a
  /// :cpp:class:`RuntimeEvent` describing the new state with action
  /// :cpp:enumerator:`RuntimeEventAction::kAdd` when ``name`` was absent
  /// and :cpp:enumerator:`RuntimeEventAction::kReplace` when an existing
  /// entry was overwritten. ``kind`` defaults to
  /// :cpp:enumerator:`RuntimeEventKind::kIntermediate`, the typical role
  /// of values written by node kernels through :cpp:func:`SetOutput`.
  void Put(const std::string &name, Tensor tensor,
           RuntimeEventKind kind = RuntimeEventKind::kIntermediate);

  /**
   * Returns the tensor stored under ``name``.
   *
   * @throws std::out_of_range if ``name`` is not in the map.
   */
  const Tensor &Get(const std::string &name) const;
  Tensor &Get(const std::string &name);

  /// Append-only log of every tensor map mutation performed through
  /// :cpp:func:`Set`, :cpp:func:`Put` and :cpp:func:`Remove`. See
  /// :cpp:class:`RuntimeEvent` for the captured fields.
  const RuntimeEventLog &events() const noexcept { return events_; }
  RuntimeEventLog &events() noexcept { return events_; }

  /// Empties the event log without otherwise touching the tensor map.
  void ClearEvents() noexcept { events_.clear(); }

  /// Creates a fresh child context for executing a subgraph (e.g. the
  /// ``then_branch`` or ``else_branch`` of ``If``, or the ``body`` of
  /// ``Loop`` / ``Scan``). The child context inherits the parent's
  /// kernel context, allocator, function registry, tensor map,
  /// sequence map, verbosity and event-logging flag so outer-scope
  /// values are visible inside the subgraph. :cpp:func:`current_subgraph`
  /// is set to ``(current_node_index(), attr_name)`` on the child.
  /// The subgraph's writes remain local and do not pollute this context.
  ///
  /// Returns:
  ///   A new :cpp:class:`RuntimeContext` initialised for subgraph execution.
  RuntimeContext MakeSubgraphContext(const std::string &attr_name) const;

  /// Creates a fresh child context for executing a model-local function.
  /// The child inherits the parent's kernel context, allocator, function
  /// registry and verbosity, but starts with an empty tensor and sequence
  /// map so the function's formal inputs are bound explicitly by the
  /// caller.
  ///
  /// Returns:
  ///   A new :cpp:class:`RuntimeContext` initialised for function execution.
  RuntimeContext MakeFunctionContext() const;

  /// Resets the per-invocation state so the context can be reused for a
  /// fresh run: clears the tensor map, the sequence map and the event
  /// log, and resets :cpp:func:`current_node_index` to ``-1``. The kernel
  /// context, registered model-local functions and custom kernels, the
  /// cached :cpp:class:`ExecutionPlan` instances and the
  /// :cpp:func:`events_enabled` / :cpp:func:`release_intermediates`
  /// settings are intentionally preserved, so the execution-plan cache
  /// is amortised across repeated runs of the same model.
  void Clear() noexcept {
    tensors_.clear();
    sequences_.clear();
    maps_.clear();
    events_.clear();
    current_node_index_ = -1;
  }

  /// Appends a :cpp:class:`RuntimeEvent` with action
  /// :cpp:enumerator:`RuntimeEventAction::kRunNode` summarising the
  /// dispatch of a single ``NodeProto``. ``timestamp_ns`` is set to the
  /// wall-clock time at which the dispatch started and ``duration_ns``
  /// to its measured wall-clock duration in nanoseconds. Inserted by
  /// :cpp:func:`RunNode` for every kernel call so callers can profile
  /// per-node execution from the event log alongside the tensor
  /// add/replace/remove records.
  void AppendRunNodeEvent(const std::string &op_domain, const std::string &op_type,
                          std::vector<std::string> inputs, int64_t start_time_ns,
                          int64_t duration_ns);

  /**
   * Returns the list of input names referenced by ``nodes`` that are
   * not produced as outputs by any node in the same list — i.e. the
   * external dependencies of the node set.
   *
   * Subgraph attributes (``GRAPH`` / ``GRAPHS``) are inspected
   * recursively: for every subgraph, names read by the subgraph's
   * nodes that are neither produced inside the subgraph (formal
   * inputs, initializers, intermediate node outputs) nor produced by
   * the outer ``nodes`` are appended to the result, mirroring the
   * captured-value semantics of ONNX control-flow operators.
   *
   * The returned list preserves the order in which each external
   * name is first encountered and contains no duplicates. Empty
   * input names (optional inputs left unbound) are skipped.
   */
  static std::vector<std::string>
  CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes);

  /// ``std::vector``-overload of :cpp:func:`CollectExternalInputs`.
  static std::vector<std::string> CollectExternalInputs(const std::vector<NodeProto> &nodes);

  /**
   * Returns the full list of tensor / sequence names a single ``node``
   * depends on at runtime.
   *
   * The result is the union of:
   *  * the names referenced by ``node.input()`` (skipping empty
   *    optional-input slots), and
   *  * every external input of the subgraph attributes
   *    (``GRAPH`` / ``GRAPHS``) attached to ``node`` — i.e. the
   *    captured names a subgraph reads from the enclosing scope, as
   *    computed recursively by :cpp:func:`CollectExternalInputs`.
   *
   * The returned list preserves the order in which each name is first
   * encountered and contains no duplicates.
   */
  static std::vector<std::string> CollectNodeInputs(const NodeProto &node);

  /**
   * Returns, for each node in ``nodes``, the list of input names that
   * become unused once that node has finished executing — i.e. names
   * whose last reference (per :cpp:func:`CollectNodeInputs`) appears at
   * that node and that do not appear in ``keep``.
   *
   * Empty names (optional inputs left unbound) are skipped. The
   * returned vector has exactly ``nodes.size()`` entries; each inner
   * vector preserves the order in which the corresponding names were
   * first encountered in the input list of the producing/consuming
   * node.
   */
  static std::vector<std::vector<std::string>>
  ComputeReleasableInputs(const utils::RepeatedProtoField<NodeProto> &nodes,
                          const std::unordered_set<std::string> &keep);

  /// ``std::vector``-overload of :cpp:func:`ComputeReleasableInputs`.
  static std::vector<std::vector<std::string>>
  ComputeReleasableInputs(const std::vector<NodeProto> &nodes,
                          const std::unordered_set<std::string> &keep);

  /// Returns the cached :cpp:class:`ExecutionPlan` for ``graph``,
  /// building it on first use. The plan precomputes, for every node in
  /// ``graph``, the list of input names whose last reference falls at
  /// that node and that are not declared inputs / initializers /
  /// outputs of ``graph`` — i.e. the intermediates that may be removed
  /// from this context as soon as the node has finished executing.
  /// The plan is keyed by the address of ``graph`` and reused across
  /// subsequent runs of the same model, so the analysis is paid only
  /// once for the lifetime of this :cpp:class:`RuntimeContext`.
  const ExecutionPlan &GetExecutionPlan(const GraphProto &graph);

  /// Returns the cached :cpp:class:`ExecutionPlan` for ``func``,
  /// building it on first use. Same caching semantics as the
  /// :cpp:class:`GraphProto` overload — the structural keep set
  /// consists of the function's declared inputs and outputs.
  const ExecutionPlan &GetExecutionPlan(const FunctionProto &func);

  /// Clears every cached :cpp:class:`ExecutionPlan`. Useful when the
  /// owning model has been mutated in place (rare).
  void ClearExecutionPlans() noexcept;

  /// Enables or disables the per-node release of unused intermediates
  /// performed by :cpp:func:`RunNodes` / :cpp:func:`RunGraph` /
  /// :cpp:func:`RunFunction` / :cpp:func:`RunModel`. When enabled, a
  /// name whose last reference (declared input of a node, or captured
  /// input of a subgraph attribute) appears at node ``i`` is removed
  /// from :cpp:func:`tensors` (and :cpp:func:`sequences`) right after
  /// node ``i`` finishes — emitting a
  /// :cpp:enumerator:`RuntimeEventAction::kRemove` event when event
  /// logging is on. Graph / function outputs are always preserved.
  /// Disabled by default to keep intermediate values observable after
  /// the run (e.g. so callers can fetch any node output by name).
  void set_release_intermediates(bool enabled) noexcept { release_intermediates_ = enabled; }
  bool release_intermediates() const noexcept { return release_intermediates_; }

  /// In/out sequence map shared across every node in a chain. Only
  /// sequence-typed graph edges are stored here; tensor-typed edges
  /// live in :cpp:func:`tensors`.
  SequenceMap &sequences() noexcept { return sequences_; }
  const SequenceMap &sequences() const noexcept { return sequences_; }

  /// Returns ``true`` if a sequence named ``name`` is currently held.
  bool HasSequence(const std::string &name) const {
    return sequences_.find(name) != sequences_.end();
  }

  /// Inserts or overwrites the sequence stored under ``name``. The
  /// stored sequence's ``name`` field is updated to ``name``. No event
  /// is appended to the event log: sequence values are intentionally
  /// outside the tensor event stream.
  void PutSequence(const std::string &name, Sequence sequence) {
    sequence.name = name;
    sequences_[name] = std::move(sequence);
  }

  /// Removes the sequence stored under ``name`` if present. Returns
  /// ``true`` if an entry was erased, ``false`` otherwise.
  bool RemoveSequence(const std::string &name) { return sequences_.erase(name) > 0; }

  /**
   * Returns the sequence stored under ``name``.
   *
   * @throws std::out_of_range if ``name`` is not in the sequence map.
   */
  const Sequence &GetSequence(const std::string &name) const {
    auto it = sequences_.find(name);
    if (it == sequences_.end()) {
      throw std::out_of_range("RuntimeContext::GetSequence: no sequence named '" + name + "'.");
    }
    return it->second;
  }

  /// In/out map store shared across every node in a chain. Only
  /// map-typed graph edges are stored here; tensor-typed edges
  /// live in :cpp:func:`tensors`.
  OnnxMapMap &maps() noexcept { return maps_; }
  const OnnxMapMap &maps() const noexcept { return maps_; }

  /// Returns ``true`` if a map named ``name`` is currently held.
  bool HasMap(const std::string &name) const { return maps_.find(name) != maps_.end(); }

  /// Inserts or overwrites the map stored under ``name``.
  void PutMap(const std::string &name, Map map) {
    map.name = name;
    maps_[name] = std::move(map);
  }

  /// Removes the map stored under ``name`` if present.
  bool RemoveMap(const std::string &name) { return maps_.erase(name) > 0; }

  /**
   * Returns the map stored under ``name``.
   *
   * @throws std::out_of_range if ``name`` is not in the map store.
   */
  const Map &GetMap(const std::string &name) const {
    auto it = maps_.find(name);
    if (it == maps_.end()) {
      throw std::out_of_range("RuntimeContext::GetMap: no map named '" + name + "'.");
    }
    return it->second;
  }

private:
  TensorMap tensors_;
  kernel::KernelContext kernel_ctx_;
  FunctionMap functions_;
  CustomKernelMap custom_kernels_;
  RuntimeEventLog events_;
  SequenceMap sequences_;
  OnnxMapMap maps_;
  bool events_enabled_ = false;
  int verbose_ = 0;
  bool release_intermediates_ = false;
  int64_t current_node_index_ = -1;
  /// Index of the control-flow node in the parent graph currently being
  /// executed (see :cpp:func:`set_current_subgraph`). ``-1`` for the
  /// top-level graph.
  int64_t current_subgraph_node_index_ = -1;
  /// Attribute name of the subgraph currently being executed (see
  /// :cpp:func:`set_current_subgraph`). Empty for the top-level graph;
  /// set to ``"body"``, ``"then_branch"``, ``"else_branch"``, etc. when
  /// running a control-flow body subgraph.
  std::string current_subgraph_attr_name_;
  /// Lazily-populated cache of :cpp:class:`ExecutionPlan` instances
  /// keyed by the address of the :cpp:class:`GraphProto` /
  /// :cpp:class:`FunctionProto` they describe. Built on first use by
  /// :cpp:func:`GetExecutionPlan` and reused across subsequent runs of
  /// the same model.
  std::unordered_map<const void *, ExecutionPlan> execution_plans_;
  /// Optional allocator for :cpp:struct:`RawBuffer` instances. Non-owning;
  /// ``nullptr`` when no allocator has been attached.
  RawBufferAllocator *allocator_ = nullptr;
};

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
