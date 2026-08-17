// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_core/runtime/tuning/runtime_parameters.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @file runtime_session.h
 * @brief Declares :cpp:class:`RuntimeSession`, a reusable execution session
 *        that separates kernel initialization from execution.
 *
 * :cpp:class:`RuntimeSession` binds a precomputed :cpp:class:`ExecutionPlan`
 * (which already carries the node list it drives) and, on its first
 * :cpp:func:`Run`, resolves every executed node's kernel once against the
 * supplied :cpp:class:`RuntimeContext`; subsequent runs reuse the cached
 * kernels. Every entry point that runs a node list — model callers (via
 * :cpp:func:`RegisterModelFunctions` followed by their own
 * :cpp:class:`RuntimeSession`), :cpp:class:`SubgraphSession`, the model-local
 * function call helper, and the ``If`` / ``Loop`` / ``Scan`` control-flow
 * kernels — constructs one of these sessions (over the graph's or
 * function's cached :cpp:class:`ExecutionPlan`) and calls :cpp:func:`Run` a
 * single time.
 */

namespace ONNX_LIGHT_NAMESPACE::core {
namespace shapes {
// Forward declaration: VerifyDeclaredShape threads a ShapesContext (defined
// in onnx_core/shapes/shapes_context.h) through SymShape::FitsConcreteShape
// to bind symbolic dimensions to their concrete values during a run.
class ShapesContext;
} // namespace shapes
namespace runtime {

/**
 * Construction-time settings for :cpp:class:`RuntimeSession`.
 *
 * Sessions are intended to be built once and reused, so these values stay
 * fixed for the session's lifetime.
 */
struct RuntimeSessionOptions {
  RuntimeParameters parameters = {};
  int verbose = 0;
  bool check_shapes = false;
  /// When ``false`` (the default), :cpp:func:`RuntimeSession::Run` verifies
  /// that every allocator-backed output a node produces is owned by the
  /// session's unique allocator, rejecting a kernel that allocates its output
  /// from a different allocator (or from none at all). When ``true``, that
  /// check is skipped so a kernel may legitimately return an output allocated
  /// outside the common allocator.
  bool allow_external_output_allocators = false;
};

/** Reports the one-time kernel tuning work performed by a runtime session. */
struct KernelTuningResolutionStatistics {
  /// Time spent capturing the immutable registry generation.
  uint64_t snapshot_duration_ns = 0;
  /// Time spent resolving execution-specific profiles from that generation.
  uint64_t resolution_duration_ns = 0;
  /// Number of kernels that exposed a defined tuning key.
  size_t tunable_kernels = 0;
  /// Number of tunable kernels for which registered parameters were found.
  size_t resolved_profiles = 0;

  /// Returns the measured cold tuning duration.
  uint64_t TotalDurationNs() const noexcept {
    return snapshot_duration_ns + resolution_duration_ns;
  }

  bool operator==(const KernelTuningResolutionStatistics &) const = default;
};

/**
 * A reusable execution session that binds a precomputed
 * :cpp:class:`ExecutionPlan` and separates the runtime lifecycle into three
 * explicit phases:
 *
 *  1. **Construction** — the session records the ``plan`` it will drive. The
 *     node list is recovered from :cpp:func:`ExecutionPlan::nodes`, so the
 *     session no longer needs the nodes (nor a :cpp:class:`RuntimeContext`)
 *     passed separately.
 *  2. **Kernel initialization** — the first :cpp:func:`Run` resolves the
 *     kernel for every node the plan executes once (against the model-local
 *     function registry, the control-flow handlers, the user custom kernels
 *     and the static :cpp:func:`KernelDispatchTable` of the supplied
 *     :cpp:class:`RuntimeContext`), builds the resulting per-node kernel
 *     instance, and caches it. Any unsupported operator is rejected here
 *     rather than mid-run. It also records the external inputs the scheduled
 *     nodes read (:cpp:func:`required_inputs`) so :cpp:func:`Run` can verify
 *     they are supplied before executing.
 *  3. **Execution** — :cpp:func:`Run` replays the plan, invoking each
 *     pre-resolved kernel instance and releasing intermediates as scheduled. It
 *     may be called more than once (e.g. to re-run the same graph with fresh
 *     inputs) without redoing the per-node dispatch lookup or re-constructing
 *     the concrete per-node kernel objects.
 *
 * This mirrors how an inference runtime prepares an executable graph once and
 * then runs it repeatedly. Every caller that needs to run a node list builds
 * one of these sessions over the list's :cpp:class:`ExecutionPlan` and calls
 * :cpp:func:`Run` on it.
 */
class RuntimeSession {
public:
  /**
   * Builds a session over an :cpp:class:`ExecutionPlan` the session owns,
   * constructed from ``model``'s graph (:cpp:func:`ModelProto::graph`). Use
   * this when no precomputed plan is available: the session builds and owns
   * the plan itself, so a caller can create a runnable session from a model
   * alone (without first building an :cpp:class:`ExecutionPlan`). Kernel
   * resolution is still deferred to the first :cpp:func:`Run`.
   *
   * @param model Model whose graph drives execution. The model (and the graph
   *              it owns) must outlive the session, since the built plan holds
   *              non-owning pointers into the graph's nodes.
   * @param verbose Verbosity level used by :cpp:func:`Run` for its progress
   *                lines. ``0`` (the default) leaves verbosity to the
   *                :cpp:class:`RuntimeContext`.
   */
  explicit RuntimeSession(const ModelProto &model, int verbose = 0);
  RuntimeSession(const ModelProto &model, RuntimeSessionOptions options);

  /**
   * Builds a session over ``plan``. Kernel resolution is deferred to the first
   * :cpp:func:`Run` (which supplies the :cpp:class:`RuntimeContext` the
   * kernels are resolved against).
   *
   * @param plan Precomputed execution / release schedule. Its node list
   *             (:cpp:func:`ExecutionPlan::nodes`) drives execution. The plan
   *             (and the graph / function it was built from) must outlive the
   *             session.
   * @param verbose Verbosity level used by :cpp:func:`Run` for its progress
   *                lines. ``0`` (the default) leaves verbosity to the
   *                :cpp:class:`RuntimeContext`.
   */
  explicit RuntimeSession(const ExecutionPlan &plan, int verbose = 0);
  RuntimeSession(const ExecutionPlan &plan, RuntimeSessionOptions options);

  // A session caches one owning ``std::unique_ptr<KernelBase>`` per node (see
  // :cpp:member:`kernels_`), so it is move-only. It is always created in place
  // (by value inside :cpp:class:`SubgraphSession`, or via ``std::make_shared``)
  // and never copied; deleting the copy operations keeps that explicit and lets
  // the Python binding treat it as a non-copyable type.
  RuntimeSession(const RuntimeSession &) = delete;
  RuntimeSession &operator=(const RuntimeSession &) = delete;

  /**
   * Executes the plan once against ``rt``: on the first call it resolves and
   * caches the kernel for every scheduled node (rejecting unsupported
   * operators) and records the external inputs those nodes read; every call
   * then verifies ``rt`` supplies each of those required inputs and runs each
   * scheduled node using its resolved kernel. When
   * :cpp:func:`RuntimeContext::release_intermediates` is enabled on ``rt``,
   * it additionally frees each intermediate whose last reference has been
   * reached, as scheduled by the plan; when disabled, every intermediate the
   * plan would have released instead stays observable in ``rt`` after
   * ``Run`` returns. Safe to call repeatedly on the same session.
   *
   * The allocator attached to ``rt`` (:cpp:func:`RuntimeContext::allocator`)
   * is captured once, on the first call, as the session's unique allocator.
   * After each scheduled node executes, every tensor it produced that is
   * allocator-backed (:cpp:func:`Tensor::has_allocation`) is verified to be
   * owned by that same allocator, catching kernels that allocate their output
   * from the wrong allocator (or from none at all). This verification is
   * skipped when :cpp:func:`allow_external_output_allocators` is enabled, so a
   * kernel may legitimately return an output allocated outside the common
   * allocator.
   *
   * @param rt In/out runtime context used both to resolve the kernels
   *           (function registry / custom kernels) and to exchange tensors.
   *
   * @throws std::invalid_argument if the plan references an out-of-range node
   *         index, if any executed node cannot be dispatched, if ``rt`` does
   *         not define one of the plan's required external inputs, or if a
   *         node produces an output tensor backed by an allocator other than
   *         the session's.
   */
  void Run(RuntimeContext &rt);

  /// Model-independent execution parameters (e.g. the requested degree of
  /// parallelism, :cpp:var:`RuntimeParameters::num_threads`) applied to the
  /// nodes this session runs.
  const RuntimeParameters &parameters() const noexcept { return parameters_; }

  /// Verbosity level requested for :cpp:func:`Run`. When non-zero, it overrides
  /// :cpp:func:`RuntimeContext::verbose` for this session's progress lines
  /// without mutating the context itself.
  int verbose() const noexcept { return verbose_; }

  /// Returns the external input names the scheduled nodes read (the inputs that
  /// must be present in the :cpp:class:`RuntimeContext` before :cpp:func:`Run`).
  /// Populated during kernel initialization; empty until the first
  /// :cpp:func:`Run`.
  const std::vector<std::string> &required_inputs() const noexcept { return required_inputs_; }

  /// Returns the normalized ``"<domain>:<op_type>"`` identifiers of the kernels
  /// resolved by this session, in execution order. Repeated operators are
  /// preserved because each node owns a distinct kernel instance. The list is
  /// empty until the first :cpp:func:`Run` initializes the kernels.
  std::vector<std::string> used_kernels() const;

  /// Returns the immutable tuning-registry generation captured while kernels
  /// were initialized, or ``0`` before the first :cpp:func:`Run`.
  uint64_t tuning_generation() const noexcept {
    return tuning_snapshot_.has_value() ? tuning_snapshot_->generation() : 0;
  }

  /// Returns the cold tuning work recorded during first-run kernel initialization.
  const KernelTuningResolutionStatistics &tuning_resolution_statistics() const noexcept {
    return tuning_resolution_statistics_;
  }

  /// Enables or disables concrete-shape validation. When enabled, :cpp:func:`Run`
  /// checks that the concrete shape of every tensor carrying a declared
  /// (possibly symbolic) shape — the graph inputs, outputs and ``value_info``
  /// recorded by :cpp:func:`SetDeclaredShapes` — is consistent with that
  /// declaration: a concrete ``dim_value`` must match exactly, and every
  /// symbolic ``dim_param`` must resolve to the same concrete value everywhere
  /// it appears during a single :cpp:func:`Run`. Disabled by default so the hot
  /// path stays free of the extra checks. Declared shapes are populated
  /// automatically when the session is built from a :cpp:class:`ModelProto`; a
  /// session built from an :cpp:class:`ExecutionPlan` alone must call
  /// :cpp:func:`SetDeclaredShapes` for the check to have anything to validate.
  bool check_shapes() const noexcept { return check_shapes_; }

  /// When ``true``, :cpp:func:`Run` does not require a node's allocator-backed
  /// outputs to be owned by the session's unique allocator, so a kernel may
  /// return an output allocated outside the common allocator. When ``false``
  /// (the default), such an output is rejected. See
  /// :cpp:member:`RuntimeSessionOptions::allow_external_output_allocators`.
  bool allow_external_output_allocators() const noexcept {
    return allow_external_output_allocators_;
  }

  /// Records the declared (possibly symbolic) shapes carried by ``graph``'s
  /// inputs, outputs and ``value_info`` so that, when :cpp:func:`check_shapes`
  /// is enabled, :cpp:func:`Run` can validate concrete tensor shapes against
  /// them. Only tensor-typed values whose type carries a shape are recorded;
  /// values without a shape (unknown rank) are ignored. Calling this replaces
  /// any previously recorded declarations for the listed names. ``graph`` is
  /// read but not retained, so it need not outlive the session.
  void SetDeclaredShapes(const GraphProto &graph);

  /**
   * Returns the list of input names referenced by ``nodes`` that are not
   * produced as outputs by any node in the same list — i.e. the external
   * dependencies of the node set. Subgraph attributes (``GRAPH`` / ``GRAPHS``)
   * are inspected recursively. The returned list preserves first-seen order
   * and contains no duplicates; empty input names are skipped.
   */
  static std::vector<std::string>
  CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes);

  /**
   * Returns the full list of tensor / sequence names a single ``node`` depends
   * on at runtime: the names referenced by ``node.input()`` (skipping empty
   * optional-input slots) plus every external input of the subgraph attributes
   * (``GRAPH`` / ``GRAPHS``) attached to ``node``. The returned list preserves
   * first-seen order and contains no duplicates.
   */
  static std::vector<std::string> CollectNodeInputs(const NodeProto &node);

protected:
  /// Constructs a session over a bare :cpp:class:`GraphProto`, owning the
  /// resulting :cpp:class:`ExecutionPlan` in ``default_plan_``. Used
  /// by :cpp:class:`SubgraphSession` so a control-flow subgraph can be a
  /// :cpp:class:`RuntimeSession` with the same default resolution behavior as a
  /// top-level graph session.
  explicit RuntimeSession(const GraphProto &graph, int verbose = 0);

  /// Default node-kernel resolution used during
  /// :cpp:func:`InitializeKernels`, so :cpp:class:`RuntimeSession` and
  /// derived sessions that reuse the base initialization path share the same
  /// default resolution behavior. Builds and returns the ready-to-invoke
  /// kernel instance for ``node``.
  std::unique_ptr<KernelBase> ResolveNodeKernel(const NodeProto &node, RuntimeContext &rt,
                                                const std::string &domain,
                                                const std::string &op_type) const;

private:
  /// A node's kernel instance built once during
  /// :cpp:func:`InitializeKernels`, together with the normalised ``domain``
  /// and ``op_type`` fused into a single ``"<domain>:<op_type>"`` key (the
  /// same format used to look the kernel up in the
  /// :cpp:func:`KernelDispatchTable`) so :cpp:func:`Run` never has to
  /// recompute or re-store them separately.
  struct PreparedKernel {
    std::string key;
    std::unique_ptr<KernelBase> instance;
  };

  /// Resolves and builds the kernel instance for every node the plan executes,
  /// resolving against ``rt``, and records the external inputs those nodes
  /// read in :cpp:member:`required_inputs_`.
  void InitializeKernels(RuntimeContext &rt);

  /// Normalizes every raw tensor output of ``node`` into the arena implied by
  /// that output slot's role: a declared graph output (a name present in
  /// :cpp:member:`output_names_set_`) is normalized into
  /// :cpp:member:`session_io_allocator_` when an I/O allocator is attached,
  /// every other (intermediate) output into :cpp:member:`session_allocator_`.
  /// This is output-slot routing: a node producing both a declared output and
  /// an intermediate keeps each value in its own arena. Called after a node's
  /// kernel has run, once :cpp:member:`session_allocator_` has been captured.
  void VerifyOutputAllocators(const NodeProto &node, RuntimeContext &rt) const;

  /// Returns whether ``node`` produces at least one declared graph output
  /// (a name present in :cpp:member:`output_names_`). Used by
  /// :cpp:func:`Run` to route that node's kernel invocation through the
  /// I/O allocator instead of the execution allocator.
  bool ProducesDeclaredOutput(const NodeProto &node) const;

  /// Verifies, when :cpp:func:`check_shapes` is enabled, that the concrete
  /// shape of the tensor stored under ``name`` in ``rt`` (if any) matches the
  /// declared :cpp:class:`core::symbolic::SymShape` recorded in
  /// :cpp:member:`declared_shapes_`. Concrete dimensions must match exactly;
  /// symbolic dimensions are resolved against ``bindings`` (a
  /// :cpp:class:`core::shapes::ShapesContext` binding each symbolic expression
  /// to the concrete value it first resolved to during the current
  /// :cpp:func:`Run`), so an inconsistent reuse of the same symbol is rejected.
  /// Names without a recorded declaration, or not currently present as a
  /// tensor, are ignored.
  void VerifyDeclaredShape(const std::string &name, const RuntimeContext &rt,
                           core::shapes::ShapesContext &bindings) const;

  /// Detaches every graph output present in ``rt`` from any external memory it
  /// borrows: for each name in :cpp:member:`output_names_` whose tensor is a
  /// borrowed view (:cpp:func:`Tensor::is_borrowed`), the entry is replaced
  /// with an owned deep copy (:cpp:func:`Tensor::ToOwned`). A graph output can
  /// borrow into the model (e.g. a ``Constant`` reading its value's
  /// ``raw_data`` or an initializer passed straight through), which would
  /// dangle once the model is released; owning the bytes keeps the output
  /// valid independently of the model's lifetime. Called at the end of
  /// :cpp:func:`Run`. Only runs when :cpp:member:`output_names_` is populated
  /// (sessions built from a :cpp:class:`ModelProto` / :cpp:class:`GraphProto`).
  void MaterializeBorrowedOutputs(RuntimeContext &rt) const;

  /// Plan owned by the session, referenced by :cpp:member:`plan_` when the
  /// session is constructed from a :cpp:class:`ModelProto` (no external plan
  /// supplied). Built from the model's graph. Left empty (and unused) when a
  /// plan is passed in through the plan-taking constructor.
  ExecutionPlan default_plan_;
  const ExecutionPlan &plan_;
  std::vector<PreparedKernel> kernels_;
  /// One immutable registry generation shared by every kernel in this session.
  /// Kernels copy resolved values during initialization; retaining the snapshot
  /// also makes the generation available for diagnostics.
  std::optional<KernelTuningRegistrySnapshot> tuning_snapshot_;
  KernelTuningResolutionStatistics tuning_resolution_statistics_;
  std::vector<std::string> required_inputs_;
  /// Declared (possibly symbolic) shapes keyed by tensor name, populated by
  /// :cpp:func:`SetDeclaredShapes` and consulted by :cpp:func:`Run` when
  /// :cpp:member:`check_shapes_` is enabled.
  std::unordered_map<std::string, core::symbolic::SymShape> declared_shapes_;
  /// Names of the graph's declared outputs, populated from the
  /// :cpp:class:`ModelProto` / :cpp:class:`GraphProto` the session is built
  /// from (empty for a session built from a bare :cpp:class:`ExecutionPlan`).
  /// Consulted by :cpp:func:`MaterializeBorrowedOutputs` so borrowed graph
  /// outputs are detached from the model before :cpp:func:`Run` returns.
  std::vector<std::string> output_names_;
  /// Same names as :cpp:member:`output_names_`, indexed for O(1) membership
  /// checks by :cpp:func:`ProducesDeclaredOutput` and
  /// :cpp:func:`VerifyOutputAllocators`.
  std::unordered_set<std::string> output_names_set_;
  bool kernels_initialized_ = false;
  /// When ``true``, :cpp:func:`Run` validates concrete tensor shapes against
  /// the declarations in :cpp:member:`declared_shapes_`.
  bool check_shapes_ = false;
  /// When ``true``, :cpp:func:`Run` skips the per-node output allocator check
  /// (see :cpp:func:`VerifyOutputAllocators`), allowing outputs allocated
  /// outside the session's common allocator.
  bool allow_external_output_allocators_ = false;
  RuntimeParameters parameters_;
  /// Verbosity level used by :cpp:func:`Run` when non-zero.
  int verbose_ = 0;
  /// Allocator observed on ``rt`` the first time :cpp:func:`Run` executes;
  /// every output tensor produced by a scheduled node is verified to be
  /// backed by this same allocator (see :cpp:func:`Run`).
  RawBufferAllocator *session_allocator_ = nullptr;
  /// I/O allocator observed on ``rt`` (:cpp:func:`RuntimeContext::io_allocator`)
  /// the first time :cpp:func:`Run` executes, alongside
  /// :cpp:member:`session_allocator_`. When non-null, :cpp:func:`Run` routes
  /// the kernel invocation of any node that produces a declared graph output
  /// (see :cpp:member:`output_names_set_`) through this allocator instead of
  /// :cpp:member:`session_allocator_`.
  RawBufferAllocator *session_io_allocator_ = nullptr;
  bool session_allocator_captured_ = false;
};

} // namespace runtime
} // namespace ONNX_LIGHT_NAMESPACE::core
