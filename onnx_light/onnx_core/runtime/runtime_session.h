// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_parameters.h"
#include "onnx_proto/onnx.h"

#include <memory>
#include <string>
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
 * :cpp:class:`RuntimeSession`), :cpp:func:`RunSubgraph`, the model-local
 * function call helper, and the ``If`` / ``Loop`` / ``Scan`` control-flow
 * kernels — constructs one of these sessions (over the graph's or
 * function's cached :cpp:class:`ExecutionPlan`) and calls :cpp:func:`Run` a
 * single time.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

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
   * Builds a session over ``plan``. Kernel resolution is deferred to the first
   * :cpp:func:`Run` (which supplies the :cpp:class:`RuntimeContext` the
   * kernels are resolved against).
   *
   * @param plan Precomputed execution / release schedule. Its node list
   *             (:cpp:func:`ExecutionPlan::nodes`) drives execution. The plan
   *             (and the graph / function it was built from) must outlive the
   *             session.
   */
  explicit RuntimeSession(const ExecutionPlan &plan);

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
   * @param rt In/out runtime context used both to resolve the kernels
   *           (function registry / custom kernels) and to exchange tensors.
   *
   * @throws std::invalid_argument if the plan references an out-of-range node
   *         index, if any executed node cannot be dispatched, or if ``rt`` does
   *         not define one of the plan's required external inputs.
   */
  void Run(RuntimeContext &rt);

  /// Model-independent execution parameters (e.g. the requested degree of
  /// parallelism, :cpp:var:`RuntimeParameters::num_threads`) applied to the
  /// nodes this session runs.
  void set_parameters(RuntimeParameters parameters) noexcept { parameters_ = parameters; }
  const RuntimeParameters &parameters() const noexcept { return parameters_; }

  /// Returns the external input names the scheduled nodes read (the inputs that
  /// must be present in the :cpp:class:`RuntimeContext` before :cpp:func:`Run`).
  /// Populated during kernel initialization; empty until the first
  /// :cpp:func:`Run`.
  const std::vector<std::string> &required_inputs() const noexcept { return required_inputs_; }

  /**
   * Returns the list of input names referenced by ``nodes`` that are not
   * produced as outputs by any node in the same list — i.e. the external
   * dependencies of the node set. Subgraph attributes (``GRAPH`` / ``GRAPHS``)
   * are inspected recursively. The returned list preserves first-seen order
   * and contains no duplicates; empty input names are skipped.
   */
  static std::vector<std::string>
  CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes);

  /// ``std::vector``-overload of :cpp:func:`CollectExternalInputs`.
  static std::vector<std::string> CollectExternalInputs(const std::vector<NodeProto> &nodes);

  /**
   * Returns the full list of tensor / sequence names a single ``node`` depends
   * on at runtime: the names referenced by ``node.input()`` (skipping empty
   * optional-input slots) plus every external input of the subgraph attributes
   * (``GRAPH`` / ``GRAPHS``) attached to ``node``. The returned list preserves
   * first-seen order and contains no duplicates.
   */
  static std::vector<std::string> CollectNodeInputs(const NodeProto &node);

private:
  /// A node's kernel instance built once during
  /// :cpp:func:`InitializeKernels`, together with the normalised ``domain``
  /// and ``op_type`` fused into a single ``"<domain>:<op_type>"`` key (the
  /// same format used to look the kernel up in the
  /// :cpp:func:`KernelDispatchTable`) so :cpp:func:`Run` never has to
  /// recompute or re-store them separately.
  struct PreparedKernel {
    PreparedKernel() = default;
    PreparedKernel(const PreparedKernel &other)
        : key(other.key),
          instance(other.instance != nullptr ? std::make_unique<ResolvedKernel>(*other.instance)
                                             : nullptr) {}
    PreparedKernel &operator=(const PreparedKernel &other) {
      if (this == &other) {
        return *this;
      }
      key = other.key;
      instance =
          other.instance != nullptr ? std::make_unique<ResolvedKernel>(*other.instance) : nullptr;
      return *this;
    }
    PreparedKernel(PreparedKernel &&) noexcept = default;
    PreparedKernel &operator=(PreparedKernel &&) noexcept = default;

    std::string key;
    std::unique_ptr<ResolvedKernel> instance;
  };

  /// Resolves and builds the kernel instance for every node the plan executes,
  /// resolving against ``rt``, and records the external inputs those nodes
  /// read in :cpp:member:`required_inputs_`.
  void InitializeKernels(RuntimeContext &rt);

  const ExecutionPlan &plan_;
  std::vector<PreparedKernel> kernels_;
  std::vector<std::string> required_inputs_;
  bool kernels_initialized_ = false;
  RuntimeParameters parameters_;
};

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
