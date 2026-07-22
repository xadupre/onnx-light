// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx.h"

#include <string>
#include <vector>

/**
 * @file runtime_session.h
 * @brief Declares :cpp:class:`RuntimeSession`, a reusable execution session
 *        that separates kernel initialization from execution.
 *
 * :cpp:class:`RuntimeSession` binds a fixed list of nodes to a
 * :cpp:class:`RuntimeContext` and a precomputed :cpp:class:`ExecutionPlan`,
 * resolving every node's kernel once at construction and then running the plan
 * (repeatably). The release-aware :cpp:func:`RunNodes` overload in
 * ``run_nodes.h`` is a thin wrapper that constructs a session and calls
 * :cpp:func:`RuntimeSession::Run` a single time.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

/**
 * A reusable execution session that binds a fixed list of nodes to a
 * :cpp:class:`RuntimeContext` and a precomputed :cpp:class:`ExecutionPlan`,
 * separating the runtime lifecycle into three explicit phases:
 *
 *  1. **Construction** — the session records the ``nodes``, the ``rt`` and
 *     the ``plan`` it will drive.
 *  2. **Kernel initialization** — during construction, the kernel for every
 *     node the plan will execute is resolved once (against the model-local
 *     function registry, the control-flow handlers, the user custom kernels
 *     and the static :cpp:func:`KernelDispatchTable`) and cached. Any
 *     unsupported operator is rejected here rather than mid-run.
 *  3. **Execution** — :cpp:func:`Run` replays the plan, invoking each
 *     pre-resolved kernel and releasing intermediates as scheduled. It may
 *     be called more than once (e.g. to re-run the same graph with fresh
 *     inputs) without redoing the per-node dispatch lookup.
 *
 * This mirrors how an inference runtime prepares an executable graph once
 * and then runs it repeatedly. The release-aware :cpp:func:`RunNodes`
 * overload is a thin wrapper that constructs a session and calls
 * :cpp:func:`Run` a single time.
 */
class RuntimeSession {
public:
  /**
   * Builds a session over ``nodes`` and eagerly initializes the kernel for
   * every node the ``plan`` will execute.
   *
   * @param nodes The list of nodes to execute, in topological order. The
   *              referenced object must outlive the session.
   * @param rt    In/out runtime context used both to resolve the kernels
   *              (function registry / custom kernels) and to exchange
   *              tensors during :cpp:func:`Run`. Must outlive the session.
   * @param plan  Precomputed execution / release schedule covering
   *              ``nodes``. Must outlive the session.
   *
   * @throws std::invalid_argument if the plan references an out-of-range
   *         node index or if any executed node cannot be dispatched.
   */
  RuntimeSession(const utils::RepeatedProtoField<NodeProto> &nodes, RuntimeContext &rt,
                 const ExecutionPlan &plan);

  /**
   * Executes the plan once: runs every scheduled node using its
   * pre-resolved kernel and frees each intermediate whose last reference
   * has been reached. Safe to call repeatedly on the same session.
   */
  void Run();

private:
  /// A node's kernel resolved once during construction, together with the
  /// normalised ``(domain, op_type)`` used for progress printing and event
  /// logging so :cpp:func:`Run` never has to recompute them.
  struct PreparedKernel {
    std::string domain;
    std::string op_type;
    NodeKernelFn kernel;
  };

  /// Resolves and caches the kernel for every node the plan executes.
  void InitializeKernels();

  const utils::RepeatedProtoField<NodeProto> &nodes_;
  RuntimeContext &rt_;
  const ExecutionPlan &plan_;
  std::vector<PreparedKernel> kernels_;
};

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
