// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx.h"

#include <string>

/**
 * @file run_nodes_internal.h
 * @brief Internal helpers shared between :cpp:func:`RunNode`
 *        (``run_nodes.cc``) and :cpp:class:`RuntimeSession`
 *        (``runtime_session.cc``).
 *
 * These declarations are not part of the public API; they exist only so the
 * resolve-on-demand path (:cpp:func:`RunNode`) and the default
 * :cpp:class:`RuntimeSession` resolution path can share the exact same node
 * dispatch logic. Their definitions live in ``run_nodes.cc`` (next to the
 * private control-flow / function-call helpers they depend on).
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime::detail {

/**
 * Resolves how ``node`` must be dispatched, building and returning the
 * ready-to-invoke kernel instance (without any progress printing or event
 * logging). This is the "kernel initialization" step: the ``(domain, op_type)``
 * resolution against the model-local function registry, the control-flow
 * handlers, the user-registered custom kernels and the static
 * :cpp:func:`KernelDispatchTable` is performed once here so that a caller (e.g.
 * :cpp:class:`RuntimeSession`) can prepare every node's kernel instance up
 * front and then invoke it repeatedly without redoing the lookup.
 *
 * Resolution precedence mirrors the historical inline logic of
 * :cpp:func:`RunNode`: model-local functions override built-ins, then the
 * control-flow operators, then per-context custom kernels, then global custom
 * kernels, then the dispatch table.
 * An unsupported ``(domain, op_type)`` is rejected here (at resolution time)
 * with the same diagnostic previously emitted at run time.
 */
std::unique_ptr<KernelBase> ResolveNodeKernelDefault(const NodeProto &node, RuntimeContext &rt,
                                                     const std::string &domain,
                                                     const std::string &op_type);

/**
 * Emits the ReferenceEvaluator verbose progress line for one node dispatch. The
 * format is ``[ReferenceEvaluator] #<node_index> Domain::OpType(inputs) ->
 * (outputs)``. Nothing is printed when the effective verbosity (the explicit
 * ``verbose_override`` when it is non-negative, otherwise ``rt.verbose()``) is
 * ``<= 0``. Shared by the inlined invoke logic of :cpp:func:`RunNode`
 * (``run_nodes.cc``) and :cpp:class:`RuntimeSession` (``runtime_session.cc``)
 * so both log identically.
 */
void PrintNodeProgress(const RuntimeContext &rt, const NodeProto &node, const std::string &domain,
                       const std::string &op_type, int verbose_override = -1);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime::detail
