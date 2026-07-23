// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_dispatch_table.h"
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
 * resolve-on-demand path (:cpp:func:`RunNode`) and the resolve-once path
 * (:cpp:class:`RuntimeSession`) can share the exact same node dispatch and
 * kernel-invocation logic without duplicating it. Their definitions live in
 * ``run_nodes.cc`` (next to the private control-flow / function-call helpers
 * they depend on).
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {
namespace detail {

/**
 * Resolves how ``node`` must be dispatched, returning the factory that builds
 * the ready-to-invoke kernel instance (without any progress printing or event
 * logging). This is the "kernel initialization" step: the ``(domain, op_type)``
 * resolution against the model-local function registry, the control-flow
 * handlers, the user-registered custom kernels and the static
 * :cpp:func:`KernelDispatchTable` is performed once here so that a caller (e.g.
 * :cpp:class:`RuntimeSession`) can prepare every node's kernel instance up
 * front and then invoke it repeatedly without redoing the lookup.
 *
 * Resolution precedence mirrors the historical inline logic of
 * :cpp:func:`RunNode`: model-local functions override built-ins, then the
 * control-flow operators, then user custom kernels, then the dispatch table.
 * An unsupported ``(domain, op_type)`` is rejected here (at resolution time)
 * with the same diagnostic previously emitted at run time.
 */
NodeKernelFn ResolveNodeKernel(const NodeProto &node, RuntimeContext &rt, const std::string &domain,
                               const std::string &op_type);

/**
 * Invokes an already-built kernel instance for ``node``, wrapping the call with
 * the verbose progress line and (when enabled) the per-node timing event.
 * Shared by :cpp:func:`RunNode` and :cpp:class:`RuntimeSession` so both the
 * resolve-on-demand and the resolve-once execution paths log identically.
 */
void InvokeResolvedKernel(const NodeProto &node, RuntimeContext &rt, const std::string &domain,
                          const std::string &op_type, const ResolvedKernel &kernel);

} // namespace detail
} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
