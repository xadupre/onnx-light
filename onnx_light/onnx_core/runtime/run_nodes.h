// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_proto/onnx.h"

#include <string>
#include <utility>
#include <vector>

/**
 * @file run_nodes.h
 * @brief Tiny dispatcher that runs the matching backend test
 *        kernel for one ``NodeProto``, mirroring the per-operator
 *        :cpp:func:`core::shapes::ShapesContext::ComputeShapeNode` /
 *        :cpp:func:`core::shapes::ShapesContext::ComputeShapes` pair used
 *        by ``onnx_shapes`` for shape inference.
 *
 * Inputs and outputs are exchanged through a name-keyed
 * :cpp:type:`TensorMap` (owned by :cpp:class:`RuntimeContext`). A node is
 * dispatched by its ``(domain, op_type)`` pair through
 * :cpp:func:`KernelDispatchTable`; new operators are added by
 * registering a single new entry in that table without changing
 * :cpp:func:`RunNode`.
 *
 * Only a small, working baseline of operators is registered today
 * (the simple element-wise ``ai.onnx`` ops ``Abs``, ``Add``, ``Div``,
 * ``Mul``, ``Neg``, ``Sub``). The dispatcher is deliberately
 * extensible: as more kernels become wirable through a uniform
 * ``NodeProto``-driven call site, additional entries can be added
 * to ``KernelDispatchTable`` and they become callable from
 * :cpp:func:`RunNode` automatically.
 *
 * Whenever a whole node *list* (as opposed to a single node) needs to be
 * executed — a graph, a function body, or a subgraph — callers build an
 * :cpp:class:`ExecutionPlan` for it and drive it through a
 * :cpp:class:`RuntimeSession` themselves (:cpp:func:`RunSubgraph` does this
 * internally for embedded control-flow subgraphs since it also has to
 * propagate the subgraph's outputs back to the caller).
 *
 * In addition to the static :cpp:func:`KernelDispatchTable`,
 * :cpp:func:`RunNode` also consults
 * :cpp:func:`RuntimeContext::functions` for model-local functions
 * (``ModelProto::functions``). When a node's
 * ``(domain, op_type, overload)`` triple matches a registered
 * :cpp:type:`FunctionProto`, the call is dispatched to a fresh child
 * :cpp:class:`RuntimeContext` bound to the function's formal inputs and run
 * through a :cpp:class:`RuntimeSession`; the function's formal outputs are
 * then propagated back to the caller under the names declared by
 * ``node.output``. :cpp:func:`RegisterModelFunctions` populates that
 * registry from a ``ModelProto``'s ``functions()`` field so nodes referring
 * to local functions are resolved transparently once the caller runs the
 * model's graph through its own :cpp:class:`ExecutionPlan` /
 * :cpp:class:`RuntimeSession`.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

/**
 * Signature of every per-operator factory registered in
 * :cpp:func:`core::runtime::KernelDispatchTable`. Implementations validate
 * the node, read any construction-time attributes, construct the matching
 * kernel with ``rt.kernel_ctx()``, and return a reusable
 * :cpp:class:`ResolvedKernel` whose :cpp:func:`Invoke` performs the actual
 * per-run tensor reads / writes.
 *
 * The alias and the table itself are declared in
 * ``onnx_core/runtime/kernel_dispatch_table.h`` (transitively included
 * above); this file is left as a comment so the public API surface of
 * ``run_nodes.h`` remains documented in one place. Kernel implementations
 * (``onnx_kernels``) populate the table via
 * ``onnx_kernels::RegisterKernelFunctions``.
 */

/**
 * Runs the kernel registered for ``node`` and stores its outputs in
 * ``rt.tensors()``.
 *
 * The node's input descriptors are read from ``rt.tensors()`` by
 * name (so every non-empty input must already be present), and the
 * output descriptors are inserted into ``rt.tensors()`` under the
 * names declared by ``node.output(i)``.
 *
 * In addition to table-dispatched kernels and model-local functions,
 * this dispatcher also evaluates control-flow nodes (``If``, ``Loop``,
 * ``Scan``) by recursively executing their embedded subgraphs.
 *
 * @param node The node to execute.
 * @param rt   In/out runtime context. ``rt.tensors()`` must already
 *             contain entries for every input referenced by ``node``;
 *             on return it also contains entries for every output
 *             declared by ``node``. ``rt.kernel_ctx()`` is used to
 *             construct the per-operator kernel instance.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         registered in :cpp:func:`KernelDispatchTable`, if a required
 *         input is missing from ``rt.tensors()``, or if the per-operator
 *         factory / resolved kernel rejects the node.
 */
void RunNode(const NodeProto &node, RuntimeContext &rt);

/**
 * Registers every ``FunctionProto`` in ``model.functions()`` in
 * :cpp:func:`RuntimeContext::functions` so that nodes referring to a
 * model-local function by ``(domain, op_type, overload)`` are dispatched
 * to it rather than the static :cpp:func:`KernelDispatchTable`.
 *
 * Callers running a ``ModelProto``'s graph must call this once (before
 * building the graph's :cpp:class:`ExecutionPlan` and driving it through a
 * :cpp:class:`RuntimeSession`) so that any node referring to a model-local
 * function resolves correctly; this function itself does not run any
 * nodes.
 *
 * @param model The model whose ``functions()`` are registered.
 * @param rt    In/out runtime context whose function registry is updated.
 *
 * @throws std::invalid_argument if ``model`` has no graph.
 */
void RegisterModelFunctions(const ModelProto &model, RuntimeContext &rt);

/**
 * Evaluates a subgraph in a fresh child :cpp:class:`RuntimeContext` that
 * inherits the caller's tensor map and function registry, additionally
 * seeded with ``bindings`` (typically the formal-input ↔ actual-input
 * tensor pairs for the subgraph). Returns the subgraph's outputs in the
 * order declared by ``graph.output()``.
 *
 * The subgraph is executed the same way as any other node list: its
 * initializers are seeded into the child context and its cached
 * :cpp:class:`ExecutionPlan` is driven through a fresh
 * :cpp:class:`RuntimeSession`.
 *
 * When the caller's context has event logging enabled
 * (:cpp:func:`RuntimeContext::events_enabled`), child events are appended
 * to the caller's event log after the subgraph finishes. Each propagated
 * event carries :cpp:var:`RuntimeEvent::subgraph_node_index` set to
 * ``rt.current_node_index()`` (the index of the control-flow node in the
 * parent graph) and :cpp:var:`RuntimeEvent::subgraph_attr_name` set to
 * ``attr_name``, so consumers can distinguish subgraph events from
 * top-level events.
 *
 * Exposed publicly so control-flow kernels (e.g. :cpp:class:`kernel::Scan`)
 * can run their body subgraph without going through
 * :cpp:func:`RunNode` themselves.
 *
 * ``bindings`` is taken by value so callers can move allocator-backed tensors
 * into the subgraph without retaining dangling ownership in the caller.
 *
 * @param attr_name  Attribute name identifying the subgraph within its
 *                   owning control-flow node (e.g. ``"body"``,
 *                   ``"then_branch"``, ``"else_branch"``). Stored in
 *                   :cpp:var:`RuntimeEvent::subgraph_attr_name` of every
 *                   event produced during the subgraph run.
 *
 * @throws std::invalid_argument if a subgraph output has an empty name or
 *         is not produced by the body.
 */
Tensors RunSubgraph(const GraphProto &graph, std::vector<std::pair<std::string, Tensor>> bindings,
                    RuntimeContext &rt, const std::string &attr_name = "");

/**
 * Resolves a possibly-negative ``axis`` against a tensor of rank
 * ``rank`` and returns the non-negative axis in ``[0, rank)``. Throws
 * ``std::invalid_argument`` with a message that mentions ``op_name``
 * when the axis is out of range.
 */
int64_t ResolveAxis(int64_t axis, std::size_t rank, const std::string &op_name);

/**
 * Returns the tensor obtained by selecting the ``index``-th slice of
 * ``t`` along axis ``axis`` (the resulting tensor's rank is
 * ``t.shape.size() - 1``). The slice is copied into a fresh buffer; the
 * source tensor is not modified. ``op_name`` only appears in error
 * messages.
 *
 * @throws std::invalid_argument if ``t`` is a rank-0 tensor, the index
 *         is out of range, or the slice would exceed addressable buffer
 *         size.
 */
Tensor SliceTensorAlongAxis(const Tensor &t, int64_t axis, int64_t index,
                            const std::string &op_name);

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
