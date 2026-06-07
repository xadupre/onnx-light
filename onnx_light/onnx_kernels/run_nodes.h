// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/runtime_context.h"
#include "onnx_proto/onnx.h"

#include <functional>
#include <string>
#include <unordered_map>

/**
 * @file run_nodes.h
 * @brief Tiny dispatcher that runs the matching backend test
 *        kernel for one ``NodeProto`` or for a list of nodes (an
 *        iterator), mirroring the per-operator
 *        :cpp:func:`onnx_optim::shapes::ComputeShapeNode` /
 *        :cpp:func:`onnx_optim::shapes::ComputeShapes` pair used
 *        by ``onnx_optim`` for shape inference.
 *
 * Inputs and outputs are exchanged through a name-keyed
 * :cpp:type:`TensorMap` (owned by :cpp:class:`RuntimeContext`) so a
 * chain of nodes can be evaluated in topological order (as required
 * by the ONNX specification for ``GraphProto::node()``). A node is
 * dispatched by its ``(domain, op_type)`` pair through
 * :cpp:func:`KernelDispatchTable`; new operators are added by
 * registering a single new entry in that table without changing
 * :cpp:func:`RunNode` / :cpp:func:`RunNodes`.
 *
 * Only a small, working baseline of operators is registered today
 * (the simple element-wise ``ai.onnx`` ops ``Abs``, ``Add``, ``Div``,
 * ``Mul``, ``Neg``, ``Sub``). The dispatcher is deliberately
 * extensible: as more kernels become wirable through a uniform
 * ``NodeProto``-driven call site, additional entries can be added
 * to ``KernelDispatchTable`` and they become callable from
 * :cpp:func:`RunNode` / :cpp:func:`RunNodes` automatically.
 *
 * In addition to the static :cpp:func:`KernelDispatchTable`,
 * :cpp:func:`RunNode` also consults
 * :cpp:func:`RuntimeContext::functions` for model-local functions
 * (``ModelProto::functions``). When a node's
 * ``(domain, op_type, overload)`` triple matches a registered
 * :cpp:type:`FunctionProto`, the call is dispatched to
 * :cpp:func:`RunFunction` with a fresh child :cpp:class:`RuntimeContext`
 * bound to the function's formal inputs; the function's formal outputs
 * are then propagated back to the caller under the names declared by
 * ``node.output``. :cpp:func:`RunModel` populates that registry from
 * ``model.functions()`` before delegating to :cpp:func:`RunGraph`, so
 * nodes referring to local functions are resolved transparently.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

/**
 * Signature of every per-operator trampoline registered in
 * :cpp:func:`KernelDispatchTable`. Implementations read their inputs
 * from ``rt.tensors()`` by name, call the matching kernel
 * (constructed with ``rt.kernel_ctx()``), and insert the produced
 * outputs back into ``rt.tensors()`` under the names declared by
 * ``node.output(i)``.
 */
using NodeKernelFn = std::function<void(const NodeProto &node, RuntimeContext &rt)>;

/**
 * Returns the ``"<domain>:<op_type>"`` dispatch table used by
 * :cpp:func:`RunNode`. Constructed on first use and shared across
 * calls. The default ONNX domain (empty string in
 * ``NodeProto::domain()``) is normalised to ``"ai.onnx"`` before
 * lookup; adding a new operator only requires inserting one new
 * entry in this table.
 */
const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable();

/**
 * Runs the kernel registered for ``node`` and stores its outputs in
 * ``rt.tensors()``.
 *
 * The node's input descriptors are read from ``rt.tensors()`` by
 * name (so every non-empty input must already be present), and the
 * output descriptors are inserted into ``rt.tensors()`` under the
 * names declared by ``node.output(i)``.
 *
 * @param node The node to execute.
 * @param rt   In/out runtime context. ``rt.tensors()`` must already
 *             contain entries for every input referenced by ``node``;
 *             on return it also contains entries for every output
 *             declared by ``node``. ``rt.kernel_ctx()`` is used to
 *             construct the per-operator kernel.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         registered in :cpp:func:`KernelDispatchTable`, if a
 *         required input is missing from ``rt.tensors()``, or if the
 *         per-operator trampoline rejects the node.
 */
void RunNode(const NodeProto &node, RuntimeContext &rt);

/**
 * Runs :cpp:func:`RunNode` on every node of ``nodes`` in order.
 *
 * The sequence must be topologically sorted with respect to data
 * dependencies (as required by the ONNX specification for
 * ``GraphProto::node``) so that every input of a node has already
 * been produced — either as a pre-existing graph input/initializer
 * carried in ``rt.tensors()`` or as the output of an earlier node
 * in ``nodes`` — by the time the node is processed.
 *
 * @param nodes The list of nodes to execute, in topological order.
 * @param rt    In/out runtime context seeded with the graph inputs
 *              and initializers in ``rt.tensors()``; on return it
 *              additionally contains every node output.
 *
 * @throws std::invalid_argument if any node cannot be dispatched.
 */
void RunNodes(const utils::RepeatedProtoField<NodeProto> &nodes, RuntimeContext &rt);

/**
 * Generic iterator overload of :cpp:func:`RunNodes`. Accepts any
 * input-iterator range whose ``value_type`` (after dereferencing) is
 * convertible to ``const NodeProto &``, so callers can drive the
 * dispatcher from ``std::vector<NodeProto>``, ``std::list<NodeProto>``
 * or any other container — not only ``RepeatedProtoField``.
 */
template <class InputIt> void RunNodes(InputIt first, InputIt last, RuntimeContext &rt) {
  for (auto it = first; it != last; ++it) {
    RunNode(*it, rt);
  }
}

/**
 * Runs all nodes in a ``GraphProto`` using the provided runtime context.
 *
 * Before executing the node sequence the function seeds ``rt.tensors()``
 * with every ``TensorProto`` in ``graph.initializer()``, so that
 * downstream nodes can look up constant values by name.  Graph inputs
 * that the caller has already placed in ``rt.tensors()`` are left as-is.
 *
 * @param graph The graph to evaluate. Its ``node`` list must already be
 *              in topological order (as required by the ONNX spec).
 * @param rt    In/out runtime context seeded with the graph inputs;
 *              on return ``rt.tensors()`` additionally contains every
 *              node output and every initializer.
 *
 * @throws std::invalid_argument if any node cannot be dispatched.
 */
void RunGraph(const GraphProto &graph, RuntimeContext &rt);

/**
 * Runs all nodes in a ``FunctionProto`` using the provided runtime context.
 *
 * The caller is responsible for inserting the function's input tensors
 * into ``rt.tensors()`` before calling this function.  On return
 * ``rt.tensors()`` additionally contains every node output.
 *
 * @param func The function to evaluate. Its ``node`` list must already be
 *             in topological order.
 * @param rt   In/out runtime context seeded with the function's inputs;
 *             on return it additionally contains every node output.
 *
 * @throws std::invalid_argument if any node cannot be dispatched.
 */
void RunFunction(const FunctionProto &func, RuntimeContext &rt);

/**
 * Runs the graph embedded in a ``ModelProto`` using the provided runtime
 * context.
 *
 * Before delegating to :cpp:func:`RunGraph`, every ``FunctionProto`` in
 * ``model.functions()`` is registered in
 * :cpp:func:`RuntimeContext::functions` so that nodes referring to a
 * model-local function by ``(domain, op_type, overload)`` are
 * dispatched through :cpp:func:`RunFunction` rather than the static
 * :cpp:func:`KernelDispatchTable`. The caller is responsible for
 * inserting the model's input tensors into ``rt.tensors()`` beforehand.
 *
 * @param model The model whose ``graph`` field will be evaluated.
 * @param rt    In/out runtime context seeded with the model inputs;
 *              on return it additionally contains every node output and
 *              every graph initializer.
 *
 * @throws std::invalid_argument if the model has no graph or any node
 *         cannot be dispatched.
 */
void RunModel(const ModelProto &model, RuntimeContext &rt);

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
