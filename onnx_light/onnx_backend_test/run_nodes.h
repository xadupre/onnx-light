// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"
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
 * :cpp:type:`TensorMap` so a chain of nodes can be evaluated in
 * topological order (as required by the ONNX specification for
 * ``GraphProto::node()``). A node is dispatched by its
 * ``(domain, op_type)`` pair through :cpp:func:`KernelDispatchTable`;
 * new operators are added by registering a single new entry in that
 * table without changing :cpp:func:`RunNode` / :cpp:func:`RunNodes`.
 *
 * Only a small, working baseline of operators is registered today
 * (the simple element-wise ``ai.onnx`` ops ``Abs``, ``Add``, ``Div``,
 * ``Mul``, ``Neg``, ``Sub``). The dispatcher is deliberately
 * extensible: as more kernels become wirable through a uniform
 * ``NodeProto``-driven call site, additional entries can be added
 * to ``KernelDispatchTable`` and they become callable from
 * :cpp:func:`RunNode` / :cpp:func:`RunNodes` automatically.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/**
 * Name-keyed map of tensors carrying both the graph inputs/initializers
 * and the intermediate values produced by previously executed nodes.
 * Stored inside :cpp:class:`RuntimeContext`; :cpp:func:`RunNode` reads
 * a node's inputs from this map by name (matching ``node.input(i)``)
 * and inserts every produced output under the name declared by
 * ``node.output(i)``.
 */
using TensorMap = std::unordered_map<std::string, Tensor>;

/**
 * Per-invocation runtime state passed to :cpp:func:`RunNode` /
 * :cpp:func:`RunNodes`.
 *
 * Bundles together everything a chain of nodes needs to execute:
 *  * ``tensors`` — the name-keyed :cpp:type:`TensorMap` carrying the
 *    graph inputs / initializers and every intermediate value
 *    produced by previously executed nodes.
 *  * ``kernel_ctx`` — the construction-time
 *    :cpp:class:`kernel::KernelContext` (opset and any future
 *    construction-time inputs) used to instantiate each per-operator
 *    kernel.
 *
 * Grouping them in a single object keeps the dispatcher signatures
 * stable as more per-invocation state (allocators, device descriptors,
 * profiling hooks, …) is added in the future without forcing every
 * trampoline or call site to take an extra argument.
 */
struct RuntimeContext {
  /// In/out tensor map shared across every node in a chain.
  TensorMap tensors;
  /// Kernel construction context (opset).
  kernel::KernelContext kernel_ctx;

  RuntimeContext() = default;
  explicit RuntimeContext(kernel::KernelContext kernel_ctx_) : kernel_ctx(std::move(kernel_ctx_)) {}
};

/**
 * Signature of every per-operator trampoline registered in
 * :cpp:func:`KernelDispatchTable`. Implementations read their inputs
 * from ``rt.tensors`` by name, call the matching kernel (constructed
 * with ``rt.kernel_ctx``), and insert the produced outputs back into
 * ``rt.tensors`` under the names declared by ``node.output(i)``.
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
 * ``rt.tensors``.
 *
 * The node's input descriptors are read from ``rt.tensors`` by name
 * (so every non-empty input must already be present), and the output
 * descriptors are inserted into ``rt.tensors`` under the names
 * declared by ``node.output(i)``.
 *
 * @param node The node to execute.
 * @param rt   In/out runtime context. ``rt.tensors`` must already
 *             contain entries for every input referenced by ``node``;
 *             on return it also contains entries for every output
 *             declared by ``node``. ``rt.kernel_ctx`` is used to
 *             construct the per-operator kernel.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         registered in :cpp:func:`KernelDispatchTable`, if a
 *         required input is missing from ``rt.tensors``, or if the
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
 * carried in ``rt.tensors`` or as the output of an earlier node in
 * ``nodes`` — by the time the node is processed.
 *
 * @param nodes The list of nodes to execute, in topological order.
 * @param rt    In/out runtime context seeded with the graph inputs
 *              and initializers in ``rt.tensors``; on return it
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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
