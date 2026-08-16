// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
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
 * :cpp:class:`RuntimeSession` themselves. For embedded control-flow
 * subgraphs — ``Loop`` / ``Scan`` / ``SequenceMap`` bodies,
 * ``FlexAttention``'s ``score_mod`` / ``prob_mod`` — callers instead build
 * one :cpp:class:`SubgraphSession` up front (which also propagates the
 * subgraph's outputs back to the caller) and call its
 * :cpp:func:`SubgraphSession::Run` once per invocation instead of
 * re-resolving the subgraph's kernels each time.
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

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Signature of every per-operator factory registered in
 * :cpp:func:`core::runtime::KernelDispatchTable`. Implementations validate
 * the node, read any construction-time attributes, construct the matching
 * kernel with ``rt.kernel_ctx()``, and return a reusable
 * :cpp:class:`Kernel` whose :cpp:func:`Kernel::Run` performs the actual
 * per-run tensor reads / writes.
 *
 * The alias and the table itself are declared in
 * ``onnx_core/runtime/kernels/kernel_dispatch_table.h`` (transitively included
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
 * Runs ``model``'s graph end-to-end and returns its outputs as named
 * :cpp:class:`Tensor` objects.
 *
 * Convenience wrapper that performs, in one call, the full sequence a caller
 * would otherwise assemble by hand to run a whole model: it derives the
 * default-domain opset version from ``model.opset_import()``, builds a
 * :cpp:class:`RuntimeContext`, registers every model-local function
 * (:cpp:func:`RegisterModelFunctions`), seeds the supplied ``inputs`` and the
 * graph's initializers, builds a :cpp:class:`RuntimeSession` over
 * ``model.graph()`` and runs it once. The returned tensors own their bytes, so
 * they remain valid after ``model`` is released.
 *
 * Each entry of ``inputs`` is stored under its :cpp:member:`Tensor::name`, so
 * the caller must set that name to the graph input the tensor feeds. Inputs are
 * consumed (moved) by this call.
 *
 * @param model   Model whose graph is executed. Must contain a graph.
 * @param inputs  External input tensors keyed by their :cpp:member:`Tensor::name`.
 * @param verbose Verbosity level forwarded to the :cpp:class:`RuntimeContext`
 *                and :cpp:class:`RuntimeSession` (``0`` disables progress output).
 *
 * @return The graph's declared outputs, in declaration order, as owned tensors.
 *
 * @throws std::invalid_argument if ``model`` has no graph, if a required input
 *         is missing, or if a declared output is not produced by the run.
 */
Tensors RunModel(const ModelProto &model, Tensors inputs, int verbose = 0);

/**
 * Reusable driver for a control-flow subgraph (``Loop`` / ``Scan`` /
 * ``SequenceMap`` body, ``FlexAttention``'s ``score_mod`` / ``prob_mod``)
 * that separates one-time setup from the repeated per-iteration run, mirroring
 * how :cpp:class:`RuntimeSession` separates kernel resolution from execution.
 *
 * **Construction** builds the subgraph's :cpp:class:`ExecutionPlan` directly
 * from ``graph`` (owned by this instance, not obtained from ``rt``'s
 * per-context plan cache — see :cpp:func:`SubgraphSession::SubgraphSession`
 * for why), the :cpp:class:`RuntimeSession` that drives it, and caches the
 * graph-derived data every run needs — the parsed initializer tensors and the
 * declared output names — so ``graph`` itself does not need to be kept
 * around, or passed again, once the session exists.
 *
 * **:cpp:func:`Run`** evaluates the subgraph in a fresh child
 * :cpp:class:`RuntimeContext` that inherits the caller's tensor map and
 * function registry, seeded with the cached initializers and with
 * ``bindings`` (typically the formal-input ↔ actual-input tensor pairs for
 * the subgraph), and returns the subgraph's outputs in the order declared by
 * the graph the session was built from. Safe to call repeatedly (once per
 * ``Loop`` / ``Scan`` / ``SequenceMap`` iteration, or once per
 * ``FlexAttention`` ``score_mod`` / ``prob_mod`` invocation): the subgraph's
 * kernels are resolved once, on the first call, and reused on every
 * subsequent one.
 *
 * When the caller's context has event logging enabled
 * (:cpp:func:`RuntimeContext::events_enabled`), child events are appended to
 * the caller's event log after the subgraph finishes. Each propagated event
 * carries :cpp:var:`RuntimeEvent::subgraph_node_index` set to
 * ``rt.current_node_index()`` (the index of the control-flow node in the
 * parent graph) and :cpp:var:`RuntimeEvent::subgraph_attr_name` set to
 * ``attr_name``, so consumers can distinguish subgraph events from top-level
 * events.
 *
 * Exposed publicly so control-flow kernels (e.g. :cpp:class:`kernel::Scan`)
 * can run their body subgraph without going through :cpp:func:`RunNode`
 * themselves.
 */
class SubgraphSession : public RuntimeSession {
public:
  using RuntimeSession::Run;

  /**
   * Initializes this :cpp:class:`RuntimeSession` for the subgraph and caches
   * ``graph``'s initializers (already parsed into :cpp:class:`Tensor`) and
   * output names.
   *
   * The :cpp:class:`ExecutionPlan` is built directly from ``graph`` by the
   * base :cpp:class:`RuntimeSession` and owned by this instance (rather than
   * obtained from ``rt``'s per-context plan cache), so that a
   * :cpp:class:`SubgraphSession` cached once by a control-flow node's kernel
   * factory and reused across
   * repeated executions of that node — including when that node itself is
   * nested inside an outer ``Loop`` / ``Scan`` / ``SequenceMap`` body and
   * ``rt`` is therefore a short-lived per-iteration child context — never
   * outlives the plan it depends on.
   *
   * @param rt    Runtime context; only used to propagate events during
   *              construction-time bookkeeping (kept for API symmetry with
   *              :cpp:func:`Run` / :cpp:func:`RunChild`; the plan itself no
   *              longer depends on it).
   * @param graph The subgraph to run repeatedly via :cpp:func:`Run`. Must
   *              outlive this :cpp:class:`SubgraphSession` (its nodes are
   *              referenced by pointer from the owned
   *              :cpp:class:`ExecutionPlan`); typically part of the parsed
   *              model, so this holds for the model's whole lifetime.
   */
  SubgraphSession(RuntimeContext &rt, const GraphProto &graph);

  /**
   * Evaluates the subgraph once in a fresh child :cpp:class:`RuntimeContext`,
   * seeded with the cached initializers and with ``bindings``. Returns the
   * subgraph's outputs in the order declared by the graph the session was
   * built from.
   *
   * ``bindings`` is taken by value so callers can move allocator-backed
   * tensors into the subgraph without retaining dangling ownership in the
   * caller.
   *
   * @param bindings  Formal-input ↔ actual-input tensor pairs for this call.
   * @param rt        The caller's runtime context; used to propagate events
   *                   and to record the caller-visible allocator.
   * @param attr_name Attribute name identifying the subgraph within its
   *                  owning control-flow node (e.g. ``"body"``,
   *                  ``"then_branch"``, ``"else_branch"``). Stored in
   *                  :cpp:var:`RuntimeEvent::subgraph_attr_name` of every
   *                  event produced during the run.
   *
   * @throws std::invalid_argument if a subgraph output has an empty name or
   *         is not produced by the subgraph.
   */
  Tensors Run(std::vector<std::pair<std::string, Tensor>> bindings, RuntimeContext &rt,
              const std::string &attr_name = "");

  /**
   * Lower-level counterpart of :cpp:func:`Run` for callers that need to
   * inspect the evaluated child context directly instead of getting back
   * only the declared tensor outputs — e.g. ``If``, whose branches may
   * produce sequence-typed outputs that :cpp:func:`Run` (which only reads
   * ``child.tensors()``) cannot represent, or ``Loop``'s sequence-typed
   * loop-carried state, which needs ``sequence_bindings`` bound before the
   * subgraph runs.
   *
   * Seeds the cached initializers and ``bindings`` / ``sequence_bindings``
   * into a fresh child :cpp:class:`RuntimeContext` (as :cpp:func:`Run`
   * does), evaluates the cached :cpp:class:`RuntimeSession` once, and
   * returns the resulting child context so the caller can pull out
   * whatever outputs (tensor- or sequence-typed) it needs. Propagates
   * child events to ``rt`` exactly like :cpp:func:`Run`.
   *
   * @param bindings           Formal-input <-> actual-input tensor pairs.
   * @param sequence_bindings  Formal-input <-> actual-input sequence pairs.
   * @param rt                 The caller's runtime context; used to propagate
   *                           events and to record the caller-visible allocator.
   * @param attr_name          Attribute name identifying the subgraph within
   *                           its owning control-flow node.
   */
  RuntimeContext RunChild(std::vector<std::pair<std::string, Tensor>> bindings,
                          std::vector<std::pair<std::string, Sequence>> sequence_bindings,
                          RuntimeContext &rt, const std::string &attr_name = "");

  /// ``Tensor``-bindings-only overload of :cpp:func:`RunChild`.
  RuntimeContext RunChild(std::vector<std::pair<std::string, Tensor>> bindings, RuntimeContext &rt,
                          const std::string &attr_name = "");

private:
  std::vector<std::pair<std::string, Tensor>> initializers_;
  std::vector<std::string> output_names_;
};

/**
 * Resolves a possibly-negative ``axis`` against a tensor of rank
 * ``rank`` and returns the non-negative axis in ``[0, rank)``. Throws
 * ``std::invalid_argument`` with a message that mentions ``op_name``
 * when the axis is out of range.
 */
int64_t ResolveAxis(int64_t axis, std::size_t rank, const std::string &op_name);

/**
 * Builds a rank-0 (scalar) INT64 tensor named ``name`` holding ``v``. Used to
 * bind a control-flow subgraph's per-iteration scalar formal inputs (e.g.
 * ``Loop``'s ``iter_num``). When ``allocator`` is non-null the returned
 * tensor stores its bytes in an allocator-owned ``RawBuffer``.
 */
Tensor MakeInt64Scalar(const std::string &name, int64_t v, RawBufferAllocator *allocator);

/**
 * Builds a rank-0 (scalar) BOOL tensor named ``name`` holding ``v``. Used to
 * bind a control-flow subgraph's per-iteration scalar formal inputs (e.g.
 * ``Loop``'s ``cond_in``). When ``allocator`` is non-null the returned
 * tensor stores its bytes in an allocator-owned ``RawBuffer``.
 */
Tensor MakeBoolScalar(const std::string &name, bool v, RawBufferAllocator *allocator);

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

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
