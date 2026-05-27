// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_inference.h
 * @brief Top-level shape-inference helpers that dispatch to the
 *        per-operator ``ComputeShape*`` functions of ``onnx_optim``.
 *
 * The per-operator functions (for example
 * :cpp:func:`onnx_optim::shapes::math::ComputeShapeAbs`) each take
 * their inputs by name. The helpers in this file walk a single
 * ``NodeProto`` or a topologically-sorted sequence of ``NodeProto``
 * (typically ``GraphProto::node()``), look up the op type and forward
 * the call to the matching ``ComputeShape*`` implementation.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Dispatches a single ``NodeProto`` to the matching per-operator
 * ``ComputeShape*`` function and stores the resulting output
 * :cpp:class:`OptimTensor` descriptors in ``ctx``.
 *
 * The dispatch table is keyed on ``node.op_type()``. Only operators
 * that ``onnx_optim`` currently knows about are accepted; unsupported
 * op types throw ``std::invalid_argument``. The node's input
 * descriptors are read from ``ctx`` by name (so every input must
 * already be present), and the output descriptors are inserted into
 * ``ctx`` under the names declared by ``node.output(i)``.
 *
 * The domain of ``node`` is checked: only nodes belonging to the
 * default ONNX domain (empty string or ``"ai.onnx"``) are supported.
 *
 * @param ctx   In/out context. Must already contain entries for every
 *              input referenced by ``node``; on return it also
 *              contains entries for every output declared by ``node``.
 * @param node  The ``NodeProto`` whose output shapes should be
 *              computed.
 *
 * @throws std::invalid_argument if ``node.domain()`` is not the
 *         default ONNX domain, if ``node.op_type()`` is not supported,
 *         or if a per-operator function rejects the node (for example
 *         when an expected input is missing).
 * @throws std::out_of_range     if any input referenced by ``node``
 *         is missing from ``ctx``.
 */
void ComputeShapeNode(ShapesContext &ctx, const NodeProto &node);

/**
 * Throws ``std::invalid_argument`` if any non-empty input name
 * declared by ``node`` is missing from ``ctx``. Empty input names
 * — used by ONNX to denote optional inputs that are not provided —
 * are skipped.
 *
 * @param ctx   Context whose entries are inspected.
 * @param node  Node whose inputs are checked.
 *
 * @throws std::invalid_argument if a required (non-empty) input is
 *         not present in ``ctx``.
 */
void CheckInputsAvailable(const ShapesContext &ctx, const NodeProto &node);

/**
 * Throws ``std::invalid_argument`` if any non-empty output name
 * declared by ``node`` already has an entry in ``ctx``. Empty output
 * names — used by ONNX to denote optional outputs that are not
 * produced — are skipped.
 *
 * This is intended as a precondition check for shape-inference
 * passes that build up ``ctx`` incrementally and should never
 * overwrite an existing descriptor.
 *
 * @param ctx   Context whose entries are inspected.
 * @param node  Node whose outputs are checked.
 *
 * @throws std::invalid_argument if a non-empty output is already
 *         present in ``ctx``.
 */
void CheckOutputsNotAvailable(const ShapesContext &ctx, const NodeProto &node);

/**
 * Runs :cpp:func:`ComputeShapeNode` on every node of ``nodes`` in
 * order. The sequence must be topologically sorted with respect to
 * data dependencies (as required by the ONNX specification for
 * ``GraphProto::node``), so that every input of a node has already
 * been described in ``ctx`` (either as a pre-existing graph
 * input/initializer or as the output of an earlier node in the
 * sequence) by the time the node is processed.
 *
 * @param ctx    In/out context. On entry it must contain descriptors
 *               for every graph input and initializer referenced by
 *               ``nodes``; on return it additionally contains
 *               descriptors for every output of every node in
 *               ``nodes``.
 * @param nodes  The list of nodes to process, in topological order.
 *
 * @throws std::invalid_argument if a node's domain is not the default
 *         ONNX domain, if its op type is not supported, or if a
 *         per-operator function rejects the node.
 * @throws std::out_of_range     if an input referenced by a node is
 *         not present in ``ctx`` by the time the node is processed.
 */
void ComputeShapes(ShapesContext &ctx, const utils::RepeatedProtoField<NodeProto> &nodes);

/**
 * Seeds ``ctx`` from the initializers and inputs of ``graph`` and
 * then runs :cpp:func:`ComputeShapes` on its nodes.
 *
 * For every entry in ``graph.initializer()`` an
 * :cpp:class:`OptimTensor` describing the initializer's element type
 * and shape is inserted in ``ctx`` (small 1-D integer initializers
 * also get a ``ValueAsShape`` annotation derived from their content,
 * mirroring :cpp:func:`ComputeShapeConstant`).
 *
 * For every entry in ``graph.input()`` whose name is not already
 * present in ``ctx`` (initializers take precedence), an
 * :cpp:class:`OptimTensor` describing the value's tensor type and
 * shape is inserted. Inputs declared as sequence/map/optional/sparse
 * types are skipped (``OptimTensor`` does not model these). Missing
 * or symbolic dimensions are preserved as :cpp:class:`OptimDim`
 * expressions: ``dim_value`` becomes a concrete integer dimension,
 * ``dim_param`` becomes a symbolic dimension with the same name, and
 * an unset dimension becomes a fresh ``"?"`` symbolic dimension.
 *
 * After running shape inference on the nodes, ``ctx`` is reconciled
 * with the entries in ``graph.value_info()``: entries whose name is
 * not yet in ``ctx`` are inserted from the proto, and entries that
 * are already in ``ctx`` are compared to the inferred descriptor.
 * A contradiction between the inferred value and the existing
 * ``value_info`` (incompatible dtype, rank, or any pair of concrete
 * integer dimensions) raises ``std::invalid_argument``. Symbolic
 * and concrete dimensions are treated as compatible at the same
 * position because one is simply more specific than the other.
 *
 * @param ctx    In/out context. On entry it may already contain
 *               outer-scope entries (for sub-graphs). On return it
 *               additionally contains descriptors for every graph
 *               initializer, every graph input (when not already
 *               present), every node output, and every
 *               ``value_info`` entry that was not previously known.
 * @param graph  The graph whose initializers, inputs and nodes are
 *               processed in topological order.
 *
 * @throws std::invalid_argument propagated from
 *         :cpp:func:`ComputeShapeNode`, or raised when an inferred
 *         descriptor contradicts an existing ``value_info`` entry.
 */
void ComputeShapeGraph(ShapesContext &ctx, const GraphProto &graph);

/**
 * Runs shape inference on ``model.graph()``.
 *
 * Records every ``(domain, version)`` pair in ``model.opset_import()``
 * in ``ctx`` via :cpp:func:`ShapesContext::SetOpsetVersion` and then
 * delegates to :cpp:func:`ComputeShapeGraph`. Any opset entry already
 * recorded in ``ctx`` is overwritten so that the values from the
 * ``ModelProto`` take precedence.
 *
 * @param ctx    In/out context. On return it contains the model's
 *               opset versions, the graph initializers, the graph
 *               inputs and every intermediate value computed by the
 *               graph nodes.
 * @param model  The model whose main graph is processed.
 *
 * @throws std::invalid_argument when ``model`` has no graph or when
 *         shape inference of the graph rejects a node.
 */
void ComputeShapeModel(ShapesContext &ctx, const ModelProto &model);

/**
 * Writes the shape and element-type descriptors stored in ``ctx``
 * back into ``graph`` so that the inferred information persists in
 * the proto representation.
 *
 * For every output declared by ``graph.output()`` whose name is
 * present in ``ctx`` (and whose ``OptimTensor`` has a known dtype),
 * the matching ``ValueInfoProto`` is updated to carry the inferred
 * tensor element type and ``TensorShapeProto``. Concrete integer
 * dimensions become ``dim_value`` entries and symbolic dimensions
 * become ``dim_param`` entries.
 *
 * For every other name in ``ctx.Tensors()`` (intermediate results) a
 * fresh ``ValueInfoProto`` is appended to ``graph.value_info()``.
 * Names that already correspond to a graph input, initializer or
 * existing ``value_info`` entry are skipped so the function never
 * overwrites authoritative type information already present in the
 * proto.
 *
 * Entries whose ``OptimTensor`` has :cpp:enumerator:`TensorType::kUndefined`
 * element type are skipped — there is no valid ``TensorProto::DataType``
 * to write for them.
 *
 * @param ctx    Context populated by a previous call to
 *               :cpp:func:`ComputeShapeGraph` /
 *               :cpp:func:`ComputeShapeModel`.
 * @param graph  In/out graph whose ``output`` and ``value_info``
 *               entries are updated in place.
 */
void ApplyInferredShapesToGraph(const ShapesContext &ctx, GraphProto &graph);

/**
 * Writes the shape and element-type descriptors stored in ``ctx``
 * back into ``model.graph()`` by delegating to
 * :cpp:func:`ApplyInferredShapesToGraph`.
 *
 * @param ctx    Context populated by a previous call to
 *               :cpp:func:`ComputeShapeModel`.
 * @param model  In/out model whose main graph is updated in place.
 *
 * @throws std::invalid_argument when ``model`` has no graph.
 */
void ApplyInferredShapesToModel(const ShapesContext &ctx, ModelProto &model);

/**
 * Convenience helper: runs :cpp:func:`ComputeShapeModel` on ``model``
 * and then :cpp:func:`ApplyInferredShapesToModel`, mutating ``model``
 * in place so that its ``graph.output()`` and ``graph.value_info()``
 * carry the inferred shapes.
 *
 * @param model  In/out model on which shape inference is run and
 *               whose proto is updated with the inferred results.
 *
 * @throws std::invalid_argument when ``model`` has no graph or when
 *         shape inference of the graph rejects a node.
 */
void InferShapesModel(ModelProto &model);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
