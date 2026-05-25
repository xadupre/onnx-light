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

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
