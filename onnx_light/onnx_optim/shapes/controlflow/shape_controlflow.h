// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_controlflow.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``control flow`` family (``If``, ``Loop``, ``Scan``, ...).
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace controlflow {

/**
 * Computes the output :cpp:class:`OptimTensor` descriptors of an
 * ``If`` node and stores them in ``ctx``.
 *
 * ``If`` (since opset 1 in the ``ai.onnx`` domain) selects one of two
 * subgraphs depending on the boolean ``cond`` input and yields the
 * outputs of the selected subgraph as its own outputs. Both
 * subgraphs must declare the same number of outputs as the ``If``
 * node and the corresponding outputs must be type-compatible.
 *
 * Shape inference walks both ``then_branch`` and ``else_branch``
 * sub-graphs by calling :cpp:func:`ComputeShapes` on a copy of ``ctx``
 * (so that outer-scope values referenced by the sub-graph remain
 * visible) and then merges the resulting per-output descriptors:
 *
 *   - the element dtype is kept when both branches agree and is set
 *     to :cpp:enumerator:`TensorType::kUndefined` otherwise;
 *   - the shape is kept verbatim when both branches agree (same rank
 *     and identical dimensions); when the ranks match but some
 *     dimensions differ, those dimensions are replaced by a symbolic
 *     placeholder string of the form
 *     ``"If_<output_name>_d<i>"``; rank mismatches between the two
 *     branches are rejected with ``std::invalid_argument``.
 *
 * @param ctx   In/out context. Must already contain entries for every
 *              non-empty input of ``node`` and for every outer-scope
 *              value referenced from the sub-graphs; on return it also
 *              contains entries for every non-empty output declared by
 *              ``node``.
 * @param node  The ``If`` ``NodeProto`` whose outputs should be
 *              described. ``node.op_type()`` must be ``"If"`` and
 *              ``node`` must declare exactly one input and at least
 *              one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"If"``,
 *         if ``node`` has no output, if the ``then_branch`` or
 *         ``else_branch`` attribute is missing or not a ``GraphProto``,
 *         if a sub-graph does not declare the same number of
 *         outputs as ``node``, or if the two branches produce outputs
 *         with mismatching rank.
 */
void ComputeShapeIf(ShapesContext &ctx, const NodeProto &node);

} // namespace controlflow
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
