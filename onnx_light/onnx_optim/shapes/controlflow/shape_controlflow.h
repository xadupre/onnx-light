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

/**
 * Computes the output :cpp:class:`OptimTensor` descriptors of a
 * ``Loop`` node and stores them in ``ctx``.
 *
 * ``Loop`` (since opset 1 in the ``ai.onnx`` domain) takes ``2 + N``
 * inputs ``(M, cond, v_initial[0..N-1])`` and yields ``N + K``
 * outputs: ``N`` final loop-carried dependency values followed by
 * ``K`` scan outputs. The loop ``body`` subgraph declares
 * ``2 + N`` inputs ``(iter_num, cond_in, v_in[0..N-1])`` and
 * ``1 + N + K`` outputs ``(cond_out, v_out[0..N-1], scan_out[0..K-1])``.
 *
 * Shape inference walks the ``body`` subgraph by calling
 * :cpp:func:`ComputeShapes` on a copy of ``ctx`` augmented with
 * descriptors for ``iter_num`` (INT64 scalar), ``cond_in`` (BOOL
 * scalar) and ``v_in[i]`` (taken from the matching ``v_initial[i]``
 * outer descriptor). The output descriptors are then derived as:
 *
 *   - the ``i``-th loop-carried output adopts the element dtype of
 *     ``v_out[i]`` from the body; the shape is kept when it matches
 *     ``v_initial[i]`` and is left fully symbolic
 *     (``Loop_<output_name>_d<j>``) otherwise to model the fact that
 *     the body may produce a shape that differs from the initial one;
 *   - the ``k``-th scan output adopts the element dtype of
 *     ``scan_out[k]`` from the body and its shape is the body's
 *     scan-output shape prefixed by a symbolic leading axis named
 *     ``Loop_<output_name>_d0`` (the trip count, generally unknown
 *     statically).
 *
 * @param ctx   In/out context. Must already contain entries for every
 *              non-empty input of ``node`` and for every outer-scope
 *              value referenced from the body; on return it also
 *              contains entries for every non-empty output declared by
 *              ``node``.
 * @param node  The ``Loop`` ``NodeProto`` whose outputs should be
 *              described. ``node.op_type()`` must be ``"Loop"`` and
 *              ``node`` must declare at least two inputs (``M`` and
 *              ``cond``, either of which may be omitted via an empty
 *              input name).
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Loop"``, if ``node`` has fewer than two inputs, if the
 *         ``body`` attribute is missing or not a ``GraphProto``, if
 *         the body does not declare ``2 + N`` inputs and
 *         ``1 + N + K`` outputs consistent with the node's input/
 *         output arity, or if the body's loop-carried output dtypes
 *         do not match the corresponding ``v_initial`` dtypes.
 */
void ComputeShapeLoop(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` descriptors of a
 * ``Scan`` node and stores them in ``ctx``.
 *
 * ``Scan`` iterates over ``M`` ``scan_input`` tensors and produces ``K``
 * ``scan_output`` tensors, optionally maintaining ``N`` loop-carried
 * state variables. The node has ``N + M`` inputs and ``N + K`` outputs.
 * Its ``body`` subgraph declares ``N + M`` inputs and ``N + K`` outputs:
 *
 *   - body input ``i`` for ``i < N`` is the current value of the ``i``-th
 *     state variable; for ``i >= N`` it is the per-iteration slice of the
 *     ``(i - N)``-th ``scan_input`` (rank equals the scan input's rank
 *     minus one for opset >= 9 when the scan axis is removed, and minus
 *     one for opset 8 which always has axis 0).
 *   - body output ``i`` for ``i < N`` is the updated state variable; for
 *     ``i >= N`` it is the per-iteration scan output element.
 *
 * The output descriptors are derived as:
 *
 *   - the ``i``-th state output (for ``i < N``) adopts the body's
 *     ``v_out[i]`` element dtype and the shape of ``v_initial[i]`` when
 *     they agree, otherwise the dtype/shape produced by the body;
 *   - the ``k``-th scan output adopts the body's scan-output element
 *     dtype and shape, prefixed by a leading axis of length equal to the
 *     scan dimension of the first ``scan_input`` (or a symbolic
 *     placeholder when that dimension is unknown).
 *
 * @param ctx   In/out context. Must already contain entries for every
 *              non-empty input of ``node`` and for every outer-scope
 *              value referenced from the body; on return it also
 *              contains entries for every non-empty output declared by
 *              ``node``.
 * @param node  The ``Scan`` ``NodeProto`` whose outputs should be
 *              described. ``node.op_type()`` must be ``"Scan"``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Scan"``,
 *         if the ``body`` attribute is missing or not a ``GraphProto``,
 *         if the ``num_scan_inputs`` attribute is missing or non-positive,
 *         or if the body's arity is inconsistent with ``node``.
 */
void ComputeShapeScan(ShapesContext &ctx, const NodeProto &node);

} // namespace controlflow
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
