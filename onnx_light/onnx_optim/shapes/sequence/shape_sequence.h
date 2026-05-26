// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_sequence.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``sequence`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

/**
 * Computes the output :cpp:class:`OptimSequence` of a
 * ``SequenceConstruct`` node and stores it in ``ctx``.
 *
 * ``SequenceConstruct`` (since opset 11 in the ``ai.onnx`` domain) takes
 * ``N >= 1`` tensor inputs that share the same element type and
 * produces a single tensor-sequence output of length ``N``. The element
 * dtype of the output sequence is the common dtype of the inputs; the
 * ONNX schema does not require the inputs to share a common shape, so
 * the output :cpp:class:`OptimSequence` records one
 * :cpp:class:`OptimShape` per input verbatim (see
 * :cpp:func:`OptimSequence::ElemShapes`).
 *
 * When called with zero inputs, the output sequence has length ``0``,
 * an unknown element dtype (:cpp:enumerator:`TensorType::kUndefined`)
 * and an empty per-element shapes vector.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimTensor` entry for every named input of
 *              ``node``; on return it also contains an
 *              :cpp:class:`OptimSequence` entry for ``node.output(0)``.
 * @param node  The ``SequenceConstruct`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be
 *              ``"SequenceConstruct"`` and ``node`` must declare at
 *              least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceConstruct"``, if ``node`` has no output, or if the
 *         input tensors do not share a common element dtype.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeSequenceConstruct(ShapesContext &ctx, const NodeProto &node);

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
