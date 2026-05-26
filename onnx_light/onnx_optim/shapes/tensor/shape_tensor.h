// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_tensor.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``tensor`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Concat`` node
 * and stores it in ``ctx``.
 *
 * ``Concat`` concatenates a variadic list of input tensors along the
 * axis specified by the ``axis`` attribute. All inputs must share the
 * same rank and the same dimension sizes on every axis other than the
 * concatenation axis. The output dtype always matches the dtype of
 * the first input (type constraint ``T``); the output shape is:
 *
 *   - on the concatenation axis: the sum of all input dimensions when
 *     every input dimension on that axis is a concrete integer;
 *     otherwise a fresh symbolic dimension;
 *   - on every other axis: the merged dimension between all inputs
 *     (concrete dimensions must match across inputs, otherwise an
 *     exception is thrown; a concrete value overrides a symbolic
 *     one).
 *
 * The ``axis`` attribute can be negative, in which case it is
 * interpreted as ``axis + rank``. When the attribute is missing the
 * default of ``1`` (the opset 1 default) is used.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              every name in ``node.input``. On return it also
 *              contains an entry for ``node.output(0)``.
 * @param node  The ``Concat`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Concat"``,
 *              ``node`` must declare at least one input and at least
 *              one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Concat"``, if ``node`` has no input or output, if the
 *         inputs have different ranks, if their non-concat dimensions
 *         disagree, if the resolved axis is out of range, or if the
 *         input dtypes differ.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeConcat(ShapesContext &ctx, const NodeProto &node);

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
