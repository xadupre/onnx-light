// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_quantization.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``quantization`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace quantization {

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``QuantizeLinear``
 * node and stores it in ``ctx``.
 *
 * ``QuantizeLinear`` produces an output ``y`` that always has the same
 * shape as the input ``x``. The output element type is resolved as
 * follows:
 *
 *   - when ``y_zero_point`` is supplied (third input, non-empty name)
 *     the output dtype is the dtype of ``y_zero_point``;
 *   - otherwise, when the ``output_dtype`` integer attribute is set
 *     (opset 23+), it is interpreted as a ``TensorProto::DataType`` and
 *     mapped to the matching :cpp:enum:`TensorType`;
 *   - otherwise the output dtype defaults to ``uint8``.
 *
 * The ``axis``, ``saturate``, ``block_size``, and ``precision``
 * attributes do not affect the output shape and are therefore not
 * inspected by this function.
 *
 * @param ctx           In/out context. Must already contain entries for
 *                      ``x``, ``y_scale``, and ``y_zero_point`` when the
 *                      latter is provided; on return it also contains an
 *                      entry for ``node.output(0)``.
 * @param node          The ``QuantizeLinear`` ``NodeProto`` whose output
 *                      should be described. ``node.op_type()`` must be
 *                      ``"QuantizeLinear"`` and ``node`` must declare at
 *                      least one output.
 * @param x             Name of the input value to read from ``ctx``.
 *                      Must be present in ``ctx``.
 * @param y_zero_point  Name of the ``y_zero_point`` input value, or
 *                      ``nullptr`` / empty string when the input is
 *                      omitted. When non-empty it must be present in
 *                      ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"QuantizeLinear"``, if ``node`` has no output, or if
 *         ``output_dtype`` is set to a value that does not map to a
 *         supported :cpp:enum:`TensorType`.
 * @throws std::out_of_range     if ``x`` (or ``y_zero_point`` when
 *         non-empty) is not present in ``ctx``.
 */
void ComputeShapeQuantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x,
                                const char *y_zero_point);

} // namespace quantization
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
