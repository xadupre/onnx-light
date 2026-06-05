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

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``DequantizeLinear``
 * node and stores it in ``ctx``.
 *
 * ``DequantizeLinear`` produces an output ``y`` that always has the same
 * shape as the input ``x``. The output element type is resolved as
 * follows:
 *
 *   - when the ``output_dtype`` integer attribute is set (opset 23+) it is
 *     interpreted as a ``TensorProto::DataType`` and mapped to the
 *     matching :cpp:enum:`TensorType`;
 *   - otherwise, the output dtype is the dtype of ``x_scale``.
 *
 * The ``axis`` and ``block_size`` attributes do not affect the output
 * shape and are therefore not inspected by this function.
 *
 * @param ctx           In/out context. Must already contain entries for
 *                      ``x`` and ``x_scale``; on return it also contains
 *                      an entry for ``node.output(0)``.
 * @param node          The ``DequantizeLinear`` ``NodeProto`` whose
 *                      output should be described. ``node.op_type()``
 *                      must be ``"DequantizeLinear"`` and ``node`` must
 *                      declare at least one output.
 * @param x             Name of the input value to read from ``ctx``.
 *                      Must be present in ``ctx``.
 * @param x_scale       Name of the ``x_scale`` input value, used to
 *                      derive the default output element type. Must be
 *                      present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"DequantizeLinear"``, if ``node`` has no output, or if
 *         ``output_dtype`` is set to a value that does not map to a
 *         supported :cpp:enum:`TensorType`.
 * @throws std::out_of_range     if ``x`` or ``x_scale`` is not present
 *         in ``ctx``.
 */
void ComputeShapeDequantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x,
                                  const char *x_scale);

/**
 * Computes the output :cpp:class:`OptimTensor` entries of a
 * ``DynamicQuantizeLinear`` node and stores them in ``ctx``.
 *
 * ``DynamicQuantizeLinear`` produces three outputs (since opset 11 in the
 * ai.onnx domain):
 *
 *   - ``y`` — same shape as the input ``x``, with dtype ``uint8``;
 *   - ``y_scale`` — scalar (rank 0) ``float``;
 *   - ``y_zero_point`` — scalar (rank 0) ``uint8``.
 *
 * The operator takes no attributes that affect output shapes.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``; on
 *              return it also contains entries for the (up to three)
 *              non-empty outputs of ``node``.
 * @param node  The ``DynamicQuantizeLinear`` ``NodeProto`` whose outputs
 *              should be described. ``node.op_type()`` must be
 *              ``"DynamicQuantizeLinear"`` and ``node`` must declare at
 *              least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"DynamicQuantizeLinear"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeDynamicQuantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``QLinearConv`` node and
 * stores it in ``ctx``.
 *
 * The output shape rule matches :cpp:func:`ComputeShapeConv` applied to the
 * quantized inputs ``x`` (input 0) and ``w`` (input 3). The output dtype is
 * the dtype of ``y_zero_point`` (input 7).
 *
 * @param ctx           In/out context. Must already contain entries for
 *                      ``x``, ``w``, and ``y_zero_point``; on return it
 *                      also contains an entry for ``node.output(0)``.
 * @param node          The ``QLinearConv`` ``NodeProto`` whose output should
 *                      be described. ``node.op_type()`` must be
 *                      ``"QLinearConv"`` and ``node`` must declare at least
 *                      one output.
 * @param x             Name of the input data value (rank >= 3) in ``ctx``.
 * @param w             Name of the weight value (rank >= 3) in ``ctx``.
 * @param y_zero_point  Name of the ``y_zero_point`` input value, used to
 *                      derive the output element type. Must be present in
 *                      ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"QLinearConv"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x``, ``w``, or ``y_zero_point`` is not
 *         present in ``ctx``.
 */
void ComputeShapeQLinearConv(ShapesContext &ctx, const NodeProto &node, const char *x,
                             const char *w, const char *y_zero_point);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``QLinearMatMul`` node
 * and stores it in ``ctx``.
 *
 * The output shape rule matches :cpp:func:`ComputeShapeMatMul` applied to
 * the quantized inputs ``a`` (input 0) and ``b`` (input 3). The output
 * dtype is the dtype of ``y_zero_point`` (input 7).
 *
 * @param ctx           In/out context. Must already contain entries for
 *                      ``a``, ``b``, and ``y_zero_point``; on return it
 *                      also contains an entry for ``node.output(0)``.
 * @param node          The ``QLinearMatMul`` ``NodeProto`` whose output
 *                      should be described. ``node.op_type()`` must be
 *                      ``"QLinearMatMul"`` and ``node`` must declare at
 *                      least one output.
 * @param a             Name of the input ``a`` value (rank >= 1) in ``ctx``.
 * @param b             Name of the input ``b`` value (rank >= 1) in ``ctx``.
 * @param y_zero_point  Name of the ``y_zero_point`` input value, used to
 *                      derive the output element type. Must be present in
 *                      ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"QLinearMatMul"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``a``, ``b``, or ``y_zero_point`` is
 *         not present in ``ctx``.
 */
void ComputeShapeQLinearMatMul(ShapesContext &ctx, const NodeProto &node, const char *a,
                               const char *b, const char *y_zero_point);

} // namespace quantization
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
