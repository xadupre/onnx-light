// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_math.h
 * @brief Shape-inference functions for ONNX operators in the ``math`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Abs`` node and
 * stores it in ``ctx``.
 *
 * ``Abs`` is element-wise and unary in every revision of its schema
 * (v1, v6, v13 — later revisions only widen the accepted dtype set),
 * so the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Abs`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Abs"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Abs"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAbs(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Add`` node and
 * stores it in ``ctx``.
 *
 * ``Add`` is element-wise and binary, with numpy-style multidirectional
 * broadcasting between its two operands (since opset 7; earlier
 * revisions had an explicit ``broadcast`` attribute but the shape
 * propagation rules are identical when broadcasting is enabled, which
 * onnx-light assumes). The output dtype matches the input dtype (both
 * operands share the same type via the ``T`` type constraint) and the
 * output shape is the broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Add`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Add"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Add"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeAdd(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Acos`` node and
 * stores it in ``ctx``.
 *
 * ``Acos`` is element-wise and unary in every revision of its schema
 * (v7, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Acos`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Acos"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Acos"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAcos(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Acosh`` node and
 * stores it in ``ctx``.
 *
 * ``Acosh`` is element-wise and unary in every revision of its schema
 * (v9, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Acosh`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Acosh"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Acosh"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAcosh(ShapesContext &ctx, const NodeProto &node, const char *x);

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
