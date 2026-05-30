// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_reduction.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``reduction`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace reduction {

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ReduceSum`` node
 * and stores it in ``ctx``.
 *
 * ``ReduceSum`` reduces the input tensor along a set of axes. The
 * output dtype always matches the input dtype (type constraint
 * ``T``); the output shape is the input shape with the reduced axes
 * either dropped (``keepdims=0``) or replaced by ``1``
 * (``keepdims=1``, the default).
 *
 * The way axes are specified depends on the opset:
 *
 *   - opset < 13: ``axes`` is a repeated INTS attribute. If absent
 *     every dimension is reduced.
 *   - opset >= 13: ``axes`` is an optional second input. When the
 *     input is missing or empty the behaviour is controlled by the
 *     ``noop_with_empty_axes`` attribute (default ``0``): ``0`` means
 *     reduce all axes, ``1`` means identity (no axes reduced).
 *
 *     When the ``axes`` input is provided this function looks at the
 *     :cpp:func:`OptimTensor::ValueAsShape` annotation of ``axes`` to
 *     identify the reduced axes. If the annotation is missing the
 *     output rank can still be inferred when ``keepdims=1`` (same
 *     rank as the input, with each previously-known dimension kept
 *     as is and reduced positions left symbolic) or when the number
 *     of axes is known via the shape of ``axes``
 *     (``keepdims=0``: output rank = input rank − number of axes,
 *     with every dimension marked symbolic).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``data`` and, when provided and non-empty, ``axes``.
 *              On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``ReduceSum`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"ReduceSum"``
 *              and ``node`` must declare at least one output.
 * @param data  Name of the data input value to read from ``ctx``.
 *              Must be present in ``ctx``.
 * @param axes  Name of the axes input value to read from ``ctx`` for
 *              opset >= 13, or ``nullptr`` / empty string when the
 *              axes input is omitted (use the attribute / default
 *              "reduce all" behaviour). For opset < 13 the value is
 *              ignored and the ``axes`` attribute is consulted.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"ReduceSum"``, if ``node`` has no output, or if an axis
 *         value is out of range.
 * @throws std::out_of_range     if ``data`` (or ``axes`` when given)
 *         is not present in ``ctx``.
 */
void ComputeShapeReduceSum(ShapesContext &ctx, const NodeProto &node, const char *data,
                           const char *axes);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ReduceMax`` node.
 * Shape/attribute semantics are the same as :cpp:func:`ComputeShapeReduceSum`
 * and the output dtype matches the input dtype.
 */
void ComputeShapeReduceMax(ShapesContext &ctx, const NodeProto &node, const char *data,
                           const char *axes);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ReduceMin`` node.
 * Shape/attribute semantics are the same as :cpp:func:`ComputeShapeReduceSum`
 * and the output dtype matches the input dtype.
 */
void ComputeShapeReduceMin(ShapesContext &ctx, const NodeProto &node, const char *data,
                           const char *axes);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``ArgMax`` or
 * ``ArgMin`` node and stores it in ``ctx``.
 *
 * ``ArgMax``/``ArgMin`` reduce the input tensor along a single ``axis``
 * (attribute, default ``0``; accepts negative values from opset 11). The
 * output dtype is always ``tensor(int64)`` (independent of the input
 * dtype); the output shape is the input shape with the reduced axis
 * either dropped (``keepdims=0``) or replaced by ``1`` (``keepdims=1``,
 * the default).
 *
 * The ``select_last_index`` attribute introduced in opset 12 does not
 * affect the output shape and is therefore ignored here.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``data``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``ArgMax`` or ``ArgMin`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be either
 *              ``"ArgMax"`` or ``"ArgMin"`` and ``node`` must declare at
 *              least one output.
 * @param data  Name of the data input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"ArgMax"``
 *         or ``"ArgMin"``, if ``node`` has no output, if the input has
 *         rank 0, or if the ``axis`` attribute is out of range.
 * @throws std::out_of_range     if ``data`` is not present in ``ctx``.
 */
void ComputeShapeArgReduce(ShapesContext &ctx, const NodeProto &node, const char *data);

} // namespace reduction
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
