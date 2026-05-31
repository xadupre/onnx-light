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

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Cast`` node and
 * stores it in ``ctx``.
 *
 * ``Cast`` produces an output whose shape is identical to the shape of
 * its single input and whose element type is given by the required
 * integer attribute ``to`` (a ``TensorProto::DataType`` value). The
 * other optional attributes (``saturate``, ``round_mode``) do not
 * affect the output shape or dtype and are therefore not inspected by
 * this function.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Cast`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Cast"``,
 *              ``node`` must declare at least one input and at least
 *              one output and must carry the required ``to``
 *              attribute.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Cast"``, if ``node`` has no input or output, if the
 *         ``to`` attribute is missing, or if its value does not map to
 *         a supported :cpp:enum:`TensorType`.
 * @throws std::out_of_range     if the input name is missing from
 *         ``ctx``.
 */
void ComputeShapeCast(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``CastLike`` node and
 * stores it in ``ctx``.
 *
 * ``CastLike`` produces an output whose shape is identical to the shape of
 * its first input (``input``) and whose element type is taken from the
 * second input (``target_type``); the values of ``target_type`` are
 * ignored. The optional attributes (``saturate``, ``round_mode``) do not
 * affect the output shape or dtype and are therefore not inspected by this
 * function.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` and ``node.input(1)``. On return it also
 *              contains an entry for ``node.output(0)``.
 * @param node  The ``CastLike`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"CastLike"``,
 *              ``node`` must declare two inputs and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"CastLike"``,
 *         if ``node`` has fewer than two inputs or no output, or if the
 *         dtype of ``target_type`` is :cpp:enum:`TensorType::kUndefined`.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeCastLike(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Reshape`` node
 * and stores it in ``ctx``.
 *
 * ``Reshape`` takes a ``data`` tensor and a 1-D int64 ``shape`` tensor
 * whose values describe the desired output shape. The output dtype is
 * the dtype of ``data`` (type constraint ``T``). The output shape is
 * derived element-by-element from the target shape:
 *
 *   - a positive value is used verbatim;
 *   - ``0`` means "copy from the input ``data`` shape at the same
 *     index", unless the ``allowzero`` attribute is set to ``1`` (in
 *     which case ``0`` is honoured literally);
 *   - exactly one ``-1`` is allowed; the corresponding dimension is
 *     inferred so that the total number of elements is preserved
 *     (when ``data`` is fully known and the other dims are concrete);
 *   - symbolic target dims are forwarded as symbolic output dims.
 *
 * Shape values are read from the ``shape`` input's
 * :cpp:func:`OptimTensor::ValueAsShape` annotation (populated for
 * small constants, e.g. by :cpp:func:`ComputeShapeConstant`). When
 * that annotation is missing the output rank is taken from the static
 * shape of the ``shape`` input (its single dimension, when concrete)
 * and every output dim is left symbolic. When the rank itself is
 * unknown the output is left as a fully-symbolic rank-1 tensor.
 *
 * @param ctx   In/out context. Must contain entries for
 *              ``node.input(0)`` (``data``) and ``node.input(1)``
 *              (``shape``). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Reshape`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Reshape"``,
 *              ``node`` must declare two inputs and at least one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Reshape"``, if ``node`` has fewer than two inputs or no
 *         output, if the target shape contains more than one ``-1``,
 *         contains a value strictly less than ``-1``, if a ``0`` entry
 *         (with ``allowzero == 0``) refers to a position outside the
 *         input rank, or if a ``-1`` cannot be reconciled with the
 *         input's element count.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeReshape(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Expand`` node
 * and stores it in ``ctx``.
 *
 * ``Expand`` broadcasts its first input (``input``) to the shape
 * described by its second input (``shape``), following the ONNX
 * numpy-style broadcasting rules. The output dtype matches the dtype
 * of ``input`` (type constraint ``T``).
 *
 * Shape values are read from the ``shape`` input's
 * :cpp:func:`OptimTensor::ValueAsShape` annotation (populated for
 * small constants). When that annotation is present the output shape
 * is computed as ``BroadcastShapes(input.shape, target)``. When it is
 * absent the output rank is taken from the static shape of the
 * ``shape`` input (its single dimension, when concrete) and every
 * output dim is left symbolic.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` (``input``) and ``node.input(1)``
 *              (``shape``). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Expand`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Expand"``,
 *              ``node`` must declare two inputs and at least one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Expand"``, if ``node`` has fewer than two inputs or no
 *         output, or if the two shapes are not broadcast-compatible.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeExpand(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Transpose`` node
 * and stores it in ``ctx``.
 *
 * ``Transpose`` permutes the axes of its input tensor according to the
 * optional ``perm`` attribute. When ``perm`` is absent, axes are reversed.
 * The output dtype matches the input dtype.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Transpose`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Transpose"``,
 *              ``node`` must declare one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Transpose"``,
 *         if ``node`` has no input or output, if ``perm`` length differs from
 *         input rank, if ``perm`` contains an out-of-range value, or if it
 *         contains duplicates.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeTranspose(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``AffineGrid`` node
 * and stores it in ``ctx``.
 *
 * ``AffineGrid`` produces a flow field of sampling coordinates from a
 * batch of affine matrices ``theta`` and a target ``size``. The output
 * dtype matches ``theta``'s dtype (type constraint ``T1``).
 *
 * The output shape is derived as follows:
 *
 *   - If ``size`` exposes a value via :cpp:func:`OptimTensor::ValueAsShape`
 *     (typically because it is a Constant), the spatial output dims
 *     ``(H, W)`` for 2-D / ``(D, H, W)`` for 3-D are taken verbatim from
 *     ``size`` and the rank of the output is fully known. ``size`` must
 *     have 4 (2-D) or 5 (3-D) entries.
 *   - Otherwise the rank of ``theta`` determines whether the output is
 *     2-D (``theta`` shape ``(N, 2, 3)``) or 3-D (``theta`` shape
 *     ``(N, 3, 4)``), and the spatial dims are left symbolic.
 *
 * The leading batch dim ``N`` is taken from ``theta[0]``; the final
 * inner dim is the constant ``2`` (2-D) or ``3`` (3-D).
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``theta`` and ``size``. On return it also contains an
 *              entry for ``node.output(0)``.
 * @param node  The ``AffineGrid`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"AffineGrid"``,
 *              ``node`` must declare two inputs and at least one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"AffineGrid"``, if ``node`` has fewer than two inputs or no
 *         output, if ``theta`` is not rank 3, if its inner dims are
 *         neither ``(2, 3)`` nor ``(3, 4)``, or if a known ``size``
 *         input has a length other than 4 or 5.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeAffineGrid(ShapesContext &ctx, const NodeProto &node);

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
