// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_logical.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``logical`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace logical {

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``And`` node and
 * stores it in ``ctx``.
 *
 * ``And`` is the logical, element-wise AND of two boolean operands
 * with numpy-style multidirectional broadcasting (since opset 7;
 * earlier revisions used an explicit ``broadcast`` attribute but the
 * shape propagation rules are identical when broadcasting is enabled,
 * which onnx-light assumes). The output dtype is always
 * :cpp:enumerator:`TensorType::kBool` and the output shape is the
 * broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``And`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"And"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"And"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeAnd(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Or`` node and
 * stores it in ``ctx``.
 *
 * ``Or`` is the logical, element-wise OR of two boolean operands with
 * numpy-style multidirectional broadcasting (since opset 7). The output
 * dtype is always :cpp:enumerator:`TensorType::kBool` and the output
 * shape is the broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Or`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Or"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Or"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeOr(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Xor`` node and
 * stores it in ``ctx``.
 *
 * ``Xor`` is the logical, element-wise XOR of two boolean operands with
 * numpy-style multidirectional broadcasting (since opset 7). The output
 * dtype is always :cpp:enumerator:`TensorType::kBool` and the output
 * shape is the broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Xor`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Xor"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Xor"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeXor(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Greater`` node and
 * stores it in ``ctx``.
 *
 * ``Greater`` is the element-wise ``A > B`` comparison of two numeric
 * operands with numpy-style multidirectional broadcasting (since
 * opset 7; opset 1 only supported broadcasting via an explicit
 * ``broadcast`` attribute but the shape propagation rules are
 * identical when broadcasting is enabled, which onnx-light assumes).
 * The output dtype is always :cpp:enumerator:`TensorType::kBool` and
 * the output shape is the broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Greater`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Greater"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Greater"``, if ``node`` has no output, or if the two
 *         input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeGreater(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Less`` node and
 * stores it in ``ctx``.
 *
 * ``Less`` is the element-wise ``A < B`` comparison of two numeric
 * operands with numpy-style multidirectional broadcasting (since
 * opset 7; opset 1 only supported broadcasting via an explicit
 * ``broadcast`` attribute but the shape propagation rules are
 * identical when broadcasting is enabled, which onnx-light assumes).
 * The output dtype is always :cpp:enumerator:`TensorType::kBool` and
 * the output shape is the broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Less`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Less"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Less"``, if ``node`` has no output, or if the two input
 *         shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeLess(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``GreaterOrEqual`` node
 * and stores it in ``ctx``.
 *
 * ``GreaterOrEqual`` is the element-wise ``A >= B`` comparison of two
 * numeric operands with numpy-style multidirectional broadcasting (since
 * opset 12). The output dtype is always
 * :cpp:enumerator:`TensorType::kBool` and the output shape is the
 * broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``GreaterOrEqual`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"GreaterOrEqual"`` and ``node`` must declare at least
 *              one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"GreaterOrEqual"``, if ``node`` has no output, or if the
 *         two input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeGreaterOrEqual(ShapesContext &ctx, const NodeProto &node, const char *a,
                                const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Equal`` node and
 * stores it in ``ctx``.
 *
 * ``Equal`` is the element-wise ``A == B`` comparison of two operands
 * with numpy-style multidirectional broadcasting (since opset 7;
 * opset 1 only supported broadcasting via an explicit ``broadcast``
 * attribute but the shape propagation rules are identical when
 * broadcasting is enabled, which onnx-light assumes). The output
 * dtype is always :cpp:enumerator:`TensorType::kBool` and the output
 * shape is the broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Equal`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Equal"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Equal"``, if ``node`` has no output, or if the two
 *         input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeEqual(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Where`` node and
 * stores it in ``ctx``.
 *
 * ``Where`` returns elements from ``x`` or ``y`` depending on ``condition``.
 * The output dtype is the dtype of ``x``/``y`` and the output shape is the
 * multidirectional broadcast of ``condition``, ``x`` and ``y``.
 */
void ComputeShapeWhere(ShapesContext &ctx, const NodeProto &node, const char *condition,
                       const char *x, const char *y);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BitwiseAnd`` node
 * (opset 18) and stores it in ``ctx``.
 *
 * ``BitwiseAnd`` is element-wise with numpy-style multidirectional
 * broadcasting; both inputs must share the same integer dtype and the
 * output dtype equals that input dtype.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``BitwiseAnd`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"BitwiseAnd"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"BitwiseAnd"``, if ``node`` has no output, or if the two
 *         input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeBitwiseAnd(ShapesContext &ctx, const NodeProto &node, const char *a,
                            const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BitwiseOr`` node
 * (opset 18) and stores it in ``ctx``. Shape/type semantics match
 * :cpp:func:`ComputeShapeBitwiseAnd`.
 */
void ComputeShapeBitwiseOr(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BitwiseXor`` node
 * (opset 18) and stores it in ``ctx``. Shape/type semantics match
 * :cpp:func:`ComputeShapeBitwiseAnd`.
 */
void ComputeShapeBitwiseXor(ShapesContext &ctx, const NodeProto &node, const char *a,
                            const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BitwiseNot`` node
 * (opset 18) and stores it in ``ctx``.
 *
 * ``BitwiseNot`` is element-wise and unary: the output dtype and shape
 * match the input.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``;
 *              on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``BitwiseNot`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"BitwiseNot"``
 *              and ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"BitwiseNot"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is missing from ``ctx``.
 */
void ComputeShapeBitwiseNot(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BitShift`` node
 * (opset 11) and stores it in ``ctx``.
 *
 * ``BitShift`` is element-wise with numpy-style multidirectional
 * broadcasting; both inputs must share the same unsigned-integer dtype and
 * the output dtype equals that input dtype. The required ``direction``
 * attribute does not affect the output type or shape and is therefore not
 * inspected by this function.
 */
void ComputeShapeBitShift(ShapesContext &ctx, const NodeProto &node, const char *x,
                          const char *y);

} // namespace logical
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
