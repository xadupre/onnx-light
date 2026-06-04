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
 * Computes the output :cpp:class:`OptimTensor` of a ``Neg`` node and
 * stores it in ``ctx``.
 *
 * ``Neg`` is element-wise and unary in every revision of its schema
 * (v1, v6, v13 — later revisions only widen the accepted dtype set),
 * so the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Neg`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Neg"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Neg"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeNeg(ShapesContext &ctx, const NodeProto &node, const char *x);

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
 * Computes the output :cpp:class:`OptimTensor` of a ``Sub`` node and
 * stores it in ``ctx``.
 *
 * ``Sub`` is element-wise and binary, with numpy-style multidirectional
 * broadcasting between its two operands (since opset 7). The output
 * dtype matches the input dtype (both operands share the same type via
 * the ``T`` type constraint) and the output shape is the broadcast of
 * the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Sub`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Sub"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Sub"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeSub(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Mul`` node and
 * stores it in ``ctx``.
 *
 * ``Mul`` is element-wise and binary, with numpy-style multidirectional
 * broadcasting between its two operands (since opset 7). The output
 * dtype matches the input dtype (both operands share the same type via
 * the ``T`` type constraint) and the output shape is the broadcast of
 * the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Mul`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Mul"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Mul"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeMul(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Pow`` node and
 * stores it in ``ctx``.
 *
 * ``Pow`` is element-wise and binary, with numpy-style multidirectional
 * broadcasting between its two operands (since opset 7). The output
 * dtype matches the first input dtype and the output shape is the
 * broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Pow`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Pow"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Pow"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapePow(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``PRelu`` node and
 * stores it in ``ctx``.
 *
 * ``PRelu`` is element-wise binary, with the ``slope`` operand
 * unidirectionally broadcastable to the ``X`` operand (since opset 7).
 * The output dtype matches the input dtype (both operands share the same
 * type via the ``T`` type constraint) and the output shape is the
 * broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``x`` and ``slope``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``PRelu`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"PRelu"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the ``X`` input to read from ``ctx``.
 * @param slope Name of the ``slope`` input to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"PRelu"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``x`` or ``slope`` is missing
 *         from ``ctx``.
 */
void ComputeShapePRelu(ShapesContext &ctx, const NodeProto &node, const char *x, const char *slope);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Div`` node and
 * stores it in ``ctx``.
 *
 * ``Div`` is element-wise and binary, with numpy-style multidirectional
 * broadcasting between its two operands (since opset 7). The output
 * dtype matches the input dtype (both operands share the same type via
 * the ``T`` type constraint) and the output shape is the broadcast of
 * the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Div`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Div"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Div"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeDiv(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Mod`` node and
 * stores it in ``ctx``.
 *
 * ``Mod`` is element-wise and binary, with numpy-style multidirectional
 * broadcasting between its two operands (since opset 13). The output
 * dtype matches the input dtype (both operands share the same type via
 * the ``T`` type constraint) and the output shape is the broadcast of
 * the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Mod`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Mod"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Mod"``,
 *         if ``node`` has no output, or if the two input shapes are not
 *         broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeMod(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

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

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Asin`` node and
 * stores it in ``ctx``.
 *
 * ``Asin`` is element-wise and unary in every revision of its schema
 * (v7, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Asin`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Asin"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Asin"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAsin(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Asinh`` node and
 * stores it in ``ctx``.
 *
 * ``Asinh`` is element-wise and unary in every revision of its schema
 * (v9, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Asinh`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Asinh"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Asinh"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAsinh(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Atan`` node and
 * stores it in ``ctx``.
 *
 * ``Atan`` is element-wise and unary in every revision of its schema
 * (v7, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Atan`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Atan"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Atan"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAtan(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Atanh`` node and
 * stores it in ``ctx``.
 *
 * ``Atanh`` is element-wise and unary in every revision of its schema
 * (v9, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Atanh`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Atanh"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Atanh"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAtanh(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Cos`` node and
 * stores it in ``ctx``.
 *
 * ``Cos`` is element-wise and unary in every revision of its schema
 * (v7, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Cos`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Cos"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Cos"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeCos(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Cosh`` node and
 * stores it in ``ctx``.
 *
 * ``Cosh`` is element-wise and unary in every revision of its schema
 * (v9, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Cosh`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Cosh"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must
 *              be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Cosh"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeCosh(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Exp is element-wise unary: output dtype and shape match the input.
void ComputeShapeExp(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Erf is element-wise unary: output dtype and shape match the input.
void ComputeShapeErf(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Gemm`` node and
 * stores it in ``ctx``.
 *
 * ``Gemm`` computes ``Y = alpha * A' * B' + beta * C`` where the
 * transposition flags ``transA``/``transB`` are read from the node
 * attributes (both default to ``0``). A must be 2-D and B must be 2-D;
 * with ``transA=0`` A has shape ``(M, K)`` and with ``transA=1`` it has
 * shape ``(K, M)``. Similarly ``transB=0`` gives B shape ``(K, N)`` and
 * ``transB=1`` gives ``(N, K)``. The output Y has shape ``(M, N)`` and
 * the same dtype as A.
 *
 * @param ctx   In/out context. Must already contain entries for ``a``
 *              and ``b``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Gemm`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Gemm"`` and
 *              ``node`` must declare at least one output.
 * @param a     Name of the A input value to read from ``ctx``.
 * @param b     Name of the B input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Gemm"``,
 *         if ``node`` has no output, or if either A or B does not have
 *         rank 2.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeGemm(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``MatMul`` node and
 * stores it in ``ctx``.
 *
 * ``MatMul`` follows NumPy matmul rules:
 * - rank-1 x rank-1 -> scalar
 * - rank-2 x rank-2 -> matrix
 * - higher-rank prefixes are broadcast, then matrix multiply is applied on the
 *   trailing two dimensions.
 */
void ComputeShapeMatMul(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b);

/// Log is element-wise unary: output dtype and shape match the input.
void ComputeShapeLog(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Det`` node and stores
 * it in ``ctx``.
 *
 * ``Det`` takes one input of shape ``[*, M, M]`` (rank >= 2) and produces an
 * output of shape ``[*]`` containing the determinants of all input
 * submatrices. The output dtype matches the input dtype.
 */
void ComputeShapeDet(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Sigmoid is element-wise unary: output dtype and shape match the input.
void ComputeShapeSigmoid(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Swish is element-wise unary: output dtype and shape match the input.
void ComputeShapeSwish(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Softmax preserves dtype/shape and validates the axis attribute.
void ComputeShapeSoftmax(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Softplus is element-wise unary: output dtype and shape match the input.
void ComputeShapeSoftplus(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Softsign is element-wise unary: output dtype and shape match the input.
void ComputeShapeSoftsign(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output ``OptimTensor``(s) of a ``SoftmaxCrossEntropyLoss``
 * node and stores them in ``ctx``.
 *
 * Inputs are ``scores`` (shape ``(N, C)`` or ``(N, C, D1, ..., Dk)``),
 * ``labels`` (shape ``(N)`` or ``(N, D1, ..., Dk)``), and optionally
 * ``weights`` (shape ``(C)``). The first output ``output`` has the
 * loss shape: ``(N, D1, ..., Dk)`` when ``reduction = "none"`` and a
 * scalar otherwise. The optional second output ``log_prob`` has the
 * same shape and dtype as ``scores``.
 */
void ComputeShapeSoftmaxCrossEntropyLoss(ShapesContext &ctx, const NodeProto &node,
                                         const char *scores, const char *labels,
                                         const char *weights);

/**
 * Computes the output ``OptimTensor`` of a ``NegativeLogLikelihoodLoss``
 * node and stores it in ``ctx``.
 *
 * Inputs are ``input`` (shape ``(N, C)`` or ``(N, C, D1, ..., Dk)``),
 * ``target`` (shape ``(N)`` or ``(N, D1, ..., Dk)``), and optionally
 * ``weight`` (shape ``(C)``). The single output ``loss`` has shape
 * ``(N, D1, ..., Dk)`` when ``reduction = "none"`` and is a scalar
 * otherwise. The output dtype matches the dtype of ``input``.
 */
void ComputeShapeNegativeLogLikelihoodLoss(ShapesContext &ctx, const NodeProto &node,
                                           const char *input, const char *target,
                                           const char *weight);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Sin`` node and
 * stores it in ``ctx``.
 *
 * ``Sin`` is element-wise and unary in every revision of its schema
 * (v7, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 */
void ComputeShapeSin(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Sinh`` node and
 * stores it in ``ctx``.
 *
 * ``Sinh`` is element-wise and unary in every revision of its schema
 * (v9, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 */
void ComputeShapeSinh(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Sqrt`` node and
 * stores it in ``ctx``.
 *
 * ``Sqrt`` is element-wise and unary in every revision of its schema
 * (v1, v6, v13 — later revisions only widen the accepted dtype set),
 * so the output dtype and shape always match those of the input.
 */
void ComputeShapeSqrt(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Tan`` node and
 * stores it in ``ctx``.
 *
 * ``Tan`` is element-wise and unary in every revision of its schema
 * (v7, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 */
void ComputeShapeTan(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Tanh`` node and
 * stores it in ``ctx``.
 *
 * ``Tanh`` is element-wise and unary in every revision of its schema
 * (v1, v6, v13 — later revisions only widen the accepted dtype set),
 * so the output dtype and shape always match those of the input.
 */
void ComputeShapeTanh(ShapesContext &ctx, const NodeProto &node, const char *x);

/// ThresholdedRelu is element-wise unary: output dtype and shape match the input.
void ComputeShapeThresholdedRelu(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Relu is element-wise unary: output dtype and shape match the input.
void ComputeShapeRelu(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Elu is element-wise unary: output dtype and shape match the input.
void ComputeShapeElu(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Celu is element-wise unary: output dtype and shape match the input.
void ComputeShapeCelu(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Gelu is element-wise unary: output dtype and shape match the input.
void ComputeShapeGelu(ShapesContext &ctx, const NodeProto &node, const char *x);

/// Selu is element-wise unary: output dtype and shape match the input.
void ComputeShapeSelu(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Floor`` node and
 * stores it in ``ctx``.
 *
 * ``Floor`` is element-wise and unary in every revision of its schema
 * (v1, v6, v13 — later revisions only widen the accepted dtype set),
 * so the output dtype and shape always match those of the input.
 */
void ComputeShapeFloor(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Ceil`` node and
 * stores it in ``ctx``.
 *
 * ``Ceil`` is element-wise and unary in every revision of its schema
 * (v1, v6, v13 — later revisions only widen the accepted dtype set),
 * so the output dtype and shape always match those of the input.
 */
void ComputeShapeCeil(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Clip`` node and
 * stores it in ``ctx``.
 *
 * ``Clip`` is element-wise; the optional ``min`` and ``max`` inputs are
 * scalars (or in v1/v6 schema attributes) that do not influence the
 * output shape or dtype, which always match the input ``x``.
 */
void ComputeShapeClip(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Round`` node and
 * stores it in ``ctx``.
 *
 * ``Round`` is element-wise and unary in every revision of its schema
 * (v11, v22 — later revisions only widen the accepted dtype set), so
 * the output dtype and shape always match those of the input.
 */
void ComputeShapeRound(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Einsum`` node
 * (opset 12) and stores it in ``ctx``.
 *
 * ``Einsum`` evaluates the Einstein summation expressed by the ``equation``
 * attribute over the variadic input tensors. The equation may contain an
 * ellipsis (``...``) to broadcast leading dimensions, and may be given
 * either in explicit form (``->`` followed by the output term) or implicit
 * form. The output dtype is the dtype of the first input.
 *
 * @param ctx   In/out context. Must already contain entries for every value
 *              listed in ``node.input``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Einsum`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Einsum"``,
 *              ``node`` must declare at least one input and one output, and
 *              must carry an ``equation`` STRING attribute.
 *
 * @throws std::invalid_argument if ``node`` is malformed, if the equation
 *         cannot be parsed, or if input ranks/labels are inconsistent.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeEinsum(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Sum`` node and stores
 * it in ``ctx``.
 *
 * ``Sum`` is a variadic element-wise operator: every input must share the
 * same float dtype (type constraint ``T``); since opset 8 the inputs may
 * have different shapes that follow NumPy-style multidirectional broadcasting
 * rules (earlier opsets required identical shapes, which is a strict subset).
 * The output dtype matches the inputs' shared dtype and the output shape is
 * the broadcast of all input shapes. Reads the descriptors of every input
 * from ``ctx`` and stores the result under ``node.output(0)``.
 *
 * @param ctx   In/out context. Must already contain entries for every value
 *              listed in ``node.input``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Sum`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Sum"`` and ``node`` must
 *              declare at least one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Sum"``,
 *         if ``node`` has no input or no output, or if any pair of inputs
 *         have shapes that are not broadcast-compatible.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeSum(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Max`` node and stores
 * it in ``ctx``.
 *
 * ``Max`` (opsets 1-13) is a variadic element-wise maximum operator. Since
 * opset 8 it supports NumPy-style multidirectional broadcasting; in earlier
 * opsets all inputs are required to share the same shape. The output dtype
 * always matches that of the first input, and the output shape is the
 * multidirectional broadcast of every input shape.
 *
 * @param ctx   In/out context. Must already contain entries for every input
 *              of ``node``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Max`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Max"`` and ``node`` must
 *              declare at least one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Max"``,
 *         if ``node`` has no input or no output, or if any pair of inputs
 *         have shapes that are not broadcast-compatible.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeMax(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Min`` node and stores
 * it in ``ctx``.
 *
 * ``Min`` (opsets 1-13) is a variadic element-wise minimum operator. Since
 * opset 8 it supports NumPy-style multidirectional broadcasting; in earlier
 * opsets all inputs are required to share the same shape. The output dtype
 * always matches that of the first input, and the output shape is the
 * multidirectional broadcast of every input shape.
 *
 * @param ctx   In/out context. Must already contain entries for every input
 *              of ``node``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Min`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Min"`` and ``node`` must
 *              declare at least one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Min"``,
 *         if ``node`` has no input or no output, or if any pair of inputs
 *         have shapes that are not broadcast-compatible.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeMin(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Mean`` node and stores
 * it in ``ctx``.
 *
 * ``Mean`` (opsets 1, 6, 8 and 13) is a variadic element-wise mean operator.
 * Since opset 8 it supports NumPy-style multidirectional broadcasting; in
 * earlier opsets all inputs are required to share the same shape. The output
 * dtype always matches that of the first input (the type constraint ``T``
 * forces every input to share the same float dtype), and the output shape is
 * the multidirectional broadcast of every input shape.
 *
 * @param ctx   In/out context. Must already contain entries for every input
 *              of ``node``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Mean`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Mean"`` and ``node`` must
 *              declare at least one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Mean"``,
 *         if ``node`` has no input or no output, or if any pair of inputs
 *         have shapes that are not broadcast-compatible.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeMean(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``CumSum`` node and
 * stores it in ``ctx``.
 *
 * ``CumSum`` (opsets 11 and 14) is a unary running-sum operator along an
 * axis selected by a second 0-D ``axis`` input tensor. The output dtype and
 * shape always match those of the first input ``x``; the ``axis``,
 * ``exclusive`` and ``reverse`` parameters affect values only.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x`` (the
 *              data input). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``CumSum`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"CumSum"`` and ``node`` must
 *              declare at least one output.
 * @param x     Name of the data input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"CumSum"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeCumSum(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``CumProd`` node and
 * stores it in ``ctx``.
 *
 * ``CumProd`` (opset 26) is a unary running-product operator along an axis
 * selected by a second 0-D ``axis`` input tensor. The output dtype and
 * shape always match those of the first input ``x``.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x`` (the
 *              data input). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``CumProd`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"CumProd"`` and ``node`` must
 *              declare at least one output.
 * @param x     Name of the data input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"CumProd"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeCumProd(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` entries of a ``TopK`` node
 * and stores them in ``ctx``.
 *
 * ``TopK`` returns two outputs that share the same shape:
 *
 * - ``Values`` has the same dtype as the input ``x`` and shape
 *   ``x.shape`` with the ``axis`` dimension replaced by ``k``.
 * - ``Indices`` has dtype ``INT64`` and the same shape as ``Values``.
 *
 * In opset 1 the number ``k`` is read from the required integer attribute
 * ``k``. In opsets 10 and 11 ``k`` is supplied as a 1-D tensor input;
 * because :class:`OptimTensor` does not always carry concrete data, the
 * axis dimension is emitted as a symbolic dimension (``TopK_<output>_k``)
 * when ``k`` cannot be resolved from a constant.
 *
 * @param ctx   In/out context. Must already contain an entry for the data
 *              input ``x``. On return it also contains entries for
 *              ``node.output(0)`` and (when present) ``node.output(1)``.
 * @param node  The ``TopK`` ``NodeProto`` whose outputs should be described.
 *              ``node.op_type()`` must be ``"TopK"`` and ``node`` must
 *              declare at least one output.
 * @param x     Name of the data input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"TopK"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeTopK(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``DFT`` node and stores
 * it in ``ctx``.
 *
 * ``DFT`` (since opset 17; ``axis`` moved from attribute to input at opset
 * 20) returns a tensor of the same rank as ``node.input(0)``. Its trailing
 * dimension is ``1`` when ``onesided`` and ``inverse`` are both set (IRFFT)
 * and ``2`` otherwise. The signal axis dimension is replaced by
 * ``dft_length`` (or ``floor(dft_length/2)+1`` for RFFT) when the
 * ``dft_length`` input is a known scalar constant; it is left symbolic
 * otherwise. The non-signal/non-trailing dimensions are copied from the
 * input.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``DFT`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"DFT"`` and ``node`` must
 *              declare at least one input and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"DFT"`` or
 *         if ``node`` declares no inputs/outputs.
 * @throws std::out_of_range     if the data input is missing from ``ctx``.
 */
void ComputeShapeDFT(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``STFT`` node and stores
 * it in ``ctx``.
 *
 * ``STFT`` (opset 17) returns a rank-4 tensor with shape
 * ``[batch_size, n_frames, dft_unique_bins, 2]`` where
 * ``n_frames = (signal_length - frame_length) / frame_step + 1`` and
 * ``dft_unique_bins`` is ``floor(frame_length / 2) + 1`` when ``onesided``
 * is enabled (its default) or ``frame_length`` otherwise. The
 * ``signal_length``, ``frame_step`` and ``frame_length`` (or window length)
 * may be unknown at shape-inference time; in that case the corresponding
 * dimension is left symbolic.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``STFT`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"STFT"`` and ``node`` must
 *              declare at least two inputs (signal, frame_step) and one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"STFT"`` or
 *         if ``node`` declares fewer than two inputs/one output.
 * @throws std::out_of_range     if the data input is missing from ``ctx``.
 */
void ComputeShapeSTFT(ShapesContext &ctx, const NodeProto &node);

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
