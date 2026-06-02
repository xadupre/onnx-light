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

/// Softmax preserves dtype/shape and validates the axis attribute.
void ComputeShapeSoftmax(ShapesContext &ctx, const NodeProto &node, const char *x);

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

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
