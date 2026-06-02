// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_nn.h
 * @brief Shape-inference functions for ONNX operators in the ``nn`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``AveragePool`` node
 * and stores it in ``ctx``.
 *
 * The output dtype matches the input dtype. The output rank is the input
 * rank (``[N, C, D1, ..., Dk]``). The first two output dimensions (``N``
 * and ``C``) are copied from the input. For each spatial axis ``i`` the
 * output dimension is computed from the ``kernel_shape``, ``strides``,
 * ``pads``, and ``ceil_mode`` attributes using the same rule as
 * :cpp:func:`onnx_backend_test::kernel::AveragePool` and ONNX Runtime:
 * when ``ceil_mode=1`` and the last sliding window would start entirely
 * in the right padded region, it is dropped. ``auto_pad`` other than the
 * default ``"NOTSET"`` (or ``"VALID"``) is not supported, and symbolic
 * spatial dimensions propagate symbolically.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``;
 *              on return it also contains an entry for ``node.output(0)``.
 * @param node  The ``AveragePool`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"AveragePool"``
 *              and ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"AveragePool"``, if ``node`` has no output, if the input
 *         rank is inconsistent with the (required) ``kernel_shape``
 *         attribute, or if ``auto_pad`` is set to a value other than
 *         ``"NOTSET"`` / ``"VALID"`` (only explicit ``pads`` are
 *         supported).
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeAveragePool(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a global pooling node
 * (``GlobalAveragePool``, ``GlobalMaxPool``, or ``GlobalLpPool``) and stores
 * it in ``ctx``.
 *
 * The output dtype matches the input dtype. The output rank equals the input
 * rank. The first two output dimensions (``N`` and ``C``) are copied from the
 * input; all remaining spatial dimensions are set to ``1``.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``;
 *              on return it also contains an entry for ``node.output(0)``.
 * @param node  The global pooling ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be one of
 *              ``"GlobalAveragePool"``, ``"GlobalMaxPool"`` or
 *              ``"GlobalLpPool"`` and ``node`` must declare at least one
 *              output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if the input rank is less than 2 or if
 *         ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeGlobalPool(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Flatten`` node and
 * stores it in ``ctx``.
 *
 * The output dtype matches the input dtype. The output shape is always
 * rank 2: ``(prod(input.shape[0:axis]), prod(input.shape[axis:rank]))``,
 * where ``axis`` is the integer attribute (default ``1``) and may be
 * negative (counted from the back). Symbolic dimensions propagate
 * symbolically — when any contributing dim is symbolic the corresponding
 * output dim becomes a fresh symbolic expression.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``;
 *              on return it also contains an entry for ``node.output(0)``.
 * @param node  The ``Flatten`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Flatten"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Flatten"``,
 *         if ``node`` has no output, or if ``axis`` is out of range
 *         ``[-r, r]`` for input rank ``r``.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeFlatten(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BatchNormalization``
 * node and stores it in ``ctx``.
 *
 * The op has between 1 and 5 outputs depending on opset / mode. The first
 * output ``Y`` always has the same dtype and shape as ``X``. The additional
 * outputs are 1-D tensors of size equal to the channel dimension ``C`` (the
 * dim at index 1 of ``X`` when ``X`` has rank >= 2; ``1`` otherwise, per the
 * opset-9+ rule). Output dtypes follow the upstream spec:
 *
 * - opset 1..9: every extra output uses the same dtype as ``X`` (type ``T``).
 * - opset 14: outputs 1/2 (``running_mean`` / ``running_var``) take the dtype
 *   of ``input_mean`` / ``input_var`` (type ``U``).
 * - opset 15+: outputs 1/2 (``running_mean`` / ``running_var``) take the
 *   dtype of ``input_mean`` / ``input_var`` (type ``T2``).
 *
 * The ``training_mode`` attribute (opset 14+) and the legacy ``is_test``
 * attribute (opset 1, 6) are honored to decide whether extra outputs are
 * required. Only the channel-dim of the secondary outputs is inferred; for
 * a symbolic channel dim a fresh symbolic expression
 * ``BatchNormalization.C(<expr>)`` is propagated.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``;
 *              on return it also contains an entry for each declared output
 *              of ``node``.
 * @param node  The ``BatchNormalization`` ``NodeProto``. ``node.op_type()``
 *              must be ``"BatchNormalization"`` and at least one output must
 *              be declared.
 * @param x     Name of the data input value to read from ``ctx``.
 * @param input_mean Name of the mean input value (used to obtain the dtype
 *                   of the secondary outputs from opset 14 onward). May be
 *                   ``nullptr`` if not needed (opset < 14 fallback).
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"BatchNormalization"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` or ``input_mean`` is not present
 *                               in ``ctx`` when required.
 */
void ComputeShapeBatchNormalization(ShapesContext &ctx, const NodeProto &node, const char *x,
                                    const char *input_mean);

/**
 * Computes the output :cpp:class:`OptimTensor`(s) of a ``Dropout`` node and
 * stores them in ``ctx``.
 *
 * ``output`` always has the same dtype and shape as ``data``. If the optional
 * second output ``mask`` is present and non-empty, it has dtype
 * ``TensorType::kBool`` and the same shape as ``data``. Optional inputs
 * ``ratio`` and ``training_mode`` must be scalars when present.
 */
void ComputeShapeDropout(ShapesContext &ctx, const NodeProto &node, const char *data,
                         const char *ratio = nullptr, const char *training_mode = nullptr);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``RoiAlign`` node and
 * stores it in ``ctx``.
 *
 * The output dtype matches the input feature-map dtype. The output shape
 * is ``(num_rois, C, output_height, output_width)`` where ``num_rois`` is
 * taken from dim 0 of ``rois`` (or, when that dim is symbolic and
 * ``batch_indices`` dim 0 is static, from ``batch_indices``); ``C`` is
 * taken from dim 1 of ``x``; and the spatial sizes come from the
 * ``output_height`` / ``output_width`` attributes (default 1).
 *
 * @param ctx              In/out context. Must already contain entries for
 *                         ``x``, ``rois``, and ``batch_indices``; on return
 *                         it also contains an entry for ``node.output(0)``.
 * @param node             The ``RoiAlign`` ``NodeProto`` whose output should
 *                         be described. ``node.op_type()`` must be
 *                         ``"RoiAlign"`` and ``node`` must declare at least
 *                         one output.
 * @param x                Name of the feature-map input value (rank 4) to
 *                         read from ``ctx``. Must be present in ``ctx``.
 * @param rois             Name of the RoIs input value (rank 2) to read
 *                         from ``ctx``. Must be present in ``ctx``.
 * @param batch_indices    Name of the batch-indices input value (rank 1) to
 *                         read from ``ctx``. Must be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"RoiAlign"``,
 *         if ``node`` has no output, if any input has the wrong rank, or
 *         if ``output_height`` / ``output_width`` is non-positive.
 * @throws std::out_of_range     if any input name is not present in ``ctx``.
 */
void ComputeShapeRoiAlign(ShapesContext &ctx, const NodeProto &node, const char *x,
                          const char *rois, const char *batch_indices);

/**
 * Computes the output :cpp:class:`OptimTensor`(s) of an ``RNN``, ``GRU`` or
 * ``LSTM`` node and stores them in ``ctx``. The three operators share the
 * same output-shape semantics — only the number of outputs differs
 * (``RNN`` / ``GRU`` expose ``Y`` and ``Y_h``; ``LSTM`` also exposes
 * ``Y_c``).
 *
 * The output dtypes are inherited from ``X``. Shapes follow the upstream
 * ``RNNShapeInference`` rules:
 *
 * - Read ``direction`` (``"forward"`` (default), ``"reverse"`` or
 *   ``"bidirectional"``) to derive ``num_directions`` (1 or 2). Unknown
 *   values leave ``num_directions`` symbolic.
 * - Read ``hidden_size``; when missing or non-positive, fall back to
 *   ``R.shape[2]`` (the recurrence weight's last dim) when known.
 * - Read ``layout`` (default 0). For ``layout=0`` derive ``seq_length`` and
 *   ``batch_size`` from ``X.shape[0]`` and ``X.shape[1]``; for ``layout=1``
 *   the order is reversed.
 *
 * ``Y`` has rank 4 (``[seq_length, num_directions, batch_size,
 * hidden_size]`` for ``layout=0``; ``[batch_size, seq_length,
 * num_directions, hidden_size]`` for ``layout=1``). ``Y_h`` (and ``Y_c``,
 * for ``LSTM``) has rank 3 (``[num_directions, batch_size, hidden_size]``
 * for ``layout=0``; ``[batch_size, num_directions, hidden_size]`` for
 * ``layout=1``). Missing output dims propagate as a fresh symbolic
 * expression labeled with the operator and field name.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``;
 *              ``r`` is consulted only as a fallback for ``hidden_size``
 *              and may be ``nullptr`` or missing from ``ctx``. On return
 *              ``ctx`` also contains an entry for each declared (non-empty)
 *              output of ``node``.
 * @param node  The ``RNN`` / ``GRU`` / ``LSTM`` ``NodeProto``.
 *              ``node.op_type()`` must be one of ``"RNN"``, ``"GRU"`` or
 *              ``"LSTM"`` and at least one output must be declared.
 * @param x     Name of the data input value to read from ``ctx``. Must be
 *              present in ``ctx`` and have rank 3.
 * @param r     Name of the recurrence-weight input value (used as a
 *              fallback source of ``hidden_size``). May be ``nullptr`` or
 *              absent from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not one of
 *         ``"RNN"`` / ``"GRU"`` / ``"LSTM"``, if ``node`` has no output,
 *         or if ``X`` does not have rank 3.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeRNN(ShapesContext &ctx, const NodeProto &node, const char *x, const char *r);

/**
 * Computes the output :cpp:class:`OptimTensor`(s) of an ``Attention`` node
 * (since opset 23 in the ``ai.onnx`` domain) and stores them in ``ctx``.
 *
 * ``Attention`` accepts 3 to 7 inputs (``Q``, ``K``, ``V`` and the optional
 * ``attn_mask``, ``past_key``, ``past_value``, ``nonpad_kv_seqlen``) and
 * exposes between 1 and 4 outputs (``Y`` plus the optional ``present_key``,
 * ``present_value`` and ``qk_matmul_output``). Only the rank-4 input form
 * is described here; the rank-3 form (where ``q_num_heads`` /
 * ``kv_num_heads`` are read from attributes) defers shape inference to
 * the dispatcher.
 *
 * For rank-4 inputs the function infers:
 *
 *   - ``Y``: ``(batch_size, q_num_heads, q_sequence_length, v_head_size)``
 *     with dtype matching ``Q``.
 *   - ``present_key``: ``(batch_size, kv_num_heads, total_sequence_length,
 *     head_size)`` where ``total_sequence_length = past_sequence_length +
 *     kv_sequence_length`` (or just ``kv_sequence_length`` when no past
 *     state is provided).
 *   - ``present_value``: ``(batch_size, kv_num_heads,
 *     total_sequence_length, v_head_size)`` with dtype matching ``V``.
 *   - ``qk_matmul_output``: ``(batch_size, q_num_heads, q_sequence_length,
 *     total_sequence_length)`` with dtype matching ``Q``.
 *
 * ``q_num_heads`` must be a multiple of ``kv_num_heads`` when both are
 * static (Grouped Query Attention). Symbolic dimensions propagate
 * symbolically.
 *
 * @param ctx          In/out context. Must already contain entries for
 *                     ``q``, ``k`` and ``v``; on return it also contains
 *                     an entry for each declared output of ``node``.
 * @param node         The ``Attention`` ``NodeProto`` whose outputs should
 *                     be described. ``node.op_type()`` must be
 *                     ``"Attention"`` and ``node`` must declare at least
 *                     one output.
 * @param q            Name of the query input value (rank 4).
 * @param k            Name of the key input value (rank 4).
 * @param v            Name of the value input value (rank 4).
 * @param past_key     Optional name of the past_key input (rank 4). When
 *                     not ``nullptr`` and present in ``ctx``, contributes
 *                     ``past_sequence_length`` to the present_* outputs.
 * @param past_value   Optional name of the past_value input (rank 4).
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Attention"``, if ``node`` has no output, if any of ``Q``/
 *         ``K``/``V`` has rank other than 4, or if static shapes are
 *         inconsistent (mismatched batch size, mismatched head dim,
 *         mismatched kv_sequence_length, or ``q_num_heads`` not a
 *         multiple of ``kv_num_heads``).
 * @throws std::out_of_range     if ``q``/``k``/``v`` is not present in
 *                               ``ctx``.
 */
void ComputeShapeAttention(ShapesContext &ctx, const NodeProto &node, const char *q, const char *k,
                           const char *v, const char *past_key = nullptr,
                           const char *past_value = nullptr);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``DeformConv`` node and
 * stores it in ``ctx``.
 *
 * The output dtype matches the input ``X`` dtype. The output shape is
 * ``(N, oC, o1, ..., on)`` where ``N`` is ``X.shape[0]``, ``oC`` is
 * ``W.shape[0]``, and each spatial dim ``oi`` is computed from the input
 * spatial dim, the (effective) kernel shape, ``strides``, ``pads`` and
 * ``dilations`` attributes — matching the upstream
 * ``convPoolShapeInference`` rule shared with ``Conv``. When the
 * ``kernel_shape`` attribute is missing, the kernel shape is taken from
 * ``W.shape[2..]``. Symbolic spatial dimensions are propagated symbolically.
 *
 * @param ctx   In/out context. Must already contain entries for ``x`` and
 *              ``w``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``DeformConv`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"DeformConv"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input data value (rank >= 3) to read from
 *              ``ctx``. Must be present in ``ctx``.
 * @param w     Name of the weight value (rank >= 3) to read from ``ctx``.
 *              Must be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"DeformConv"``, if ``node`` has no output, if ``X`` and ``W``
 *         have inconsistent ranks, or if attributes have wrong sizes.
 * @throws std::out_of_range     if ``x`` or ``w`` is not present in ``ctx``.
 */
void ComputeShapeDeformConv(ShapesContext &ctx, const NodeProto &node, const char *x,
                            const char *w);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Col2Im`` node and stores
 * it in ``ctx``.
 *
 * The output dtype matches the input ``input`` dtype. The output shape is
 * ``(N, C, dim_i1, ..., dim_iN)`` where ``N`` is ``input.shape[0]``, ``C`` is
 * ``input.shape[1] / product(block_shape)`` (when the ``block_shape``
 * initializer is known), and the spatial dimensions are taken from the
 * ``image_shape`` initializer when known. When either initializer is missing
 * or the corresponding input shape is symbolic, the affected dimensions are
 * propagated symbolically.
 *
 * @param ctx   In/out context. Must already contain entries for ``input``,
 *              ``image_shape`` and ``block_shape``; on return it also contains
 *              an entry for ``node.output(0)``.
 * @param node  The ``Col2Im`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Col2Im"`` and ``node`` must
 *              declare at least one output.
 * @param input         Name of the data input value (rank 3) in ``ctx``.
 * @param image_shape   Name of the ``image_shape`` 1-D ``tensor(int64)`` value
 *                      in ``ctx``.
 * @param block_shape   Name of the ``block_shape`` 1-D ``tensor(int64)`` value
 *                      in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Col2Im"``,
 *         if ``node`` has no output, or if the inputs have inconsistent
 *         shapes.
 * @throws std::out_of_range     if ``input``, ``image_shape`` or
 *                               ``block_shape`` is not present in ``ctx``.
 */
void ComputeShapeCol2Im(ShapesContext &ctx, const NodeProto &node, const char *input,
                        const char *image_shape, const char *block_shape);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Conv`` node and stores
 * it in ``ctx``.
 *
 * The output dtype matches ``X``. The output shape is ``(N, M, o1, ..., on)``
 * where ``N`` is ``X.shape[0]``, ``M`` is ``W.shape[0]``, and each spatial
 * dim is derived from ``kernel_shape``, ``strides``, ``pads``, ``dilations``
 * and ``auto_pad`` following the upstream ``convPoolShapeInference`` rule.
 */
void ComputeShapeConv(ShapesContext &ctx, const NodeProto &node, const char *x, const char *w);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ConvInteger`` node and
 * stores it in ``ctx``. The shape rule matches :cpp:func:`ComputeShapeConv`;
 * the output dtype is always ``TensorType::kInt32``.
 */
void ComputeShapeConvInteger(ShapesContext &ctx, const NodeProto &node, const char *x,
                             const char *w);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ConvTranspose`` node
 * and stores it in ``ctx``.
 *
 * The output dtype matches ``X``. The output shape is ``(N, M, o1, ..., on)``
 * where ``M`` is ``W.shape[1] * group``. Each spatial dim is derived from
 * ``kernel_shape``, ``strides``, ``pads``, ``dilations``, ``output_padding``,
 * ``output_shape`` and ``auto_pad`` following the upstream
 * ``convTransposeShapeInference`` rule.
 */
void ComputeShapeConvTranspose(ShapesContext &ctx, const NodeProto &node, const char *x,
                               const char *w);

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
