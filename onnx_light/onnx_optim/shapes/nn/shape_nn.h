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

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
