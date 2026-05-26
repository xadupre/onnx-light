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
