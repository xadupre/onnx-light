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

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
