// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_text.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``text`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace text {

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``StringConcat`` node
 * and stores it in ``ctx``.
 *
 * ``StringConcat`` concatenates two string tensors element-wise with
 * numpy-style multidirectional broadcasting (since opset 20 in the
 * ``ai.onnx`` domain). The output dtype is always
 * :cpp:enumerator:`TensorType::kString` and the output shape is the
 * broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``StringConcat`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"StringConcat"`` and ``node`` must declare at least
 *              one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"StringConcat"``, if ``node`` has no output, or if the two
 *         input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeStringConcat(ShapesContext &ctx, const NodeProto &node, const char *a,
                              const char *b);

} // namespace text
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
