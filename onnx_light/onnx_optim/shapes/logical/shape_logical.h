// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

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
void ComputeShapeAnd(ShapesContext &ctx, const NodeProto &node, const std::string &a,
                     const std::string &b);

} // namespace logical
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
