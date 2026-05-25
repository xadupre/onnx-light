// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file abs.h
 * @brief Shape-inference function for the ONNX ``Abs`` operator.
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
void ComputeShapeAbs(ShapesContext &ctx, const NodeProto &node, const std::string &x);

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
