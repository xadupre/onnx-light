// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_broadcast.h
 * @brief Shared helpers for shape-inference functions of binary ONNX
 *        operators that support numpy-style (multidirectional)
 *        broadcasting.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Computes the broadcast result shape of two :cpp:class:`OptimShape`
 * operands following the ONNX (numpy-style) multidirectional
 * broadcasting rules.
 *
 * The shapes are right-aligned and the dimensions are paired starting
 * from the trailing axis (missing leading dimensions are treated as
 * ``1``). For each paired dimension ``(d_a, d_b)`` the resulting
 * dimension is computed as follows:
 *
 *   - if both are concrete integers: standard broadcasting rules are
 *     enforced — equal dimensions or a dimension of ``1`` paired with
 *     anything are accepted; mismatching non-unit integers throw
 *     ``std::invalid_argument``;
 *   - if either is the integer ``1``: the result is the other
 *     dimension;
 *   - if both are equal (same integer or same symbolic expression):
 *     the result is that dimension;
 *   - if one is a concrete integer (different from ``1``) and the
 *     other is symbolic: the concrete integer wins (it is the only
 *     value compatible with broadcasting against itself);
 *   - if both are different symbolic expressions: a fresh symbolic
 *     dimension is produced, encoding the broadcast as
 *     ``"broadcast(<a>, <b>)"`` so that the symbolic information is
 *     preserved.
 *
 * @throws std::invalid_argument when two concrete integer dimensions
 *         are incompatible under broadcasting.
 */
OptimShape BroadcastShapes(const OptimShape &a, const OptimShape &b);

/**
 * Generic shape-inference helper for binary ONNX operators that
 * support numpy-style broadcasting. Reads the descriptors of
 * ``input_a`` and ``input_b`` from ``ctx``, computes the broadcast
 * output shape and stores a new entry under ``node.output(0)`` with
 * the given ``output_dtype``.
 *
 * The helper enforces the following preconditions:
 *
 *   - ``node.op_type()`` must equal ``expected_op_type``;
 *   - ``node`` must declare at least one output;
 *   - both ``input_a`` and ``input_b`` must be present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` differs from
 *         ``expected_op_type`` or if ``node`` has no output, or if the
 *         two input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either input name is missing from
 *         ``ctx``.
 */
void ComputeShapeBinaryBroadcast(ShapesContext &ctx, const NodeProto &node, const char *input_a,
                                 const char *input_b, const char *expected_op_type,
                                 TensorType output_dtype);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
