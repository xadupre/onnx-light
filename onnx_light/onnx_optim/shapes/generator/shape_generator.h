// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_generator.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``generator`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

/// Maximum element count of a ``Constant`` value tensor for which
/// :cpp:func:`ComputeShapeConstant` populates the output
/// :cpp:func:`OptimTensor::ValueAsShape` annotation. Constants beyond
/// this threshold are not data-propagated (the output dtype and shape
/// are still inferred normally).
inline constexpr int64_t kConstantValueAsShapeMaxElements = 8;

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Constant`` node
 * and stores it in ``ctx``.
 *
 * ``Constant`` declares its output as the value of exactly one of the
 * attributes ``value``, ``sparse_value``, ``value_int``,
 * ``value_ints``, ``value_float``, ``value_floats``, ``value_string``
 * or ``value_strings`` (which one is allowed depends on the schema
 * revision, but for shape inference the union of every revision is
 * accepted). The output dtype and shape are taken from the present
 * attribute.
 *
 * When the resulting tensor carries at most
 * :cpp:var:`kConstantValueAsShapeMaxElements` integer elements and
 * has rank at most one — i.e. it is small enough to plausibly be used
 * later as a shape input of operators such as ``Reshape``,
 * ``Expand`` or ``ConstantOfShape`` — its integer values are also
 * recorded via :cpp:func:`OptimTensor::SetValueAsShape`. This mirrors
 * the upstream ONNX shape-inference data-propagation behaviour for
 * small integer constants.
 *
 * @param ctx   In/out context. On return contains an entry for
 *              ``node.output(0)`` describing the constant output.
 * @param node  The ``Constant`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Constant"``
 *              and ``node`` must declare at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Constant"``, if ``node`` has no output, or if the
 *         attributes do not specify exactly one of the allowed value
 *         forms.
 */
void ComputeShapeConstant(ShapesContext &ctx, const NodeProto &node);

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
