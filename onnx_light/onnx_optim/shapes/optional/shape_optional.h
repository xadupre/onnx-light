// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_optional.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``optional`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace optional {

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Optional`` node
 * and stores it in ``ctx``.
 *
 * ``Optional`` (since opset 15) wraps a value into an optional-type
 * output. The wrapped element type is determined either by the input
 * value (when an input is provided) or by the ``type`` ``TypeProto``
 * attribute (when no input is provided). Since :cpp:class:`OptimTensor`
 * does not model optional or sequence types, this implementation only
 * supports the **tensor-element** path: the output descriptor mirrors
 * the dtype and shape of the wrapped tensor.
 *
 * Supported cases:
 *
 *   - ``node`` has one input: the output dtype and shape are copied
 *     from the input descriptor stored in ``ctx``.
 *   - ``node`` has no input and the ``type`` attribute wraps a tensor
 *     type (either directly or as the ``elem_type`` of an
 *     ``optional_type``): the output dtype and shape are taken from
 *     the attribute.
 *
 * @param ctx   In/out context. When the node has an input, ``ctx``
 *              must already contain an entry for it; on return ``ctx``
 *              contains an entry for ``node.output(0)``.
 * @param node  The ``Optional`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Optional"`` and
 *              ``node`` must declare exactly one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Optional"``, if ``node`` has no output or more than one
 *         input, if the ``type`` attribute wraps a sequence or
 *         non-tensor element type, or if neither an input nor a valid
 *         ``type`` attribute is provided.
 * @throws std::out_of_range     if the input name is missing from
 *         ``ctx``.
 */
void ComputeShapeOptional(ShapesContext &ctx, const NodeProto &node);

} // namespace optional
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
