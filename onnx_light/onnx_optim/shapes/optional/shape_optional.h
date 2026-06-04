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

/**
 * Computes the output descriptor of an ``OptionalGetElement`` node and
 * stores it in ``ctx``.
 *
 * ``OptionalGetElement`` (since opset 15 in the ``ai.onnx`` domain)
 * extracts the element from an optional-type input. Since opset 18 the
 * operator also accepts non-optional tensor or sequence inputs as a
 * no-op. Because :cpp:class:`OptimTensor` does not model optional
 * values, this implementation forwards the input descriptor verbatim:
 *
 *   - if the input name is bound to an :cpp:class:`OptimSequence` in
 *     ``ctx``, the output is registered as the same sequence;
 *   - otherwise the input must be bound to an :cpp:class:`OptimTensor`
 *     and the output is registered as a tensor with the same dtype and
 *     shape.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``OptionalGetElement`` ``NodeProto``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"OptionalGetElement"``, if ``node`` does not declare
 *         exactly one input and one output, or if the input name is
 *         empty.
 * @throws std::out_of_range     if the input name is missing from
 *         ``ctx``.
 */
void ComputeShapeOptionalGetElement(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output descriptor of an ``OptionalHasElement`` node and
 * stores it in ``ctx``.
 *
 * ``OptionalHasElement`` (since opset 15 in the ``ai.onnx`` domain)
 * produces a scalar boolean tensor indicating whether the input
 * optional contains an element. Since opset 18 the input may also be a
 * non-optional tensor or sequence, and the input may be omitted
 * entirely (in which case the output is ``false``). The output is
 * always a scalar :cpp:class:`OptimTensor` of dtype
 * :cpp:enumerator:`TensorType::kBool`.
 *
 * @param ctx   In/out context. On return it contains an
 *              :cpp:class:`OptimTensor` entry for ``node.output(0)``.
 * @param node  The ``OptionalHasElement`` ``NodeProto``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"OptionalHasElement"``, if ``node`` declares more than one
 *         input or does not declare exactly one output.
 */
void ComputeShapeOptionalHasElement(ShapesContext &ctx, const NodeProto &node);

} // namespace optional
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
