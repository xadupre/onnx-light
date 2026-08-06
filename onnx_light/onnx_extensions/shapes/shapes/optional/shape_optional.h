// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_map.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_optional.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``optional`` family.
 */

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes {

// The generic shape-inference engine (ShapesContext, dispatch table,
// domain constants, ...) lives in ``onnx_core`` so it never depends on
// any particular set of operator implementations; the symbolic value
// descriptors (SymDim, SymShape, SymTensor, ...) live in
// ``onnx_core::symbolic`` for the same reason. Bring them into
// ``onnx_shapes::shapes`` so shape functions can keep referring to them
// unqualified, exactly as before those types moved to ``onnx_core``.
using ::onnx_light::core::shapes::BroadcastDimOp;
using ::onnx_light::core::shapes::BroadcastShapes;
using ::onnx_light::core::shapes::CheckNodeOpAndOutput;
using ::onnx_light::core::shapes::ComputeShapeBinaryBroadcast;
using ::onnx_light::core::shapes::InferShapesModel;
using ::onnx_light::core::shapes::kOnnxDomain;
using ::onnx_light::core::shapes::kUnknownOpsetVersion;
using ::onnx_light::core::shapes::PropagateValueAsShapeArithmetic;
using ::onnx_light::core::shapes::ShapeEvent;
using ::onnx_light::core::shapes::ShapeEventAction;
using ::onnx_light::core::shapes::ShapeEventActionName;
using ::onnx_light::core::shapes::ShapesContext;

using ::onnx_light::core::symbolic::DataTypeToTensorType;
using ::onnx_light::core::symbolic::Device;
using ::onnx_light::core::symbolic::GPUIndex;
using ::onnx_light::core::symbolic::IsGPU;
using ::onnx_light::core::symbolic::IsIntegerTensorType;
using ::onnx_light::core::symbolic::kMaxGPUIndex;
using ::onnx_light::core::symbolic::kMaxOptimRank;
using ::onnx_light::core::symbolic::kOptimValueAsShapeMaxElements;
using ::onnx_light::core::symbolic::kValueInfoDeviceMetadataKey;
using ::onnx_light::core::symbolic::kValueInfoMaxMetadataKey;
using ::onnx_light::core::symbolic::kValueInfoMinMetadataKey;
using ::onnx_light::core::symbolic::ShapeFromTensorProtoDims;
using ::onnx_light::core::symbolic::SymCmpResult;
using ::onnx_light::core::symbolic::SymDim;
using ::onnx_light::core::symbolic::SymMap;
using ::onnx_light::core::symbolic::SymSequence;
using ::onnx_light::core::symbolic::SymShape;
using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

namespace optional {

/**
 * Computes the output :cpp:class:`SymTensor` of an ``Optional`` node
 * and stores it in ``ctx``.
 *
 * ``Optional`` (since opset 15) wraps a value into an optional-type
 * output. The wrapped element type is determined either by the input
 * value (when an input is provided) or by the ``type`` ``TypeProto``
 * attribute (when no input is provided). Since :cpp:class:`SymTensor`
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
 * no-op. Because :cpp:class:`SymTensor` does not model optional
 * values, this implementation forwards the input descriptor verbatim:
 *
 *   - if the input name is bound to an :cpp:class:`SymSequence` in
 *     ``ctx``, the output is registered as the same sequence;
 *   - otherwise the input must be bound to an :cpp:class:`SymTensor`
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
 * always a scalar :cpp:class:`SymTensor` of dtype
 * :cpp:enumerator:`TensorType::kBool`.
 *
 * @param ctx   In/out context. On return it contains an
 *              :cpp:class:`SymTensor` entry for ``node.output(0)``.
 * @param node  The ``OptionalHasElement`` ``NodeProto``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"OptionalHasElement"``, if ``node`` declares more than one
 *         input or does not declare exactly one output.
 */
void ComputeShapeOptionalHasElement(ShapesContext &ctx, const NodeProto &node);

} // namespace optional
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes
