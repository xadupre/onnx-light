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
 * @file shape_image.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``image`` family.
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

namespace image {

/**
 * Computes the output :cpp:class:`SymTensor` of an ``ImageDecoder``
 * node and stores it in ``ctx``.
 *
 * ``ImageDecoder`` (since opset 20 in the ``ai.onnx`` domain) takes a
 * 1-D ``tensor(uint8)`` carrying an encoded image bytestream and
 * returns the decoded image as a 3-D ``tensor(uint8)`` laid out as
 * ``(Height, Width, Channels)``. The spatial extent of the decoded
 * image only becomes known at runtime, so the output's ``H`` and ``W``
 * dimensions are produced as symbolic dims; the channel count is
 * derived from the ``pixel_format`` attribute (``"Grayscale"`` ⇒ 1,
 * ``"RGB"``/``"BGR"`` ⇒ 3, defaulting to ``"RGB"`` when omitted).
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`SymTensor` entry for ``a``; on return it
 *              also contains an entry for ``node.output(0)``.
 * @param node  The ``ImageDecoder`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"ImageDecoder"`` and ``node`` must declare at least
 *              one output.
 * @param a     Name of the input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"ImageDecoder"``, if ``node`` has no output, if the input
 *         tensor is not 1-dimensional, or if ``pixel_format`` is not
 *         one of ``"RGB"``, ``"BGR"`` or ``"Grayscale"``.
 * @throws std::out_of_range     if ``a`` is missing from ``ctx``.
 */
void ComputeShapeImageDecoder(ShapesContext &ctx, const NodeProto &node, const char *a);

} // namespace image
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes
