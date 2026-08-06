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
 * @file shape_preview.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``ai.onnx.preview`` domain.
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

namespace preview {

/// The ai.onnx.preview operator domain string.
inline constexpr const char *kOnnxPreviewDomain = "ai.onnx.preview";

/**
 * Computes the output :cpp:class:`SymTensor` of a ``FlexAttention``
 * node and stores it in ``ctx``.
 *
 * ``FlexAttention`` expects three rank-4 inputs:
 *
 *   - ``Q`` with shape ``(batch_size, q_num_heads, q_seq_len, head_size)``;
 *   - ``K`` with shape ``(batch_size, kv_num_heads, kv_seq_len, head_size)``;
 *   - ``V`` with shape ``(batch_size, kv_num_heads, kv_seq_len, v_head_size)``.
 *
 * The output ``Y`` has the same element type as ``Q`` and shape
 * ``(batch_size, q_num_heads, q_seq_len, v_head_size)``. The function
 * validates that ``Q``/``K``/``V`` share the same dtype, that ``K`` and
 * ``V`` share the same head count and sequence length, that ``Q`` and
 * ``K`` share the same embedding dimension, and that ``q_num_heads`` is
 * a multiple of ``kv_num_heads`` (Grouped Query Attention).
 *
 * Symbolic dimensions propagate symbolically: any constraint that
 * cannot be verified because one side is symbolic is skipped, as in the
 * upstream ``FlexAttentionShapeInference`` in ``onnx_lib``.
 *
 * @param ctx   In/out context. Must already contain entries for ``q``,
 *              ``k``, and ``v``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``FlexAttention`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"FlexAttention"`` and ``node`` must declare at least
 *              one output.
 * @param q     Name of the query input value to read from ``ctx``.
 * @param k     Name of the key input value to read from ``ctx``.
 * @param v     Name of the value input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"FlexAttention"``, if ``node`` has no output, if any input
 *         is not rank 4, if Q/K/V have inconsistent element types, or
 *         if static dimensions violate the constraints above.
 * @throws std::out_of_range     if ``q``, ``k``, or ``v`` is not present
 *         in ``ctx``.
 */
void ComputeShapeFlexAttention(ShapesContext &ctx, const NodeProto &node, const char *q,
                               const char *k, const char *v);

} // namespace preview
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes
