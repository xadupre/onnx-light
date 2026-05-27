// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_preview.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``ai.onnx.preview`` domain.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace preview {

/// The ai.onnx.preview operator domain string.
inline constexpr const char *kOnnxPreviewDomain = "ai.onnx.preview";

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``FlexAttention``
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
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
