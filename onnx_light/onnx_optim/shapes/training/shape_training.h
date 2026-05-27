// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_training.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``ai.onnx.preview.training`` (training) family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace training {

/// Canonical domain string for the ``ai.onnx.preview.training`` operator set.
inline constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Adam`` node and
 * stores it in ``ctx``.
 *
 * ``Adam`` (``ai.onnx.preview.training``) updates ``N`` optimised
 * tensors and their accumulated first / second moments. The input list
 * has the shape ``[R, T, X_1, ..., X_N, G_1, ..., G_N, V_1, ..., V_N,
 * H_1, ..., H_N]`` (so ``input_size == 2 + 4 * N``) and the output list
 * has the shape ``[X_1_new, ..., X_N_new, V_1_new, ..., V_N_new,
 * H_1_new, ..., H_N_new]`` (so ``output_size == 3 * N``).
 *
 * Each ``*_new`` output mirrors the dtype and shape of the
 * corresponding ``X_i`` / ``V_i`` / ``H_i`` input.
 *
 * @param ctx   In/out context. Must already contain entries for every
 *              ``X_i``, ``V_i`` and ``H_i`` input read from ``node``;
 *              on return it also contains an entry for every output of
 *              ``node``.
 * @param node  The ``Adam`` ``NodeProto`` whose outputs should be
 *              described. ``node.op_type()`` must be ``"Adam"``,
 *              ``node.input_size()`` must be ``2 + 4 * N`` for some
 *              ``N >= 1`` and ``node.output_size()`` must be ``3 * N``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Adam"``, if ``node`` has no output, if the input count
 *         minus two is not a positive multiple of four, or if the
 *         output count does not match three times the number of
 *         optimised tensors.
 * @throws std::out_of_range     if any input read from ``node`` is not
 *                               present in ``ctx``.
 */
void ComputeShapeAdam(ShapesContext &ctx, const NodeProto &node);

} // namespace training
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
