// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_rt.h
 * @brief Shape-inference functions for lightweight runtime operators in the
 *        ``ai.rt`` domain.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace rt {

/// The ai.rt operator domain string.
inline constexpr const char *kAiRtDomain = "ai.rt";

/**
 * This function computes the output ``OptimTensor`` of a
 * ``DelayedInitializer`` node and stores it in ``ctx``.
 *
 * ``DelayedInitializer`` is a lightweight runtime-only operator with no inputs.
 * Its output shape is given by the required ``shape`` attribute and its element
 * type is given by the required ``dtype`` attribute.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"DelayedInitializer"``, if ``node`` declares any inputs or no
 *         outputs, if the ``shape`` attribute is missing or contains a negative
 *         dimension, or if the ``dtype`` attribute is missing or unsupported.
 */
void ComputeShapeDelayedInitializer(ShapesContext &ctx, const NodeProto &node);

} // namespace rt
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
