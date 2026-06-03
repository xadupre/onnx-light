// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_image.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``image`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace image {

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``ImageDecoder``
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
 *              :cpp:class:`OptimTensor` entry for ``a``; on return it
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
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
