// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_traditionalml.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``ai.onnx.ml`` (traditional machine-learning) family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

/// Canonical domain string for the ``ai.onnx.ml`` operator set.
inline constexpr const char *kOnnxMlDomain = "ai.onnx.ml";

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``LabelEncoder``
 * node and stores it in ``ctx``.
 *
 * ``LabelEncoder`` (``ai.onnx.ml``) is a one-to-one mapping from input
 * keys to output values, so the output shape always matches the input
 * shape. The output dtype is determined by which of the ``values_*``
 * attributes is set:
 *
 *   - ``values_tensor`` — dtype is the tensor's ``data_type``;
 *   - ``values_strings`` — dtype is ``string``;
 *   - ``values_int64s``  — dtype is ``int64``;
 *   - ``values_floats``  — dtype is ``float``.
 *
 * Exactly one of these attributes must be set; an error is raised
 * otherwise.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``LabelEncoder`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"LabelEncoder"`` and ``node`` must declare at least
 *              one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"LabelEncoder"``, if ``node`` has no output, or if the
 *         attributes do not specify exactly one of the allowed
 *         ``values_*`` forms.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeLabelEncoder(ShapesContext &ctx, const NodeProto &node, const char *x);

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
