// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_inference.h
 * @brief Top-level shape-inference helpers that dispatch to the
 *        per-operator ``ComputeShape*`` functions of ``onnx_shapes``.
 *
 * The per-operator functions (for example
 * :cpp:func:`onnx_shapes::shapes::math::ComputeShapeAbs`) each take
 * their inputs by name. The methods of :cpp:class:`ShapesContext`
 * (declared in ``shapes_context.h`` and defined in
 * ``shape_inference.cc``) walk a single ``NodeProto`` or a
 * topologically-sorted sequence of ``NodeProto`` (typically
 * ``GraphProto::node()``), look up the op type and forward the call
 * to the matching ``ComputeShape*`` implementation.
 *
 * The shape-inference entry points
 * (:cpp:func:`ShapesContext::ComputeShapeNode`,
 * :cpp:func:`ShapesContext::ComputeShapes`,
 * :cpp:func:`ShapesContext::ComputeShapeGraph`,
 * :cpp:func:`ShapesContext::ComputeShapeModel`,
 * :cpp:func:`ShapesContext::ApplyInferredShapesToGraph`,
 * :cpp:func:`ShapesContext::ApplyInferredShapesToModel`,
 * :cpp:func:`ShapesContext::CheckInputsAvailable` and
 * :cpp:func:`ShapesContext::CheckOutputsNotAvailable`) are defined as
 * methods of :cpp:class:`ShapesContext`. The only free function left
 * in this header is :cpp:func:`InferShapesModel`, a convenience helper
 * that does not need a caller-supplied context.
 */

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

/**
 * Convenience helper: constructs a fresh :cpp:class:`ShapesContext`,
 * runs :cpp:func:`ShapesContext::ComputeShapeModel` on ``model`` and
 * then :cpp:func:`ShapesContext::ApplyInferredShapesToModel`, mutating
 * ``model`` in place so that its ``graph.output()`` and
 * ``graph.value_info()`` carry the inferred shapes.
 *
 * @param model  In/out model on which shape inference is run and
 *               whose proto is updated with the inferred results.
 * @param prefill_with_value_info_output  When ``true``, enables the same
 *               prefill/anchor behavior as
 *               :cpp:func:`ShapesContext::ComputeShapeModel`.
 *
 * @throws std::invalid_argument when ``model`` has no graph or when
 *         shape inference of the graph rejects a node.
 */
void InferShapesModel(ModelProto &model, bool prefill_with_value_info_output = false);

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
