// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file dispatch_table.h
 * @brief Per-(domain, op_type) dispatch table used by
 *        :cpp:func:`onnx_optim::shapes::ShapesContext::ComputeShapeNode` to forward
 *        each ``NodeProto`` to the matching ``ComputeShape*``
 *        implementation.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace shapes {

/**
 * Returns the ``(normalised_domain, op_type) -> ComputeShape*``
 * dispatch table. Constructed on first use and shared across calls.
 *
 * The dispatch key is the pair ``(domain, op_type)`` encoded as
 * ``"<domain>:<op_type>"``; the empty default ONNX domain must be
 * normalised to ``kOnnxDomain`` before lookup. Adding a new operator
 * only requires inserting one new entry in the table.
 */
const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable();

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
