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
 *        :cpp:func:`onnx_optim::shapes::ComputeShapeNode` to forward
 *        each ``NodeProto`` to the matching ``ComputeShape*``
 *        implementation.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Signature of every per-operator ``ComputeShape*`` trampoline
 * registered in :cpp:func:`DispatchTable`: it reads the node's inputs
 * from ``ctx`` and inserts the resulting output descriptors back
 * into ``ctx``.
 */
using ComputeShapeFn = std::function<void(ShapesContext &, const NodeProto &)>;

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
