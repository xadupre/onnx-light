// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"

/**
 * @file shape_check.h
 * @brief Shared precondition helpers reused by every per-operator
 *        ``ComputeShape*`` shape-inference function.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

/**
 * Validates the common preamble of every ``ComputeShape*`` function:
 *
 *   - ``node.op_type()`` must equal ``expected_op_type``;
 *   - ``node`` must declare at least one output.
 *
 * The ``caller`` argument is embedded verbatim in the exception
 * messages so that callers can surface their own name (typically the
 * ``ComputeShape*`` function performing the check).
 *
 * @throws std::invalid_argument when ``node.op_type()`` differs from
 *         ``expected_op_type`` or when ``node.output_size() < 1``.
 */
void CheckNodeOpAndOutput(const NodeProto &node, const char *expected_op_type, const char *caller);

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
