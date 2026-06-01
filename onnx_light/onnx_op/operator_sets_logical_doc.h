// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {

/**
 * Returns the documentation string for a binary logical operator at the given
 * opset version.
 *
 * @param op_type Operator name (e.g. "And", "Or", "Xor", "Greater", "Less").
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the operator.
 */
std::string MakeBinaryLogicalOperatorDoc(const char *op_type, int since_version);

/**
 * Returns the documentation string for the Not operator.
 *
 * @return Documentation string for the Not operator.
 */
std::string MakeNotLogicalOperatorDoc();

/**
 * Returns the documentation string for the ``Where`` operator.
 *
 * @return Documentation string for ``Where``.
 */
std::string MakeWhereOperatorDoc();

/**
 * Returns the documentation string for a binary bitwise operator
 * (``BitwiseAnd``, ``BitwiseOr``, ``BitwiseXor``).
 *
 * @param op_type Operator name (e.g. ``"BitwiseAnd"``).
 * @return Documentation string for the operator.
 */
std::string MakeBinaryBitwiseOperatorDoc(const char *op_type);

/**
 * Returns the documentation string for the ``BitwiseNot`` operator.
 *
 * @return Documentation string for the ``BitwiseNot`` operator.
 */
std::string MakeBitwiseNotOperatorDoc();

/**
 * Returns the documentation string for the ``BitShift`` operator
 * (opset 11), matching the upstream ONNX schema text including the
 * appended multidirectional-broadcasting documentation block.
 *
 * @return Documentation string for ``BitShift``.
 */
std::string MakeBitShiftOperatorDoc();

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
