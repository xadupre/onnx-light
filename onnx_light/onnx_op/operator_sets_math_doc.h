// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

/**
 * Returns the documentation string for an element-wise math operator at the
 * given opset version.
 *
 * @param op_type Operator name (e.g. "Add", "Mul").
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the operator.
 */
std::string MakeElementwiseMathDoc(const char *op_type, int since_version);

/**
 * Returns the documentation string for a unary element-wise math operator.
 *
 * @param op_type Operator name (e.g. "Sin", "Cos", "Abs").
 * @return Documentation string for the operator.
 */
std::string MakeUnaryMathDoc(const char *op_type);

/**
 * Returns the output description for a unary element-wise math operator.
 *
 * @param op_type Operator name (e.g. "Sin", "Cos", "Abs").
 * @return Output description string.
 */
std::string MakeUnaryMathOutputDescription(const char *op_type);

/**
 * Returns the documentation string for the BlackmanWindow operator.
 *
 * @return Documentation string for the BlackmanWindow operator.
 */
std::string MakeBlackmanWindowDoc();

/**
 * Returns the documentation string for the Pow operator.
 *
 * @return Documentation string for the Pow operator.
 */
std::string MakePowDoc();

/**
 * Returns the documentation string for the MatMul operator.
 *
 * @return Documentation string for the MatMul operator.
 */
std::string MakeMatMulDoc();

/**
 * Returns the documentation string for the Gemm operator at the given opset
 * version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Gemm operator.
 */
std::string MakeGemmDoc(int since_version);

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
