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
 * Returns the documentation string for the HannWindow operator.
 *
 * @return Documentation string for the HannWindow operator.
 */
std::string MakeHannWindowDoc();

/**
 * Returns the documentation string for the HammingWindow operator.
 *
 * @return Documentation string for the HammingWindow operator.
 */
std::string MakeHammingWindowDoc();

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

/**
 * Returns the documentation string for the CumSum operator. The wording is
 * stable across opsets 11 and 14 (only the type constraint widens).
 *
 * @return Documentation string for the CumSum operator.
 */
std::string MakeCumSumDoc();

/**
 * Returns the documentation string for the CumProd operator (opset 26).
 *
 * @return Documentation string for the CumProd operator.
 */
std::string MakeCumProdDoc();

/**
 * Returns the documentation string for the Sum operator at the given opset
 * version. Opsets 1 and 6 share the same wording (variadic element-wise sum
 * with no broadcasting); opsets 8 and 13 share the wording that exposes
 * NumPy-style multidirectional broadcasting.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Sum operator.
 */
std::string MakeSumDoc(int since_version);

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
