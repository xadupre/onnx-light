// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace quantization {

/**
 * Returns the documentation string for the QuantizeLinear operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the QuantizeLinear operator.
 */
std::string MakeQuantizeLinearDoc(int since_version);

/**
 * Returns the documentation string for the DequantizeLinear operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the DequantizeLinear operator.
 */
std::string MakeDequantizeLinearDoc(int since_version);

/**
 * Returns the documentation string for the QLinearConv operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the QLinearConv operator.
 */
std::string MakeQLinearConvDoc(int since_version);

/**
 * Returns the documentation string for the QLinearMatMul operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the QLinearMatMul operator.
 */
std::string MakeQLinearMatMulDoc(int since_version);

} // namespace quantization
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
