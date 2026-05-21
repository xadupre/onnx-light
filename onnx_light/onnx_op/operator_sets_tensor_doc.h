// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

/**
 * Returns the documentation string for the Cast operator at the given opset
 * version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Cast operator.
 */
std::string MakeCastDoc(int since_version);

/**
 * Returns the input type-constraint description for the Cast operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Cast input.
 */
std::string MakeCastInputTypeConstraintDescription(int since_version);

/**
 * Returns the output type-constraint description for the Cast operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Cast output.
 */
std::string MakeCastOutputTypeConstraintDescription(int since_version);

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
