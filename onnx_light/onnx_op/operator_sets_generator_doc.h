// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace generator {

/**
 * Returns the documentation string for the Constant operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Constant operator.
 */
std::string MakeConstantDoc(int since_version);

/**
 * Returns the documentation string for the ConstantOfShape operator at the
 * given opset version. The documentation has been stable since opset 9.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the ConstantOfShape operator.
 */
std::string MakeConstantOfShapeDoc(int since_version);

} // namespace generator
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
