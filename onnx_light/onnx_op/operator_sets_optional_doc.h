// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace optional {

/**
 * Returns the documentation string for the Optional operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to return the documentation.
 * @return Documentation string for the Optional operator.
 */
std::string MakeOptionalDoc(int since_version);

/**
 * Returns the documentation string for the OptionalHasElement operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to return the documentation.
 * @return Documentation string for the OptionalHasElement operator.
 */
std::string MakeOptionalHasElementDoc(int since_version);

/**
 * Returns the documentation string for the OptionalGetElement operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to return the documentation.
 * @return Documentation string for the OptionalGetElement operator.
 */
std::string MakeOptionalGetElementDoc(int since_version);

} // namespace optional
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
