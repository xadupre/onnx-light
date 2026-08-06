// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::optional {

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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::optional
