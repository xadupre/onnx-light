// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::object_detection {

/**
 * Returns the documentation string for the RoiAlign operator.
 *
 * @return Documentation string for the RoiAlign operator.
 */
std::string MakeRoiAlignDoc();

/**
 * Returns the documentation string for the NonMaxSuppression operator.
 *
 * Matches the upstream ONNX NonMaxSuppression docs (since opset 10).
 *
 * @return Documentation string for the NonMaxSuppression operator.
 */
std::string MakeNonMaxSuppressionDoc();

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::object_detection
