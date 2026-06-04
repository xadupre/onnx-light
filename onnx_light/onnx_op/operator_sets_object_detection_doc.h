// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace object_detection {

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

} // namespace object_detection
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
