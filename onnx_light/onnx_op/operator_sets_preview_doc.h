// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace preview {

/**
 * Returns the documentation string for the FlexAttention operator.
 *
 * @return Documentation string for the FlexAttention operator.
 */
std::string MakeFlexAttentionDoc();

} // namespace preview
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
