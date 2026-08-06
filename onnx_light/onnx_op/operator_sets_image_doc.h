// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::image {

/**
 * Returns the documentation string for the ImageDecoder operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the ImageDecoder operator.
 */
std::string MakeImageDecoderDoc(int since_version);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::image
