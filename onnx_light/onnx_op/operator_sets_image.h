// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace image {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported image operators
 * (e.g. ImageDecoder).
 *
 * @return Vector of LightOpSchema objects for the image domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpImageSchemasWithHistory(bool init_doc = true);

} // namespace image
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
