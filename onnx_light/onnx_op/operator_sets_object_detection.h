// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace object_detection {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported object-detection
 * operators (currently RoiAlign) in the ai.onnx domain.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx domain, ordered
 *         by operator name and descending opset version.
 */
std::vector<LightOpSchema>
GetAllOnnxOpObjectDetectionSchemasWithHistory(bool init_doc = true,
                                              const std::string &op_type = "");

} // namespace object_detection
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
