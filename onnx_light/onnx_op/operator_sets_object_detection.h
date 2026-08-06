// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace object_detection {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported object-detection
 * operators (currently RoiAlign) in the ai.onnx domain.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx domain, ordered
 *         by operator name and descending opset version.
 */
std::vector<LightOpSchema>
GetAllOnnxOpObjectDetectionSchemasWithHistory(const std::string &op_type = "",
                                              bool init_doc = true);

} // namespace object_detection
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
