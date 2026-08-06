// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace optional {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported optional operators
 * (Optional, OptionalHasElement, OptionalGetElement).
 *
 * @return Vector of LightOpSchema objects for the optional operators, ordered
 *         by operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpOptionalSchemasWithHistory(const std::string &op_type = "",
                                                                  bool init_doc = true);

} // namespace optional
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
