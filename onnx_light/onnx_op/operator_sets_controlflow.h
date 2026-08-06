// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace controlflow {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported control-flow
 * operators (e.g. If).
 *
 * @return Vector of LightOpSchema objects for the control-flow domain,
 *         ordered by operator name and descending opset version.
 */
std::vector<LightOpSchema>
GetAllOnnxOpControlflowSchemasWithHistory(const std::string &op_type = "", bool init_doc = true);

} // namespace controlflow
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
