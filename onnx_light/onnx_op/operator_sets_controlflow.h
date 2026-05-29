// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

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
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
