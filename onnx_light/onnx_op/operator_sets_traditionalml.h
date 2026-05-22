// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported traditional ML
 * operators (e.g. LabelEncoder, ZipMap) in the ai.onnx.ml domain.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx.ml domain, ordered
 *         by operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpTraditionalMLSchemasWithHistory(bool init_doc = true);

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
