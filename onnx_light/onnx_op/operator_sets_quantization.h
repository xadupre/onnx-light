// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace quantization {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported quantization
 * operators (e.g. QuantizeLinear, DequantizeLinear) in the ai.onnx domain.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx quantization
 *         operators, ordered by operator name and descending opset version.
 */
std::vector<LightOpSchema>
GetAllOnnxOpQuantizationSchemasWithHistory(const std::string &op_type = "", bool init_doc = true);

} // namespace quantization
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
