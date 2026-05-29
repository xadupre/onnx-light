// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace text {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported text operators
 * (e.g. RegexFullMatch, StringConcat, StringNormalizer, StringSplit).
 *
 * @return Vector of LightOpSchema objects for the text domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpTextSchemasWithHistory(bool init_doc = true,
                                                              const std::string &op_type = "");

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
