// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace text {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported text operators
 * (e.g. RegexFullMatch, StringConcat, StringNormalizer, StringSplit, TfIdfVectorizer).
 *
 * @return Vector of LightOpSchema objects for the text domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpTextSchemasWithHistory(const std::string &op_type = "",
                                                              bool init_doc = true);

} // namespace text
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
