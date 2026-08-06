// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/light_op_schema/light_op_schema.h"
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace logical {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported logical operators
 * (e.g. And, Or, Xor, Not, Greater, GreaterOrEqual, Less, Equal, BitShift,
 * BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot).
 *
 * @return Vector of LightOpSchema objects for the logical domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory(const std::string &op_type = "",
                                                                 bool init_doc = true);

} // namespace logical
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
