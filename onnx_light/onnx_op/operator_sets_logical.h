// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_op/light_op_schema.h"
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported logical operators
 * (e.g. And, Or, Xor, Not, Greater, Less, Equal).
 *
 * @return Vector of LightOpSchema objects for the logical domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory(bool init_doc = true);

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
