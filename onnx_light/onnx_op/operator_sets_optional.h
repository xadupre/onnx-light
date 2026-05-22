// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace optional {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported optional operators
 * (Optional, OptionalHasElement, OptionalGetElement).
 *
 * @return Vector of LightOpSchema objects for the optional operators, ordered
 *         by operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpOptionalSchemasWithHistory();

} // namespace optional
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
