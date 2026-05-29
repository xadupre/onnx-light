// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported tensor operators
 * (e.g. AffineGrid, Cast, Concat).
 *
 * @return Vector of LightOpSchema objects for the tensor domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory(bool init_doc = true,
                                                                const std::string &op_type = "");

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
