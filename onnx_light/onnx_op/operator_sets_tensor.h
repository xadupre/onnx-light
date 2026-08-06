// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace tensor {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported tensor operators
 * (e.g. AffineGrid, BitCast, Cast, CastLike, Concat).
 *
 * @return Vector of LightOpSchema objects for the tensor domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpTensorSchemasWithHistory(const std::string &op_type = "",
                                                                bool init_doc = true);

} // namespace tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
