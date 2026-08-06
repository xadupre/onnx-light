// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace nn {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/**
 * Returns the versioned schema history for all supported neural-network
 * operators (e.g. AveragePool, RNN, GRU, LSTM).
 *
 * @return Vector of LightOpSchema objects for the nn domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpNnSchemasWithHistory(const std::string &op_type = "",
                                                            bool init_doc = true);

} // namespace nn
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
