// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace nn {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported neural-network
 * operators (e.g. AveragePool, RNN, GRU, LSTM).
 *
 * @return Vector of LightOpSchema objects for the nn domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpNnSchemasWithHistory(bool init_doc = true,
                                                            const std::string &op_type = "");

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
