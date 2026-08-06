// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op {

using namespace ONNX_LIGHT_NAMESPACE::core::schema;
namespace training {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema;

/// The ai.onnx.preview.training operator domain string.
constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

/**
 * Returns the versioned schema history for all supported preview training
 * operators (e.g. Gradient, Adam) in the ai.onnx.preview.training domain.
 *
 * Preview training operators only have a single version (1); see the upstream
 * ONNX convention which keeps preview specs at version 1 even when revised.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx.preview.training
 *         domain, ordered by operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpTrainingSchemasWithHistory(const std::string &op_type = "",
                                                                  bool init_doc = true);

} // namespace training
} // namespace ONNX_LIGHT_NAMESPACE::onnx_op
