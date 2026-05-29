// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace preview {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/// The ai.onnx.preview operator domain string.
constexpr const char *kOnnxPreviewDomain = "ai.onnx.preview";

/**
 * Returns the versioned schema history for all supported preview operators
 * (e.g. FlexAttention) in the ai.onnx.preview domain.
 *
 * Preview operators only have a single version (1); see the upstream ONNX
 * convention which keeps preview specs at version 1 even when revised.
 *
 * @return Vector of LightOpSchema objects for the ai.onnx.preview domain,
 *         ordered by operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpPreviewSchemasWithHistory(bool init_doc = true,
                                                                 const std::string &op_type = "");

} // namespace preview
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
