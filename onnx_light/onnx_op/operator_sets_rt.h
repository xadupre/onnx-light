// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace rt {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/// The ai.rt operator domain string.
constexpr const char *kAiRtDomain = "ai.rt";

/**
 * Returns the versioned schema history for lightweight runtime-only operators
 * declared in the ``ai.rt`` domain.
 *
 * @return Vector of LightOpSchema objects for the ``ai.rt`` domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpRtSchemasWithHistory(const std::string &op_type = "",
                                                            bool init_doc = true);

} // namespace rt
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
