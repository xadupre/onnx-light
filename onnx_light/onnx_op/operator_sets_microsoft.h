// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace microsoft {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/// The com.microsoft operator domain string.
constexpr const char *kMicrosoftDomain = "com.microsoft";

/**
 * Returns the versioned schema history for lightweight operators declared in
 * the ``com.microsoft`` domain.
 *
 * @return Vector of LightOpSchema objects for the ``com.microsoft`` domain,
 *         ordered by operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpMicrosoftSchemasWithHistory(const std::string &op_type = "",
                                                                   bool init_doc = true);

} // namespace microsoft
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
