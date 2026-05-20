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

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory();

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
