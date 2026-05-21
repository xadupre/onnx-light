// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace sequence {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

std::vector<LightOpSchema> GetAllOnnxOpSequenceSchemasWithHistory();

} // namespace sequence
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
