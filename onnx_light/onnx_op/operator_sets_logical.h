// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {

using FormalParameter = onnx_op::math::FormalParameter;
using LightOpSchema = onnx_op::math::LightOpSchema;
using TypeConstraintParam = onnx_op::math::TypeConstraintParam;

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory();

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
