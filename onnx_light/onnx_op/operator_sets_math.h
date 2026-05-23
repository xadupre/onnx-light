// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

using LightOpSchema = ONNX_LIGHT_NAMESPACE::onnx_op::LightOpSchema;

/**
 * Returns the versioned schema history for all supported math operators
 * (e.g. Abs, Add, Sin, Cos, Pow, BlackmanWindow, MatMul, Gemm).
 *
 * @return Vector of LightOpSchema objects for the math domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory(bool init_doc = true);

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
