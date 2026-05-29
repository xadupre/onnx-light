// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
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
 * @param op_type If non-empty, only schemas whose ``name()`` equals
 *        ``op_type`` are returned. When empty (default), schemas for every
 *        registered math operator are returned. Internally, the function uses
 *        a name → builder map so that only the matching builder runs and
 *        unrelated schemas are not constructed.
 * @param init_doc If true (default), each schema's documentation string is
 *        populated. When false, documentation strings are discarded (doc()
 *        returns ""), which can save memory when documentation is not needed.
 * @return Vector of LightOpSchema objects for the math domain, ordered by
 *         operator name and descending opset version.
 */
std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory(const std::string &op_type = "",
                                                              bool init_doc = true);

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
