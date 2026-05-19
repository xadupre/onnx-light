// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

void RegisterOnnxOpMathOperatorSetSchema(bool fail_duplicate_schema = true);

std::vector<OpSchema> GetAllOnnxOpMathSchemasWithHistory();

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
