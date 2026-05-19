// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <string>
#include <vector>

#include "onnx_op/op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

const std::string &OnnxOpMathDomain();

void RegisterOnnxOpMathOperatorSetSchema(int target_version = 0, bool fail_duplicate_schema = true);

void DeregisterOnnxOpMathOperatorSetSchema();

const MathOpSchema *
GetOnnxOpMathSchema(const std::string &op_type,
                    int max_inclusive_version = std::numeric_limits<int>::max());

std::vector<MathOpSchema> GetAllOnnxOpMathSchemasWithHistory();

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
