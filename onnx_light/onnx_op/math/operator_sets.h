// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

ONNX_API const std::string &OnnxOpMathDomain();

ONNX_API void RegisterOnnxOpMathOperatorSetSchema(int target_version = 0,
                                                  bool fail_duplicate_schema = true);

ONNX_API void DeregisterOnnxOpMathOperatorSetSchema();

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
