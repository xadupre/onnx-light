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

std::string MakeElementwiseMathDoc(const char *op_type, int since_version);
std::string MakeUnaryMathDoc(const char *op_type);
std::string MakeUnaryMathOutputDescription(const char *op_type);
std::string MakeBlackmanWindowDoc();
std::string MakePowDoc();

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
