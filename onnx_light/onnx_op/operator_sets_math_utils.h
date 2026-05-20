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
namespace detail {

std::vector<std::string> FloatTypeStrings();
std::vector<std::string> NumericTypesForMathReductionStrings();
std::vector<std::string> NumericTypesForMathReductionIr4Strings();
std::vector<std::string> AllNumericTypesStrings();
std::vector<std::string> AllNumericTypesIr4Strings();

std::string MakeElementwiseMathDoc(const char *math_name, int since_version);

} // namespace detail
} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
