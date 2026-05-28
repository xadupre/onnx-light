// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectMathTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (MatchOpTypeFilter(op_type, "Abs"))
    RegisterAbsCases(registry);
  if (MatchOpTypeFilter(op_type, "Acos"))
    RegisterAcosCases(registry);
  if (MatchOpTypeFilter(op_type, "Acosh"))
    RegisterAcoshCases(registry);
  if (MatchOpTypeFilter(op_type, "Asin"))
    RegisterAsinCases(registry);
  if (MatchOpTypeFilter(op_type, "Asinh"))
    RegisterAsinhCases(registry);
  if (MatchOpTypeFilter(op_type, "Atan"))
    RegisterAtanCases(registry);
  if (MatchOpTypeFilter(op_type, "Atanh"))
    RegisterAtanhCases(registry);
  if (MatchOpTypeFilter(op_type, "Add"))
    RegisterAddCases(registry);
  if (MatchOpTypeFilter(op_type, "Sub"))
    RegisterSubCases(registry);
  if (MatchOpTypeFilter(op_type, "Mul"))
    RegisterMulCases(registry);
  if (MatchOpTypeFilter(op_type, "Div"))
    RegisterDivCases(registry);
  if (MatchOpTypeFilter(op_type, "BlackmanWindow"))
    RegisterBlackmanWindowCases(registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
