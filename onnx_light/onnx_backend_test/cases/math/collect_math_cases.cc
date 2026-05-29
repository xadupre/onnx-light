// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectMathTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"Abs", &RegisterAbsCases},     {"Acos", &RegisterAcosCases},
      {"Acosh", &RegisterAcoshCases}, {"Asin", &RegisterAsinCases},
      {"Asinh", &RegisterAsinhCases}, {"Atan", &RegisterAtanCases},
      {"Atanh", &RegisterAtanhCases}, {"Add", &RegisterAddCases},
      {"Sub", &RegisterSubCases},     {"Mul", &RegisterMulCases},
      {"Div", &RegisterDivCases},     {"BlackmanWindow", &RegisterBlackmanWindowCases},
      {"Cos", &RegisterCosCases},     {"Cosh", &RegisterCoshCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
