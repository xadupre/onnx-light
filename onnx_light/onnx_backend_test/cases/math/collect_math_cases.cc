// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectMathTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"Abs", &RegisterAbsCases},
      {"Acos", &RegisterAcosCases},
      {"Acosh", &RegisterAcoshCases},
      {"Asin", &RegisterAsinCases},
      {"Asinh", &RegisterAsinhCases},
      {"Atan", &RegisterAtanCases},
      {"Atanh", &RegisterAtanhCases},
      {"Add", &RegisterAddCases},
      {"Sub", &RegisterSubCases},
      {"Sum", &RegisterSumCases},
      {"Mul", &RegisterMulCases},
      {"Div", &RegisterDivCases},
      {"Einsum", &RegisterEinsumCases},
      {"BlackmanWindow", &RegisterBlackmanWindowCases},
      {"HannWindow", &RegisterHannWindowCases},
      {"HammingWindow", &RegisterHammingWindowCases},
      {"Exp", &RegisterExpCases},
      {"Log", &RegisterLogCases},
      {"Gemm", &RegisterGemmCases},
      {"MatMul", &RegisterMatMulCases},
      {"Cos", &RegisterCosCases},
      {"Cosh", &RegisterCoshCases},
      {"CumProd", &RegisterCumProdCases},
      {"CumSum", &RegisterCumSumCases},
      {"Sin", &RegisterSinCases},
      {"Sinh", &RegisterSinhCases},
      {"Sigmoid", &RegisterSigmoidCases},
      {"Softmax", &RegisterSoftmaxCases},
      {"Tan", &RegisterTanCases},
      {"Tanh", &RegisterTanhCases},
      {"Floor", &RegisterFloorCases},
      {"Ceil", &RegisterCeilCases},
      {"Round", &RegisterRoundCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
