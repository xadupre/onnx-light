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
      {"Neg", &RegisterNegCases},
      {"Pow", &RegisterPowCases},
      {"PRelu", &RegisterPReluCases},
      {"Div", &RegisterDivCases},
      {"Mod", &RegisterModCases},
      {"Einsum", &RegisterEinsumCases},
      {"BlackmanWindow", &RegisterBlackmanWindowCases},
      {"Erf", &RegisterErfCases},
      {"HannWindow", &RegisterHannWindowCases},
      {"HammingWindow", &RegisterHammingWindowCases},
      {"Exp", &RegisterExpCases},
      {"Log", &RegisterLogCases},
      {"Gemm", &RegisterGemmCases},
      {"MatMul", &RegisterMatMulCases},
      {"Max", &RegisterMaxCases},
      {"Mean", &RegisterMeanCases},
      {"MelWeightMatrix", &RegisterMelWeightMatrixCases},
      {"Min", &RegisterMinCases},
      {"Cos", &RegisterCosCases},
      {"Cosh", &RegisterCoshCases},
      {"CumProd", &RegisterCumProdCases},
      {"CumSum", &RegisterCumSumCases},
      {"DFT", &RegisterDFTCases},
      {"STFT", &RegisterSTFTCases},
      {"Det", &RegisterDetCases},
      {"Sin", &RegisterSinCases},
      {"Sinh", &RegisterSinhCases},
      {"Sigmoid", &RegisterSigmoidCases},
      {"Softmax", &RegisterSoftmaxCases},
      {"SoftmaxCrossEntropyLoss", &RegisterSoftmaxCrossEntropyLossCases},
      {"NegativeLogLikelihoodLoss", &RegisterNegativeLogLikelihoodLossCases},
      {"Softplus", &RegisterSoftplusCases},
      {"Softsign", &RegisterSoftsignCases},
      {"Sqrt", &RegisterSqrtCases},
      {"Tan", &RegisterTanCases},
      {"Tanh", &RegisterTanhCases},
      {"ThresholdedRelu", &RegisterThresholdedReluCases},
      {"Selu", &RegisterSeluCases},
      {"Swish", &RegisterSwishCases},
      {"Mish", &RegisterMishCases},
      {"TopK", &RegisterTopKCases},
      {"Floor", &RegisterFloorCases},
      {"Ceil", &RegisterCeilCases},
      {"Clip", &RegisterClipCases},
      {"Round", &RegisterRoundCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
