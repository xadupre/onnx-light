// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``math`` op category — exposed so
// individual cases live in separate translation units yet can be invoked from
// ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Abs`` backend test node case(s).
void RegisterAbsCases(std::vector<TestCase> &registry);

/// Registers the ``Acos`` backend test node case(s).
void RegisterAcosCases(std::vector<TestCase> &registry);

/// Registers the ``Acosh`` backend test node case(s).
void RegisterAcoshCases(std::vector<TestCase> &registry);

/// Registers the ``Asin`` backend test node case(s).
void RegisterAsinCases(std::vector<TestCase> &registry);

/// Registers the ``Asinh`` backend test node case(s).
void RegisterAsinhCases(std::vector<TestCase> &registry);

/// Registers the ``Atan`` backend test node case(s).
void RegisterAtanCases(std::vector<TestCase> &registry);

/// Registers the ``Atanh`` backend test node case(s).
void RegisterAtanhCases(std::vector<TestCase> &registry);

/// Registers the ``Cos`` backend test node case(s).
void RegisterCosCases(std::vector<TestCase> &registry);

/// Registers the ``Cosh`` backend test node case(s).
void RegisterCoshCases(std::vector<TestCase> &registry);

/// Registers the ``CumSum`` backend test node case(s).
void RegisterCumSumCases(std::vector<TestCase> &registry);

/// Registers the ``CumProd`` backend test node case(s).
void RegisterCumProdCases(std::vector<TestCase> &registry);

/// Registers the ``DFT`` backend test node case(s).
void RegisterDFTCases(std::vector<TestCase> &registry);

/// Registers the ``STFT`` backend test node case(s).
void RegisterSTFTCases(std::vector<TestCase> &registry);

/// Registers the ``Det`` backend test node case(s).
void RegisterDetCases(std::vector<TestCase> &registry);

/// Registers the ``Sigmoid`` backend test node case(s).
void RegisterSigmoidCases(std::vector<TestCase> &registry);

/// Registers the ``Softmax`` backend test node case(s).
void RegisterSoftmaxCases(std::vector<TestCase> &registry);

/// Registers the ``SoftmaxCrossEntropyLoss`` backend test node case(s).
void RegisterSoftmaxCrossEntropyLossCases(std::vector<TestCase> &registry);

/// Registers the ``NegativeLogLikelihoodLoss`` backend test node case(s).
void RegisterNegativeLogLikelihoodLossCases(std::vector<TestCase> &registry);

/// Registers the ``Softplus`` backend test node case(s).
void RegisterSoftplusCases(std::vector<TestCase> &registry);

/// Registers the ``Softsign`` backend test node case(s).
void RegisterSoftsignCases(std::vector<TestCase> &registry);

/// Registers the ``Sin`` backend test node case(s).
void RegisterSinCases(std::vector<TestCase> &registry);

/// Registers the ``Sinh`` backend test node case(s).
void RegisterSinhCases(std::vector<TestCase> &registry);

/// Registers the ``Sqrt`` backend test node case(s).
void RegisterSqrtCases(std::vector<TestCase> &registry);

/// Registers the ``Tan`` backend test node case(s).
void RegisterTanCases(std::vector<TestCase> &registry);

/// Registers the ``Tanh`` backend test node case(s).
void RegisterTanhCases(std::vector<TestCase> &registry);

/// Registers the ``ThresholdedRelu`` backend test node case(s).
void RegisterThresholdedReluCases(std::vector<TestCase> &registry);

/// Registers the ``Relu`` backend test node case(s).
void RegisterReluCases(std::vector<TestCase> &registry);

/// Registers the ``Elu`` backend test node case(s).
void RegisterEluCases(std::vector<TestCase> &registry);

/// Registers the ``Celu`` backend test node case(s).
void RegisterCeluCases(std::vector<TestCase> &registry);

/// Registers the ``Gelu`` backend test node case(s).
void RegisterGeluCases(std::vector<TestCase> &registry);

/// Registers the ``Selu`` backend test node case(s).
void RegisterSeluCases(std::vector<TestCase> &registry);

/// Registers the ``Swish`` backend test node case(s).
void RegisterSwishCases(std::vector<TestCase> &registry);

/// Registers the ``Add`` backend test node case(s).
void RegisterAddCases(std::vector<TestCase> &registry);

/// Registers the ``Sub`` backend test node case(s).
void RegisterSubCases(std::vector<TestCase> &registry);

/// Registers the ``Sum`` backend test node case(s).
void RegisterSumCases(std::vector<TestCase> &registry);

/// Registers the ``Mul`` backend test node case(s).
void RegisterMulCases(std::vector<TestCase> &registry);

/// Registers the ``Neg`` backend test node case(s).
void RegisterNegCases(std::vector<TestCase> &registry);

/// Registers the ``Pow`` backend test node case(s).
void RegisterPowCases(std::vector<TestCase> &registry);

/// Registers the ``PRelu`` backend test node case(s).
void RegisterPReluCases(std::vector<TestCase> &registry);

/// Registers the ``Div`` backend test node case(s).
void RegisterDivCases(std::vector<TestCase> &registry);

/// Registers the ``Mod`` backend test node case(s).
void RegisterModCases(std::vector<TestCase> &registry);

/// Registers the ``Einsum`` backend test node case(s).
void RegisterEinsumCases(std::vector<TestCase> &registry);

/// Registers the ``Exp`` backend test node case(s).
void RegisterExpCases(std::vector<TestCase> &registry);

/// Registers the ``Erf`` backend test node case(s).
void RegisterErfCases(std::vector<TestCase> &registry);

/// Registers the ``Gemm`` backend test node case(s).
void RegisterGemmCases(std::vector<TestCase> &registry);

/// Registers the ``MatMul`` backend test node case(s).
void RegisterMatMulCases(std::vector<TestCase> &registry);

/// Registers the ``Max`` backend test node case(s).
void RegisterMaxCases(std::vector<TestCase> &registry);

/// Registers the ``Min`` backend test node case(s).
void RegisterMinCases(std::vector<TestCase> &registry);

/// Registers the ``Mean`` backend test node case(s).
void RegisterMeanCases(std::vector<TestCase> &registry);

/// Registers the ``MelWeightMatrix`` backend test node case(s).
void RegisterMelWeightMatrixCases(std::vector<TestCase> &registry);

/// Registers the ``Log`` backend test node case(s).
void RegisterLogCases(std::vector<TestCase> &registry);

/// Registers the ``BlackmanWindow`` backend test node case(s).
void RegisterBlackmanWindowCases(std::vector<TestCase> &registry);

/// Registers the ``Floor`` backend test node case(s).
void RegisterFloorCases(std::vector<TestCase> &registry);

/// Registers the ``Ceil`` backend test node case(s).
void RegisterCeilCases(std::vector<TestCase> &registry);

/// Registers the ``Clip`` backend test node case(s).
void RegisterClipCases(std::vector<TestCase> &registry);

/// Registers the ``Round`` backend test node case(s).
void RegisterRoundCases(std::vector<TestCase> &registry);

/// Registers the ``HannWindow`` backend test node case(s).
void RegisterHannWindowCases(std::vector<TestCase> &registry);

/// Registers the ``HammingWindow`` backend test node case(s).
void RegisterHammingWindowCases(std::vector<TestCase> &registry);

/// Registers the ``TopK`` backend test node case(s).
void RegisterTopKCases(std::vector<TestCase> &registry);

/// Collects all ``math`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectMathTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
