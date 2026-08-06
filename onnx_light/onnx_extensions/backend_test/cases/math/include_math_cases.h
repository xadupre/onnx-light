// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``math`` op category — exposed so
// individual cases live in separate translation units yet can be invoked from
// ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Abs`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAbsCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Acos`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAcosCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Acosh`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAcoshCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``Asin`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAsinCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Asinh`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAsinhCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``Atan`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAtanCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Atanh`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAtanhCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``Cos`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCosCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Cosh`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCoshCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``CumSum`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCumSumCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``CumProd`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCumProdCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``DFT`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDFTCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``STFT`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSTFTCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Det`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDetCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Sigmoid`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSigmoidCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``Softmax`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSoftmaxCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``LogSoftmax`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLogSoftmaxCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``SoftmaxCrossEntropyLoss`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterSoftmaxCrossEntropyLossCases(std::vector<TestCase> &registry,
                                     TestMode mode = TestMode::TEST);

/// Registers the ``NegativeLogLikelihoodLoss`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterNegativeLogLikelihoodLossCases(std::vector<TestCase> &registry,
                                       TestMode mode = TestMode::TEST);

/// Registers the ``Softplus`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSoftplusCases(std::vector<TestCase> &registry,
                                                         TestMode mode = TestMode::TEST);

/// Registers the ``Softsign`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSoftsignCases(std::vector<TestCase> &registry,
                                                         TestMode mode = TestMode::TEST);

/// Registers the ``Sin`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSinCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Sinh`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSinhCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Sqrt`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSqrtCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Tan`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterTanCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Tanh`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterTanhCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``ThresholdedRelu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterThresholdedReluCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``Relu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReluCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``LeakyRelu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLeakyReluCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``Elu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterEluCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Celu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCeluCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Gelu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGeluCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Selu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSeluCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Swish`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSwishCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``SwiGLU`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSwiGLUCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``Mish`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMishCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Add`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAddCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Sub`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSubCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Sum`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSumCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Mul`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMulCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Neg`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterNegCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Pow`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterPowCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``PRelu`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterPReluCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``Div`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDivCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Mod`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterModCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Einsum`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterEinsumCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``Exp`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterExpCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Erf`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterErfCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Sign`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSignCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Gemm`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGemmCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``MatMul`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMatMulCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``MatMulInteger`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMatMulIntegerCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``Max`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMaxCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Min`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMinCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``Mean`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMeanCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``MelWeightMatrix`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMelWeightMatrixCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``Log`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLogCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``BlackmanWindow`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterBlackmanWindowCases(std::vector<TestCase> &registry,
                                                               TestMode mode = TestMode::TEST);

/// Registers the ``Floor`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterFloorCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``Ceil`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCeilCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Clip`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterClipCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Reciprocal`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReciprocalCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``Round`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRoundCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``HannWindow`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterHannWindowCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``HammingWindow`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterHammingWindowCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``HardSigmoid`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterHardSigmoidCases(std::vector<TestCase> &registry,
                                                            TestMode mode = TestMode::TEST);

/// Registers the ``HardSwish`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterHardSwishCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``Hardmax`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterHardmaxCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``Shrink`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterShrinkCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``TopK`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterTopKCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Collects all ``math`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectMathTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                          TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
