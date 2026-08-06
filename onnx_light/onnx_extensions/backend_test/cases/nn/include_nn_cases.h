// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``nn`` (neural network) op
// category — exposed so individual cases live in separate translation units
// yet can be invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``AveragePool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAveragePoolCases(std::vector<TestCase> &registry,
                                                            TestMode mode = TestMode::TEST);

/// Registers the ``Attention`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAttentionCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``BatchNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterBatchNormalizationCases(std::vector<TestCase> &registry,
                                                                   TestMode mode = TestMode::TEST);

/// Registers the ``Col2Im`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCol2ImCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``DeformConv`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDeformConvCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``Conv`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConvCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``ConvInteger`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConvIntegerCases(std::vector<TestCase> &registry,
                                                            TestMode mode = TestMode::TEST);

/// Registers the ``ConvTranspose`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConvTransposeCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``Dropout`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDropoutCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``Flatten`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterFlattenCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``GlobalAveragePool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGlobalAveragePoolCases(std::vector<TestCase> &registry,
                                                                  TestMode mode = TestMode::TEST);

/// Registers the ``GlobalLpPool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGlobalLpPoolCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``GlobalMaxPool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGlobalMaxPoolCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``GroupNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGroupNormalizationCases(std::vector<TestCase> &registry,
                                                                   TestMode mode = TestMode::TEST);

/// Registers the ``GRU`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterGRUCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``InstanceNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterInstanceNormalizationCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LayerNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLayerNormalizationCases(std::vector<TestCase> &registry,
                                                                   TestMode mode = TestMode::TEST);

/// Registers the ``LinearAttention`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLinearAttentionCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``LRN`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLRNCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``LpNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLpNormalizationCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``LpPool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLpPoolCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``LSTM`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLSTMCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``MaxPool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMaxPoolCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``MaxRoiPool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMaxRoiPoolCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``MaxUnpool`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMaxUnpoolCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``MeanVarianceNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterMeanVarianceNormalizationCases(std::vector<TestCase> &registry,
                                       TestMode mode = TestMode::TEST);

/// Registers the ``RNN`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRNNCases(std::vector<TestCase> &registry,
                                                    TestMode mode = TestMode::TEST);

/// Registers the ``RMSNormalization`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRMSNormalizationCases(std::vector<TestCase> &registry,
                                                                 TestMode mode = TestMode::TEST);

/// Registers the ``RotaryEmbedding`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRotaryEmbeddingCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``CausalConvWithState`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCausalConvWithStateCases(std::vector<TestCase> &registry,
                                                                    TestMode mode = TestMode::TEST);

/// Collects all ``nn`` op category backend test node cases by invoking every
/// ``Register*Cases`` helper declared in this header.
void CollectNNTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                        TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
