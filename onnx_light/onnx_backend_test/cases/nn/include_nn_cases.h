// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``nn`` (neural network) op
// category — exposed so individual cases live in separate translation units
// yet can be invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``AveragePool`` backend test node case(s).
void RegisterAveragePoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Attention`` backend test node case(s).
void RegisterAttentionCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BatchNormalization`` backend test node case(s).
void RegisterBatchNormalizationCases(std::vector<TestCase> &registry,
                                     TestMode mode = TestMode::TEST);

/// Registers the ``Col2Im`` backend test node case(s).
void RegisterCol2ImCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``DeformConv`` backend test node case(s).
void RegisterDeformConvCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Conv`` backend test node case(s).
void RegisterConvCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ConvInteger`` backend test node case(s).
void RegisterConvIntegerCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ConvTranspose`` backend test node case(s).
void RegisterConvTransposeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Dropout`` backend test node case(s).
void RegisterDropoutCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Flatten`` backend test node case(s).
void RegisterFlattenCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GlobalAveragePool`` backend test node case(s).
void RegisterGlobalAveragePoolCases(std::vector<TestCase> &registry,
                                    TestMode mode = TestMode::TEST);

/// Registers the ``GlobalLpPool`` backend test node case(s).
void RegisterGlobalLpPoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GlobalMaxPool`` backend test node case(s).
void RegisterGlobalMaxPoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GroupNormalization`` backend test node case(s).
void RegisterGroupNormalizationCases(std::vector<TestCase> &registry,
                                     TestMode mode = TestMode::TEST);

/// Registers the ``GRU`` backend test node case(s).
void RegisterGRUCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``InstanceNormalization`` backend test node case(s).
void RegisterInstanceNormalizationCases(std::vector<TestCase> &registry,
                                        TestMode mode = TestMode::TEST);

/// Registers the ``LayerNormalization`` backend test node case(s).
void RegisterLayerNormalizationCases(std::vector<TestCase> &registry,
                                     TestMode mode = TestMode::TEST);

/// Registers the ``LinearAttention`` backend test node case(s).
void RegisterLinearAttentionCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LRN`` backend test node case(s).
void RegisterLRNCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LpNormalization`` backend test node case(s).
void RegisterLpNormalizationCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LpPool`` backend test node case(s).
void RegisterLpPoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LSTM`` backend test node case(s).
void RegisterLSTMCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``MaxPool`` backend test node case(s).
void RegisterMaxPoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``MaxRoiPool`` backend test node case(s).
void RegisterMaxRoiPoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``MaxUnpool`` backend test node case(s).
void RegisterMaxUnpoolCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``MeanVarianceNormalization`` backend test node case(s).
void RegisterMeanVarianceNormalizationCases(std::vector<TestCase> &registry,
                                            TestMode mode = TestMode::TEST);

/// Registers the ``RNN`` backend test node case(s).
void RegisterRNNCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``RMSNormalization`` backend test node case(s).
void RegisterRMSNormalizationCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``RotaryEmbedding`` backend test node case(s).
void RegisterRotaryEmbeddingCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``CausalConvWithState`` backend test node case(s).
void RegisterCausalConvWithStateCases(std::vector<TestCase> &registry,
                                      TestMode mode = TestMode::TEST);

/// Collects all ``nn`` op category backend test node cases by invoking every
/// ``Register*Cases`` helper declared in this header.
void CollectNNTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                        TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
