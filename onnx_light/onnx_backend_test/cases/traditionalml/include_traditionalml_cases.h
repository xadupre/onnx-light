// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``traditionalml`` op category
// (``ai.onnx.ml`` domain) — exposed so individual cases live in separate
// translation units yet can be invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Binarizer`` backend test node case(s).
void RegisterBinarizerCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``CastMap`` backend test node case(s).
void RegisterCastMapCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``CategoryMapper`` backend test node case(s).
void RegisterCategoryMapperCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``DictVectorizer`` backend test node case(s).
void RegisterDictVectorizerCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``FeatureVectorizer`` backend test node case(s).
void RegisterFeatureVectorizerCases(std::vector<TestCase> &registry,
                                    TestMode mode = TestMode::TEST);

/// Registers the ``Imputer`` backend test node case(s).
void RegisterImputerCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ArrayFeatureExtractor`` backend test node case(s).
void RegisterArrayFeatureExtractorCases(std::vector<TestCase> &registry,
                                        TestMode mode = TestMode::TEST);

/// Registers the ``LabelEncoder`` backend test node case(s).
void RegisterLabelEncoderCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LinearClassifier`` backend test node case(s).
void RegisterLinearClassifierCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LinearRegressor`` backend test node case(s).
void RegisterLinearRegressorCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Normalizer`` backend test node case(s).
void RegisterNormalizerCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``OneHotEncoder`` backend test node case(s).
void RegisterOneHotEncoderCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``SVMClassifier`` backend test node case(s).
void RegisterSVMClassifierCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``SVMRegressor`` backend test node case(s).
void RegisterSVMRegressorCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Scaler`` backend test node case(s).
void RegisterScalerCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``TreeEnsemble`` backend test node case(s).
void RegisterTreeEnsembleCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``TreeEnsembleClassifier`` backend test node case(s).
void RegisterTreeEnsembleClassifierCases(std::vector<TestCase> &registry,
                                         TestMode mode = TestMode::TEST);

/// Registers the ``TreeEnsembleRegressor`` backend test node case(s).
void RegisterTreeEnsembleRegressorCases(std::vector<TestCase> &registry,
                                        TestMode mode = TestMode::TEST);

/// Registers the ``ZipMap`` backend test node case(s).
void RegisterZipMapCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Collects all ``traditionalml`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
ONNX_LIGHT_BACKEND_TEST_API void CollectTraditionalMLTestCases(std::vector<TestCase> &registry,
                                                               const std::string &op_type = "",
                                                               TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
