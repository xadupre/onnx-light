// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``traditionalml`` op category
// (``ai.onnx.ml`` domain) — exposed so individual cases live in separate
// translation units yet can be invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Binarizer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterBinarizerCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``CastMap`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCastMapCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``CategoryMapper`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCategoryMapperCases(std::vector<TestCase> &registry,
                                                               TestMode mode = TestMode::TEST);

/// Registers the ``DictVectorizer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDictVectorizerCases(std::vector<TestCase> &registry,
                                                               TestMode mode = TestMode::TEST);

/// Registers the ``FeatureVectorizer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterFeatureVectorizerCases(std::vector<TestCase> &registry,
                                                                  TestMode mode = TestMode::TEST);

/// Registers the ``Imputer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterImputerCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``ArrayFeatureExtractor`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterArrayFeatureExtractorCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LabelEncoder`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLabelEncoderCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``LinearClassifier`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLinearClassifierCases(std::vector<TestCase> &registry,
                                                                 TestMode mode = TestMode::TEST);

/// Registers the ``LinearRegressor`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLinearRegressorCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``Normalizer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterNormalizerCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``OneHotEncoder`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterOneHotEncoderCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``SVMClassifier`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSVMClassifierCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``SVMRegressor`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSVMRegressorCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``Scaler`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterScalerCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``TreeEnsemble`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterTreeEnsembleCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``TreeEnsembleClassifier`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTreeEnsembleClassifierCases(std::vector<TestCase> &registry,
                                    TestMode mode = TestMode::TEST);

/// Registers the ``TreeEnsembleRegressor`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterTreeEnsembleRegressorCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ZipMap`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterZipMapCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Collects all ``traditionalml`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectTraditionalMLTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                   TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
