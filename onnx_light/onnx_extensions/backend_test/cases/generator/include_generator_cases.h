// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``generator`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Bernoulli`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterBernoulliCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``Constant`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConstantCases(std::vector<TestCase> &registry,
                                                         TestMode mode = TestMode::TEST);

/// Registers the ``ConstantOfShape`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConstantOfShapeCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``EyeLike`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterEyeLikeCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Registers the ``RandomNormal`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRandomNormalCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``RandomNormalLike`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRandomNormalLikeCases(std::vector<TestCase> &registry,
                                                                 TestMode mode = TestMode::TEST);

/// Registers the ``RandomUniform`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRandomUniformCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``RandomUniformLike`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRandomUniformLikeCases(std::vector<TestCase> &registry,
                                                                  TestMode mode = TestMode::TEST);

/// Registers the ``Range`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRangeCases(std::vector<TestCase> &registry,
                                                      TestMode mode = TestMode::TEST);

/// Registers the ``Multinomial`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMultinomialCases(std::vector<TestCase> &registry,
                                                            TestMode mode = TestMode::TEST);

/// Registers the ``ai.rt::DelayedInitializer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDelayedInitializerCases(std::vector<TestCase> &registry,
                                                                   TestMode mode = TestMode::TEST);

/// Collects all ``generator`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectGeneratorTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                               TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
