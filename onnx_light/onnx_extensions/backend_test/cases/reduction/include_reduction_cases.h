// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``reduction`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ArgMax`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterArgMaxCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``ArgMin`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterArgMinCases(std::vector<TestCase> &registry,
                                                       TestMode mode = TestMode::TEST);

/// Registers the ``ReduceSum`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceSumCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``ReduceMax`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceMaxCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``ReduceMean`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceMeanCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``ReduceMin`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceMinCases(std::vector<TestCase> &registry,
                                                          TestMode mode = TestMode::TEST);

/// Registers the ``ReduceProd`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceProdCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``ReduceL1`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceL1Cases(std::vector<TestCase> &registry,
                                                         TestMode mode = TestMode::TEST);

/// Registers the ``ReduceL2`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceL2Cases(std::vector<TestCase> &registry,
                                                         TestMode mode = TestMode::TEST);

/// Registers the ``ReduceLogSum`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceLogSumCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``ReduceLogSumExp`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceLogSumExpCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers the ``ReduceSumSquare`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReduceSumSquareCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Collects all ``reduction`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                               TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
