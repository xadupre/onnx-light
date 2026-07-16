// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

// Internal (library-private) header: the per-operator ``Register*`` backend
// test registration helpers. These are only ever invoked by the ``Collect*``
// aggregators inside ``lib_onnx_backend_test`` and are compiled with hidden
// visibility, so they are not exported from the shared library. This header is
// pulled in by the matching public header only when ONNX_LIGHT_BACKEND_TEST_INTERNAL is
// defined (i.e. while building the library itself).

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
