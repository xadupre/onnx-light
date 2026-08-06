// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``sequence`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``SequenceConstruct`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceConstructCases(std::vector<TestCase> &registry,
                                                                  TestMode mode = TestMode::TEST);

/// Registers the ``SequenceEmpty`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceEmptyCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``ConcatFromSequence`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConcatFromSequenceCases(std::vector<TestCase> &registry,
                                                                   TestMode mode = TestMode::TEST);

/// Registers the ``SequenceLength`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceLengthCases(std::vector<TestCase> &registry,
                                                               TestMode mode = TestMode::TEST);

/// Registers the ``SequenceErase`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceEraseCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers the ``SequenceAt`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceAtCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Registers the ``SequenceInsert`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceInsertCases(std::vector<TestCase> &registry,
                                                               TestMode mode = TestMode::TEST);

/// Registers the ``SequenceMap`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSequenceMapCases(std::vector<TestCase> &registry,
                                                            TestMode mode = TestMode::TEST);

/// Registers the ``SplitToSequence`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSplitToSequenceCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Collects all ``sequence`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectSequenceTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                              TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
