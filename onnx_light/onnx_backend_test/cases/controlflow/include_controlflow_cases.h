// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``controlflow`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``If`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterIfCases(std::vector<TestCase> &registry,
                                                   TestMode mode = TestMode::TEST);

/// Registers the ``Loop`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterLoopCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Registers the ``Scan`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterScanCases(std::vector<TestCase> &registry,
                                                     TestMode mode = TestMode::TEST);

/// Collects all ``controlflow`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectControlflowTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                 TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
