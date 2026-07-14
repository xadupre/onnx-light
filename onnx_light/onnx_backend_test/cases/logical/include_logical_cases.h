// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``logical`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``And`` backend test node case(s).
void RegisterAndCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Or`` backend test node case(s).
void RegisterOrCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Xor`` backend test node case(s).
void RegisterXorCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Greater`` backend test node case(s).
void RegisterGreaterCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Less`` backend test node case(s).
void RegisterLessCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GreaterOrEqual`` backend test node case(s).
void RegisterGreaterOrEqualCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``LessOrEqual`` backend test node case(s).
void RegisterLessOrEqualCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Equal`` backend test node case(s).
void RegisterEqualCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Where`` backend test node case(s).
void RegisterWhereCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Not`` backend test node case(s).
void RegisterNotCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``IsNaN`` backend test node case(s).
void RegisterIsNaNCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``IsInf`` backend test node case(s).
void RegisterIsInfCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BitwiseAnd`` backend test node case(s).
void RegisterBitwiseAndCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BitwiseOr`` backend test node case(s).
void RegisterBitwiseOrCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BitwiseXor`` backend test node case(s).
void RegisterBitwiseXorCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BitwiseNot`` backend test node case(s).
void RegisterBitwiseNotCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BitShift`` backend test node case(s).
void RegisterBitShiftCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Collects all ``logical`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectLogicalTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                             TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
