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
void RegisterAndCases(std::vector<TestCase> &registry);

/// Registers the ``Or`` backend test node case(s).
void RegisterOrCases(std::vector<TestCase> &registry);

/// Registers the ``Xor`` backend test node case(s).
void RegisterXorCases(std::vector<TestCase> &registry);

/// Registers the ``Greater`` backend test node case(s).
void RegisterGreaterCases(std::vector<TestCase> &registry);

/// Registers the ``Less`` backend test node case(s).
void RegisterLessCases(std::vector<TestCase> &registry);

/// Registers the ``GreaterOrEqual`` backend test node case(s).
void RegisterGreaterOrEqualCases(std::vector<TestCase> &registry);

/// Registers the ``Equal`` backend test node case(s).
void RegisterEqualCases(std::vector<TestCase> &registry);

/// Registers the ``Where`` backend test node case(s).
void RegisterWhereCases(std::vector<TestCase> &registry);

/// Registers the ``Not`` backend test node case(s).
void RegisterNotCases(std::vector<TestCase> &registry);

/// Registers the ``IsNaN`` backend test node case(s).
void RegisterIsNaNCases(std::vector<TestCase> &registry);

/// Registers the ``IsInf`` backend test node case(s).
void RegisterIsInfCases(std::vector<TestCase> &registry);

/// Registers the ``BitwiseAnd`` backend test node case(s).
void RegisterBitwiseAndCases(std::vector<TestCase> &registry);

/// Registers the ``BitwiseOr`` backend test node case(s).
void RegisterBitwiseOrCases(std::vector<TestCase> &registry);

/// Registers the ``BitwiseXor`` backend test node case(s).
void RegisterBitwiseXorCases(std::vector<TestCase> &registry);

/// Registers the ``BitwiseNot`` backend test node case(s).
void RegisterBitwiseNotCases(std::vector<TestCase> &registry);

/// Registers the ``BitShift`` backend test node case(s).
void RegisterBitShiftCases(std::vector<TestCase> &registry);

/// Collects all ``logical`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectLogicalTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
