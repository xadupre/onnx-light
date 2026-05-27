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

/// Collects all ``logical`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectLogicalTestCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
