// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``generator`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Constant`` backend test node case(s).
void RegisterConstantCases(std::vector<TestCase> &registry);

/// Collects all ``generator`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectGeneratorTestCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
