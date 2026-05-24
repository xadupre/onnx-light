// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``quantization`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``QuantizeLinear`` backend test node case(s).
void RegisterQuantizeLinearCases(std::vector<TestCase> &registry);

/// Collects all ``quantization`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectQuantizationTestCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
