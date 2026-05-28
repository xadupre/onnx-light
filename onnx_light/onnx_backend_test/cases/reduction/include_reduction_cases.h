// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``reduction`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ArgMax`` backend test node case(s).
void RegisterArgMaxCases(std::vector<TestCase> &registry);

/// Registers the ``ArgMin`` backend test node case(s).
void RegisterArgMinCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceSum`` backend test node case(s).
void RegisterReduceSumCases(std::vector<TestCase> &registry);

/// Collects all ``reduction`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectReductionTestCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
