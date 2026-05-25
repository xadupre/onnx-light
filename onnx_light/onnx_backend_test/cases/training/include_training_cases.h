// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``training`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ai.onnx.preview.training::Adam`` backend test node case(s).
void RegisterAdamCases(std::vector<TestCase> &registry);

/// Collects all ``training`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectTrainingTestCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
