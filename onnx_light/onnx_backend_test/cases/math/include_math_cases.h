// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``math`` op category — exposed so
// individual cases live in separate translation units yet can be invoked from
// ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Abs`` backend test node case(s).
void RegisterAbsCases(std::vector<TestCase> &registry);

/// Registers the ``Add`` backend test node case(s).
void RegisterAddCases(std::vector<TestCase> &registry);

/// Registers the ``BlackmanWindow`` backend test node case(s).
void RegisterBlackmanWindowCases(std::vector<TestCase> &registry);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
