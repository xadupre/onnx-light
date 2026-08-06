// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``image`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ai.onnx::ImageDecoder`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterImageDecoderCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Collects all ``image`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectImageTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                           TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
