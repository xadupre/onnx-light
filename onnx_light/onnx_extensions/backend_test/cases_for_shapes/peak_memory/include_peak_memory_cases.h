// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Backend test cases that exercise the peak-memory annotation metadata written
// by ``WritePeakMemoryToMetadata``. These live in their own
// ``cases_for_shapes`` subtree so callers can collect them independently from
// the broader shape-inference gallery.
// ---------------------------------------------------------------------------

/// Registers backend cases whose expected ``onnx_light.peak_memory`` node
/// metadata is pre-embedded in the model so tests can verify that
/// ``WritePeakMemoryToMetadata`` reproduces it after shape inference.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterPeakMemoryCases(std::vector<TestCase> &registry,
                                                           TestMode mode = TestMode::TEST);

/// Collects all peak-memory backend test cases by invoking every
/// ``Register*PeakMemory*Cases`` helper declared in this header.
void CollectPeakMemoryTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
