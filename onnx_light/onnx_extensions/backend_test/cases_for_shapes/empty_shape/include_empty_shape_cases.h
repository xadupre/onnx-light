// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Backend test cases that exercise operators on tensors whose shape is
// "empty" — either rank-0 (the shape vector itself is empty, ``{}``) or a
// shape with a zero-sized dimension (e.g. ``{0}`` or ``{0, 3}``) yielding a
// tensor with no elements. These cases live in their own ``cases_for_shapes``
// subtree so they are easy to discover and extend with other shape-oriented
// scenarios later.
// ---------------------------------------------------------------------------

/// Registers backend test cases that add tensors with empty shapes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterAddEmptyShapeCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers backend test cases that subtract tensors with empty shapes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterSubEmptyShapeCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers backend test cases that multiply tensors with empty shapes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterMulEmptyShapeCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers backend test cases that divide tensors with empty shapes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterDivEmptyShapeCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

/// Registers backend test cases that apply ``PRelu`` to tensors with empty
/// shapes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterPReluEmptyShapeCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

/// Registers backend test cases that run the ``Compress`` node on tensors
/// with empty shapes (and/or producing outputs with empty shapes).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterCompressEmptyShapeCases(std::vector<TestCase> &registry,
                                                                   TestMode mode = TestMode::TEST);

/// Collects all empty-shape backend test cases by invoking every
/// ``Register*EmptyShapeCases`` helper declared in this header.
void CollectEmptyShapeTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
