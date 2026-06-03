// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Backend test cases that exercise operators on tensors whose shape is
// "empty" — either rank-0 (the shape vector itself is empty, ``{}``) or a
// shape with a zero-sized dimension (e.g. ``{0}`` or ``{0, 3}``) yielding a
// tensor with no elements. These cases live in their own ``cases_for_shapes``
// subtree so they are easy to discover and extend with other shape-oriented
// scenarios later.
// ---------------------------------------------------------------------------

/// Registers backend test cases that add tensors with empty shapes.
void RegisterAddEmptyShapeCases(std::vector<TestCase> &registry);

/// Registers backend test cases that run the ``Compress`` node on tensors
/// with empty shapes (and/or producing outputs with empty shapes).
void RegisterCompressEmptyShapeCases(std::vector<TestCase> &registry);

/// Collects all empty-shape backend test cases by invoking every
/// ``Register*EmptyShapeCases`` helper declared in this header.
void CollectEmptyShapeTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
