// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``optional`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Optional`` backend test node case(s).
void RegisterOptionalCases(std::vector<TestCase> &registry);

/// Registers the ``OptionalGetElement`` backend test node case(s).
void RegisterOptionalGetElementCases(std::vector<TestCase> &registry);

/// Registers the ``OptionalHasElement`` backend test node case(s).
void RegisterOptionalHasElementCases(std::vector<TestCase> &registry);

/// Collects all ``optional`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectOptionalTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
