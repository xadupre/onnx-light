// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Backend test cases that require the full ``RunModel`` runtime (rather
// than a direct single-kernel invocation) because the top-level graph
// invokes user-declared model-local ``FunctionProto`` entries — and, in
// particular, exercises a function that calls another function.
//
// The cases live in their own ``cases_runtime`` subtree to keep them
// distinct from the per-op kernel cases under ``cases/`` and the
// shape/numerical families under ``cases_for_shapes/`` /
// ``cases_numerical/``.
// ---------------------------------------------------------------------------

/// Registers a test case where a model-local function in domain
/// ``"outer"`` (``SquareThenAdd``) calls another model-local function in
/// domain ``"inner"`` (``Square``). Exercises cross-domain function-to-
/// function dispatch via ``RunModel``.
void RegisterFunctionCallsFunctionAcrossDomainsCase(std::vector<TestCase> &registry);

/// Registers a test case with three nested model-local functions in the
/// same domain (``Outer`` -> ``Middle`` -> ``Inner``). Exercises that the
/// function registry propagates through arbitrary nesting depth.
void RegisterFunctionThreeLevelNestedCallsCase(std::vector<TestCase> &registry);

/// Collects all model-local-function backend test cases by invoking every
/// ``Register*Case`` helper declared in this header. When ``op_type`` is
/// non-empty only the case whose top-level node ``op_type`` matches is
/// registered.
void CollectLocalFunctionTestCases(std::vector<TestCase> &registry,
                                   const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
