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
// Backend test cases that require the full model-run runtime (rather
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
/// function dispatch via a full model run (RegisterModelFunctions + RuntimeSession).
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterFunctionCallsFunctionAcrossDomainsCase(std::vector<TestCase> &registry,
                                               TestMode mode = TestMode::TEST);

/// Registers a test case with three nested model-local functions in the
/// same domain (``Outer`` -> ``Middle`` -> ``Inner``). Exercises that the
/// function registry propagates through arbitrary nesting depth.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterFunctionThreeLevelNestedCallsCase(std::vector<TestCase> &registry,
                                          TestMode mode = TestMode::TEST);

/// Registers a test case where a model-local function declares formal
/// attributes (``then_branch``/``else_branch``) that the body's ``If``
/// node references via ``ref_attr_name``. Exercises that
/// ``ModelLocalFunctionKernel`` resolves the references against the
/// call-site attributes before executing the function body.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterFunctionLinkedAttributeCase(std::vector<TestCase> &registry,
                                    TestMode mode = TestMode::TEST);

/// Collects all model-local-function backend test cases by invoking every
/// ``Register*Case`` helper declared in this header. When ``op_type`` is
/// non-empty only the case whose top-level node ``op_type`` matches is
/// registered.
void CollectLocalFunctionTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                                   TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
