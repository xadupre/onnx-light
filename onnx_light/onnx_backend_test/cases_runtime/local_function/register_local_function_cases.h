// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <string>
#include <vector>

// Internal (library-private) header: the per-operator ``Register*`` backend
// test registration helpers. These are only ever invoked by the ``Collect*``
// aggregators inside ``lib_onnx_backend_test`` and are compiled with hidden
// visibility, so they are not exported from the shared library. This header is
// pulled in by the matching public header only when ONNX_LIGHT_BACKEND_TEST_INTERNAL is
// defined (i.e. while building the library itself).

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
/// ``CallModelLocalFunction`` resolves the references against the
/// call-site attributes before executing the function body.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterFunctionLinkedAttributeCase(std::vector<TestCase> &registry,
                                    TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
