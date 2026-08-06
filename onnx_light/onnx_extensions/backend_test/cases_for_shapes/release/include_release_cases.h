// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Backend test cases that exercise the release-after metadata written by
// ``ComputeContext::ComputeInPlaceReuseGraph`` when an intermediate tensor
// reaches its last consumer.  These live in their own ``cases_for_shapes``
// subtree so callers can collect them independently from the broader
// shape-inference gallery.
// ---------------------------------------------------------------------------

/// Registers all release backend test cases:
///
///  - ``test_cc_release_shape_reshape``: ``Shape → Reshape`` case whose
///    intermediate tensor ``S`` is released at the ``Reshape`` node (its only
///    consumer).
///  - ``test_cc_release_initializer_add``: ``Add(input, initializer) → Relu``
///    case that verifies both a graph input and a graph initializer appear
///    under ``onnx_light.not_used_after`` at the node where they reach their
///    last use.
///
/// The expected ``onnx_light.release_after`` and
/// ``onnx_light.not_used_after`` node metadata are pre-embedded in each model
/// so tests can verify that
/// ``ComputeContext::ComputeInPlaceReuseGraph`` reproduces them.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReleaseCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

/// Collects all release backend test cases by invoking every
/// ``Register*Release*Cases`` helper declared in this header.
void CollectReleaseTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                             TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
