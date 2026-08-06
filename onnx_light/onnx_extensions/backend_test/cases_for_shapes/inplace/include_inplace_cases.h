// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Backend test cases that exercise the in-place-reuse analysis metadata stored
// in ``NodeProto::metadata_props``. These live in their own
// ``cases_for_shapes`` subtree so callers can collect them independently from
// the broader shape-inference gallery.
// ---------------------------------------------------------------------------

/// Registers an ``Abs → Abs → Abs`` case whose intermediate tensors all share
/// the same shape so in-place-reuse inference can detect the recyclable
/// buffers and record the expected metadata on the graph nodes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterInPlaceReuseCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Collects all in-place-reuse backend test cases by invoking every
/// ``Register*InPlace*Cases`` helper declared in this header.
void CollectInPlaceTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                             TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
