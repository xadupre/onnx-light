// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

// ---------------------------------------------------------------------------
// Backend test cases that exercise the constant-information analysis metadata
// stored under ``onnx_light.constant`` in ``NodeProto`` and ``ValueInfoProto``
// (and initializer ``TensorProto``) ``metadata_props``. A value is *constant*
// when its content is known before inference starts (initializers, ``Constant``
// outputs, or outputs of deterministic nodes whose inputs are all constant).
// ---------------------------------------------------------------------------

/// Registers the constant-information backend test cases (a purely
/// initializer-derived chain and a ``Constant``-node source), each carrying the
/// expected ``onnx_light.constant`` metadata pre-embedded on the graph.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterConstantInfoCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Collects all constant-information backend test cases by invoking every
/// ``Register*Constant*Cases`` helper declared in this header.
void CollectConstantTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                              TestMode mode = TestMode::TEST);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
