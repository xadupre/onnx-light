// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

// The per-operator ``Register*`` helpers used to build these cases live in the
// library-private companion header below and are only needed while compiling
// ``lib_onnx_backend_test`` itself; external consumers only see ``Collect*``.
#ifdef ONNX_LIGHT_BACKEND_TEST_INTERNAL
#include "onnx_backend_test/cases/reduction/register_reduction_cases.h"
#endif
namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

/// Collects all ``reduction`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                               TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
