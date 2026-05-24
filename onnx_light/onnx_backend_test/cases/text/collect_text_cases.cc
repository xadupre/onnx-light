// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTextTestCases(std::vector<TestCase> &registry) { RegisterStringConcatCases(registry); }

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
