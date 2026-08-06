// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/release/include_release_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectReleaseTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                             TestMode mode) {
  if (op_type.empty() or op_type == "release") {
    RegisterReleaseCases(registry);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
