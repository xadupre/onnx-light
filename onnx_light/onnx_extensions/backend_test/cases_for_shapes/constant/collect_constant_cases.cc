// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/constant/include_constant_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectConstantTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                              TestMode mode) {
  if (op_type.empty() or op_type == "constant") {
    RegisterConstantInfoCases(registry, mode);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
