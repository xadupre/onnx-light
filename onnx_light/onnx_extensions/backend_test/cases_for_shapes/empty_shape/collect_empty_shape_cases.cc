// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectEmptyShapeTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                TestMode mode) {
  if (op_type.empty() or op_type == "empty_shape") {
    RegisterAddEmptyShapeCases(registry);
    RegisterCompressEmptyShapeCases(registry);
    RegisterDivEmptyShapeCases(registry);
    RegisterMulEmptyShapeCases(registry);
    RegisterPReluEmptyShapeCases(registry);
    RegisterSubEmptyShapeCases(registry);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
