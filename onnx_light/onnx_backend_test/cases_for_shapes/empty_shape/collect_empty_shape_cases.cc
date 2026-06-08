// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectEmptyShapeTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (op_type.empty() or op_type == "empty_shape") {
    RegisterAddEmptyShapeCases(registry);
    RegisterSubEmptyShapeCases(registry);
    RegisterMulEmptyShapeCases(registry);
    RegisterDivEmptyShapeCases(registry);
    RegisterPReluEmptyShapeCases(registry);
    RegisterCompressEmptyShapeCases(registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
