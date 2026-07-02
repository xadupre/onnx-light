// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/shape_tag/include_shape_tag_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectShapeTagTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (op_type.empty() or op_type == "shape_tag") {
    RegisterShapeTagCases(registry);
    RegisterShapeTagAmbiguousCases(registry);
    RegisterShapeTagConstantMulConcatReshapeCases(registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
