// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/shape_tag/include_shape_tag_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectShapeTagTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                              TestMode mode) {
  if (op_type.empty() or op_type == "shape_tag") {
    RegisterShapeTagCases(registry);
    RegisterShapeTagAmbiguousCases(registry);
    RegisterShapeTagConstantMulConcatReshapeCases(registry);
    RegisterShapeTagOutputAsShapeCases(registry);
    RegisterShapeTagConcatWeightWinsCases(registry);
    RegisterShapeTagCastBackwardCases(registry);
    RegisterShapeTagReshapeBackwardCases(registry);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
