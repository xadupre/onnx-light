// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// The shape-inference cases are multi-node graphs whose ``op_type`` cannot
// be reduced to a single representative operator. We therefore unconditionally
// register every case; the ``op_type`` filter is applied by matching the
// graph's *first* operator only, which is sufficient for the existing
// node-filtering use cases.
void CollectShapeInferenceTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                    bool include_big, TestMode mode) {
  if (op_type.empty() or op_type == "shape" or op_type == "inference") {
    RegisterAddConcatReshapeShapeInferenceCases(registry);
    RegisterLocalFunctionAddShapeInferenceCases(registry);
    RegisterLocalFunctionRangeShapeInferenceCases(registry);
    RegisterNestedLocalFunctionAddShapeInferenceCases(registry);
    RegisterNonZeroChainNamedShapeInferenceCases(registry);
    RegisterShapeIdentityUnsqueezeShapeInferenceCases(registry);
    RegisterDimensionExpressionShapeInferenceCase(registry);
    RegisterFloorDivOffsetShapeInferenceCase(registry);
    RegisterSliceSymbolicEndShapeInferenceCases(registry);
    RegisterValueAsShapeShapeInferenceCases(registry);
    RegisterValueAsShapeBuilderShapeInferenceCases(registry);
    RegisterGatherValueAsShapeShapeInferenceCases(registry);
    RegisterCheckShapeShapeInferenceCases(registry);
    RegisterReshapeReshapeShapeInferenceCases(registry);
    RegisterConcatSplitShapeInferenceCases(registry, true);
    RegisterConcatSplitShapeInferenceCases(registry, false);
    RegisterIfSymbolicShapesShapeInferenceCases(registry);
    RegisterLoopPairwiseDistanceShapeInferenceCases(registry);
    RegisterTopKPairwiseDistanceShapeInferenceCases(registry);
    RegisterLoopTopKPairwiseDistanceShapeInferenceCases(registry);
    RegisterScanTopKPairwiseDistanceShapeInferenceCases(registry);
    RegisterTwoTopKSameKShapeInferenceCases(registry);
    RegisterTwoTopKDifferentKShapeInferenceCases(registry);
    RegisterScanRunningSumShapeInferenceCases(registry);
    RegisterResizeTileShapeInferenceCases(registry);
    RegisterPadCannyAverageShapeInferenceCases(registry);
    RegisterTinyLlmShapeInferenceCases(registry);
    RegisterTinyLlmInlinedShapeInferenceCases(registry);
    RegisterUnsqueezeVasReshapeShapeInferenceCases(registry);
    if (include_big) {
      RegisterQwen3_4LayersLikeShapeInferenceCases(registry);
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
