// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTensorTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"Concat", &RegisterConcatCases},
      {"Cast", &RegisterCastCases},
      {"CastLike", &RegisterCastLikeCases},
      {"BitCast", &RegisterBitCastCases},
      {"AffineGrid", &RegisterAffineGridCases},
      {"GridSample", &RegisterGridSampleCases},
      {"Expand", &RegisterExpandCases},
      {"Reshape", &RegisterReshapeCases},
      {"Slice", &RegisterSliceCases},
      {"Transpose", &RegisterTransposeCases},
      {"Trilu", &RegisterTriluCases},
      {"DepthToSpace", &RegisterDepthToSpaceCases},
      {"Tile", &RegisterTileCases},
      {"Squeeze", &RegisterSqueezeCases},
      {"Unsqueeze", &RegisterUnsqueezeCases},
      {"NonZero", &RegisterNonZeroCases},
      {"Shape", &RegisterShapeCases},
      {"Gather", &RegisterGatherCases},
      {"GatherElements", &RegisterGatherElementsCases},
      {"GatherND", &RegisterGatherNDCases},
      {"Compress", &RegisterCompressCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
