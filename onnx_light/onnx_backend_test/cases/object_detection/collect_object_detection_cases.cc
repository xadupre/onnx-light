// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void CollectObjectDetectionTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"RoiAlign", &RegisterRoiAlignCases},
      {"NonMaxSuppression", &RegisterNonMaxSuppressionCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
