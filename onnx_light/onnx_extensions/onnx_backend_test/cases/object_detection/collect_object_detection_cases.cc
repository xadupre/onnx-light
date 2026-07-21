// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/object_detection/include_object_detection_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectObjectDetectionTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                     TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"RoiAlign", &RegisterRoiAlignCases},
      {"NonMaxSuppression", &RegisterNonMaxSuppressionCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
