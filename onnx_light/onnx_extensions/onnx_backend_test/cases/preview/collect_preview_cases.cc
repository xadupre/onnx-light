// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/onnx_backend_test/cases/preview/include_preview_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectPreviewTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                             TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"FlexAttention", &RegisterFlexAttentionCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
