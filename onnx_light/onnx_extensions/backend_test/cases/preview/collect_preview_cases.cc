// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/preview/include_preview_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectPreviewTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                             TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"FlexAttention", &RegisterFlexAttentionCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
