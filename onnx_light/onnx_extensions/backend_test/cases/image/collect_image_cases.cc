// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/image/include_image_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectImageTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                           TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"ImageDecoder", &RegisterImageDecoderCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
