// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/controlflow/include_controlflow_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectControlflowTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                 TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"If", &RegisterIfCases},
      {"Loop", &RegisterLoopCases},
      {"Scan", &RegisterScanCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
