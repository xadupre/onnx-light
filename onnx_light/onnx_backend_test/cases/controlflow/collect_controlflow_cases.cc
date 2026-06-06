// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void CollectControlflowTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"If", &RegisterIfCases},
      {"Loop", &RegisterLoopCases},
      {"Scan", &RegisterScanCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
