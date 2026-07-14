// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectQuantizationTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                  TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"QuantizeLinear", &RegisterQuantizeLinearCases},
      {"DequantizeLinear", &RegisterDequantizeLinearCases},
      {"DynamicQuantizeLinear", &RegisterDynamicQuantizeLinearCases},
      {"QLinearMatMul", &RegisterQLinearMatMulCases},
      {"QLinearConv", &RegisterQLinearConvCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
