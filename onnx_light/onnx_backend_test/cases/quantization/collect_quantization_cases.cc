// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectQuantizationTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"QuantizeLinear", &RegisterQuantizeLinearCases},
      {"DequantizeLinear", &RegisterDequantizeLinearCases},
      {"QLinearMatMul", &RegisterQLinearMatMulCases},
      {"QLinearConv", &RegisterQLinearConvCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
