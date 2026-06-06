// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void CollectQuantizationTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"QuantizeLinear", &RegisterQuantizeLinearCases},
      {"DequantizeLinear", &RegisterDequantizeLinearCases},
      {"DynamicQuantizeLinear", &RegisterDynamicQuantizeLinearCases},
      {"QLinearMatMul", &RegisterQLinearMatMulCases},
      {"QLinearConv", &RegisterQLinearConvCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
