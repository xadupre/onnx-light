// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectQuantizationTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  DispatchRegisterByOpType(registry, op_type,
                           {
                               {"QuantizeLinear", &RegisterQuantizeLinearCases},
                           });
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
