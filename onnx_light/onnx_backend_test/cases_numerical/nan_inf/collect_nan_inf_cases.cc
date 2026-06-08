// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectNanInfTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"numerical", &RegisterAddNanInfCases},   {"numerical", &RegisterSubNanInfCases},
      {"numerical", &RegisterMulNanInfCases},   {"numerical", &RegisterDivNanInfCases},
      {"numerical", &RegisterWhereNanInfCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
