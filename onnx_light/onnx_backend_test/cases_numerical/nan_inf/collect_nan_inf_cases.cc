// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectNanInfTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (op_type.empty() or op_type == "nan_inf") {
    RegisterAddNanInfCases(registry);
    RegisterDivNanInfCases(registry);
    RegisterMulNanInfCases(registry);
    RegisterSubNanInfCases(registry);
    RegisterWhereNanInfCases(registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
