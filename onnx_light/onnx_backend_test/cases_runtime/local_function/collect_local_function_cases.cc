// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_runtime/local_function/include_local_function_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectLocalFunctionTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"runtime", &RegisterFunctionCallsFunctionAcrossDomainsCase},
      {"runtime", &RegisterFunctionThreeLevelNestedCallsCase},
      {"runtime", &RegisterFunctionLinkedAttributeCase},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
