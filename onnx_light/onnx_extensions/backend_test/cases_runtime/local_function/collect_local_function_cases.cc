// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_runtime/local_function/include_local_function_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectLocalFunctionTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                   TestMode /*mode*/) {
  if (op_type.empty() or op_type == "local_function") {
    RegisterFunctionCallsFunctionAcrossDomainsCase(registry);
    RegisterFunctionLinkedAttributeCase(registry);
    RegisterFunctionThreeLevelNestedCallsCase(registry);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
