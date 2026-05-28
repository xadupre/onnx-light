// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  const size_t start = registry.size();
  RegisterArgMaxCases(registry);
  RegisterArgMinCases(registry);
  RegisterReduceSumCases(registry);
  FilterTestCasesByOpType(registry, start, op_type);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
