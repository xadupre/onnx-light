// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  if (MatchOpTypeFilter(op_type, "ArgMax"))
    RegisterArgMaxCases(registry);
  if (MatchOpTypeFilter(op_type, "ArgMin"))
    RegisterArgMinCases(registry);
  if (MatchOpTypeFilter(op_type, "ReduceSum"))
    RegisterReduceSumCases(registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
