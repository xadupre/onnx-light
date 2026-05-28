// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectLogicalTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  const size_t start = registry.size();
  if (MatchOpTypeFilter(op_type, "And"))
    RegisterAndCases(registry);
  if (MatchOpTypeFilter(op_type, "Or"))
    RegisterOrCases(registry);
  if (MatchOpTypeFilter(op_type, "Xor"))
    RegisterXorCases(registry);
  if (MatchOpTypeFilter(op_type, "Greater"))
    RegisterGreaterCases(registry);
  if (MatchOpTypeFilter(op_type, "Less"))
    RegisterLessCases(registry);
  FilterTestCasesByOpType(registry, start, op_type);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
