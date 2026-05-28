// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectOptionalTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  const size_t start = registry.size();
  if (MatchOpTypeFilter(op_type, "Optional"))
    RegisterOptionalCases(registry);
  FilterTestCasesByOpType(registry, start, op_type);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
