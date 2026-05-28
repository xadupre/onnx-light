// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTraditionalMLTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  const size_t start = registry.size();
  if (MatchOpTypeFilter(op_type, "Binarizer"))
    RegisterBinarizerCases(registry);
  if (MatchOpTypeFilter(op_type, "LabelEncoder"))
    RegisterLabelEncoderCases(registry);
  FilterTestCasesByOpType(registry, start, op_type);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
