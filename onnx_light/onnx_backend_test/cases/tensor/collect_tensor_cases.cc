// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTensorTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  const size_t start = registry.size();
  if (MatchOpTypeFilter(op_type, "Concat"))
    RegisterConcatCases(registry);
  if (MatchOpTypeFilter(op_type, "Cast"))
    RegisterCastCases(registry);
  if (MatchOpTypeFilter(op_type, "AffineGrid"))
    RegisterAffineGridCases(registry);
  FilterTestCasesByOpType(registry, start, op_type);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
