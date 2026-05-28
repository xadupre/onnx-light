// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTensorTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  const size_t start_ = registry.size();
  RegisterConcatCases(registry);
  RegisterCastCases(registry);
  RegisterAffineGridCases(registry);
  FilterTestCasesByOpType(registry, start_, op_type);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
