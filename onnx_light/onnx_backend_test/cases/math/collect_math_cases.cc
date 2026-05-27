// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectMathTestCases(std::vector<TestCase> &registry) {
  RegisterAbsCases(registry);
  RegisterAcosCases(registry);
  RegisterAcoshCases(registry);
  RegisterAsinCases(registry);
  RegisterAsinhCases(registry);
  RegisterAddCases(registry);
  RegisterSubCases(registry);
  RegisterMulCases(registry);
  RegisterDivCases(registry);
  RegisterBlackmanWindowCases(registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
