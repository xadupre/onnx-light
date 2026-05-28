// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectLogicalTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static constexpr OpRegisterEntry kEntries[] = {
      {"And", &RegisterAndCases},         {"Or", &RegisterOrCases},     {"Xor", &RegisterXorCases},
      {"Greater", &RegisterGreaterCases}, {"Less", &RegisterLessCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
