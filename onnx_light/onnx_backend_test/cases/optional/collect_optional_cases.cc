// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void CollectOptionalTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"Optional", &RegisterOptionalCases},
      {"OptionalGetElement", &RegisterOptionalGetElementCases},
      {"OptionalHasElement", &RegisterOptionalHasElementCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
