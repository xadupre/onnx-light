// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/optional/include_optional_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectOptionalTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                              TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"Optional", &RegisterOptionalCases},
      {"OptionalGetElement", &RegisterOptionalGetElementCases},
      {"OptionalHasElement", &RegisterOptionalHasElementCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
