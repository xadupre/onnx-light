// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases_for_shapes/peak_memory/include_peak_memory_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectPeakMemoryTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                                TestMode mode) {
  if (op_type.empty() or op_type == "peak_memory") {
    RegisterPeakMemoryCases(registry, mode);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
