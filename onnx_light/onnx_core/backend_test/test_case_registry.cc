// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case_registry.h"

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

namespace {

std::vector<TestCasesCollectorFn> &GetRegisteredCollectorsMutable() {
  static std::vector<TestCasesCollectorFn> registry;
  return registry;
}

} // namespace

int RegisterTestCasesCollector(TestCasesCollectorFn fn) {
  GetRegisteredCollectorsMutable().push_back(std::move(fn));
  return 0;
}

const std::vector<TestCasesCollectorFn> &GetRegisteredCollectors() {
  return GetRegisteredCollectorsMutable();
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
