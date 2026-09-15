// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

class TestCaseUnloadGuard {
public:
  explicit TestCaseUnloadGuard(TestCase &test_case, bool enabled = true)
      : test_case_(test_case), enabled_(enabled) {}

  ~TestCaseUnloadGuard() {
    if (enabled_ && test_case_.materialized()) {
      test_case_.unload();
    }
  }

  TestCaseUnloadGuard(const TestCaseUnloadGuard &) = delete;
  TestCaseUnloadGuard &operator=(const TestCaseUnloadGuard &) = delete;

private:
  TestCase &test_case_;
  bool enabled_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
