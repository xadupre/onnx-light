// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/test_case.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::CollectTestCases;
using onnx_kernels::TestCase;

namespace Test {

TEST(BackendTestCaseShapeInference, AllCollectedCasesPassChecker) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_NO_THROW(checker::check_model(tc.model, /*full_check=*/false)) << "case: " << tc.name;
  }
}

} // namespace Test
