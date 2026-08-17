// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/backend_test/test_case.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::BuildSingleNodeCase;
using core::backend_test::CollectTestCases;
using core::backend_test::TestCase;

namespace Test {

TEST(BackendTestCaseShapeInference, GeneratedModelsUseInteroperableIrVersion) {
  NodeProto node;
  node.set_op_type("Constant");

  const auto built = BuildSingleNodeCase(node, {}, {}, "ir_version", {}, "backend-test");
  EXPECT_EQ(built.model.ir_version(), 13);
}

TEST(BackendTestCaseShapeInference, AllCollectedCasesPassChecker) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_NO_THROW(checker::check_model(tc.model(), /*full_check=*/false)) << "case: " << tc.name;
  }
}

} // namespace Test
