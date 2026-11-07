// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/backend_test/test_case.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "test_case_utils.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::BuildSingleNodeCase;
using core::backend_test::CollectTestCases;
using core::backend_test::TestCase;
using core::backend_test::TestCaseUnloadGuard;

namespace Test {

TEST(BackendTestCase, AllCollectedCaseNamesAreUnique) {
  const auto cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());
  std::unordered_set<std::string> names;
  for (const auto &tc : cases) {
    EXPECT_TRUE(names.insert(tc.name).second) << "Duplicate backend test name: " << tc.name;
  }
}

TEST(BackendTestCase, GroupedMatMulTypedCaseNames) {
  const auto cases = CollectTestCases("GroupedMatMul");
  std::unordered_set<std::string> names;
  for (const auto &tc : cases) {
    if (tc.name.find("test_cc_groupedmatmul_float") == 0) {
      EXPECT_TRUE(names.insert(tc.name).second) << tc.name;
    }
  }
  const std::unordered_set<std::string> expected{
      "test_cc_groupedmatmul_float16_int32", "test_cc_groupedmatmul_float16_int64",
      "test_cc_groupedmatmul_float64_int32", "test_cc_groupedmatmul_float64_int64"};
  EXPECT_EQ(names, expected);
}

TEST(BackendTestCaseShapeInference, GeneratedModelsUseInteroperableIrVersion) {
  NodeProto node;
  node.set_op_type("Constant");

  const auto built = BuildSingleNodeCase(node, {}, {}, "ir_version", {}, "backend-test");
  EXPECT_EQ(built.model.ir_version(), 13);
}

TEST(BackendTestCaseShapeInference, GeneratedModelsUseIrVersionForOpset) {
  NodeProto node;
  node.set_op_type("Constant");

  for (int64_t version : {27, 28}) {
    for (const std::string domain : {"", "custom"}) {
      SCOPED_TRACE(domain);
      SCOPED_TRACE(version);
      const auto built = BuildSingleNodeCase(
          node, {}, {}, "ir_version", {core::runtime::OpsetId(domain, version)}, "backend-test");
      EXPECT_EQ(built.model.ir_version(), domain.empty() && version == 28 ? 14 : 13);
    }
  }
}

TEST(BackendTestCaseShapeInference, AllCollectedCasesPassChecker) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (TestCase &tc : cases) {
    TestCaseUnloadGuard unload_guard(tc);
    SCOPED_TRACE(tc.name);
    ASSERT_NO_THROW(checker::check_model(tc.model(), /*full_check=*/false)) << "case: " << tc.name;
  }
}

} // namespace Test
