// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, LogicalCasesArePresent) {
  auto cases = CollectTestCases();
  bool has_and = false, has_and_b = false;
  bool has_or = false, has_or_b = false;
  bool has_xor = false, has_xor_b = false;
  for (const auto &c : cases) {
    if (c.name == "test_cc_and")
      has_and = true;
    if (c.name == "test_cc_and_bcast")
      has_and_b = true;
    if (c.name == "test_cc_or")
      has_or = true;
    if (c.name == "test_cc_or_bcast")
      has_or_b = true;
    if (c.name == "test_cc_xor")
      has_xor = true;
    if (c.name == "test_cc_xor_bcast")
      has_xor_b = true;
  }
  EXPECT_TRUE(has_and);
  EXPECT_TRUE(has_and_b);
  EXPECT_TRUE(has_or);
  EXPECT_TRUE(has_or_b);
  EXPECT_TRUE(has_xor);
  EXPECT_TRUE(has_xor_b);
}

TEST(BackendTestCase, AndCaseOutputsAreElementwiseAnd) {
  auto cases = CollectTestCases();
  const TestCase *tc = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_and") {
      tc = &c;
      break;
    }
  }
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  const uint8_t *x = ds.inputs[0].data.data();
  const uint8_t *y = ds.inputs[1].data.data();
  const uint8_t *z = ds.outputs[0].data.data();
  ASSERT_EQ(ds.outputs[0].element_count(), ds.inputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], (x[i] != 0 && y[i] != 0) ? 1 : 0);
  }
}

} // namespace Test
