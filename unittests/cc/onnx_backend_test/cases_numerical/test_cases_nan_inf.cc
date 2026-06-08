// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectNanInfTestCases;
using onnx_backend_test::TestCase;

namespace {

std::vector<TestCase> Collect(const std::string &op_type = "") {
  std::vector<TestCase> registry;
  CollectNanInfTestCases(registry, op_type);
  return registry;
}

const TestCase *Find(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name) {
      return &c;
    }
  }
  return nullptr;
}

bool HasNaN(const float *values, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    if (std::isnan(values[i])) {
      return true;
    }
  }
  return false;
}

bool HasInf(const float *values, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    if (std::isinf(values[i])) {
      return true;
    }
  }
  return false;
}

} // namespace

namespace Test {

TEST(NanInfCases, CollectorReturnsCases) {
  const auto cases = Collect();
  EXPECT_FALSE(cases.empty());
}

TEST(NanInfCases, AddInputsCarryNanAndInf) {
  const auto cases = Collect("nan_inf");
  const TestCase *tc = Find(cases, "test_cc_add_nan_inf");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  const auto *x = ds.inputs[0].AsFloat();
  const auto *y = ds.inputs[1].AsFloat();
  const int64_t n = ds.inputs[0].element_count();
  EXPECT_TRUE(HasNaN(x, n));
  EXPECT_TRUE(HasInf(x, n));
  EXPECT_TRUE(HasNaN(y, n));
  EXPECT_TRUE(HasInf(y, n));
}

TEST(NanInfCases, AddOutputPropagatesNanAndInf) {
  // Reference values follow IEEE-754:
  //   x = [ NaN, +Inf, -Inf, +Inf,  1.0,  0.0]
  //   y = [ 1.0,  2.0, -2.0, -Inf,  NaN, +Inf]
  //   z = [ NaN, +Inf, -Inf,  NaN,  NaN, +Inf]
  const auto cases = Collect("nan_inf");
  const TestCase *tc = Find(cases, "test_cc_add_nan_inf");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *z = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.outputs[0].element_count(), 6);
  EXPECT_TRUE(std::isnan(z[0]));
  EXPECT_TRUE(std::isinf(z[1]));
  EXPECT_GT(z[1], 0.0f);
  EXPECT_TRUE(std::isinf(z[2]));
  EXPECT_LT(z[2], 0.0f);
  EXPECT_TRUE(std::isnan(z[3])); // +Inf + -Inf = NaN
  EXPECT_TRUE(std::isnan(z[4]));
  EXPECT_TRUE(std::isinf(z[5]));
  EXPECT_GT(z[5], 0.0f);
}

} // namespace Test
