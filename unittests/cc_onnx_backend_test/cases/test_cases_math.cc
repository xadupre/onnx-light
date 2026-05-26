// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, AcosCaseOutputsMatchStdAcos) {
  auto cases = CollectTestCases();
  const TestCase *acos = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_acos") {
      acos = &c;
      break;
    }
  }
  ASSERT_NE(acos, nullptr);
  ASSERT_EQ(acos->data_sets.size(), 1u);
  const auto &ds = acos->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::acos(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AcoshCaseOutputsMatchStdAcosh) {
  auto cases = CollectTestCases();
  const TestCase *acosh = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_acosh") {
      acosh = &c;
      break;
    }
  }
  ASSERT_NE(acosh, nullptr);
  ASSERT_EQ(acosh->data_sets.size(), 1u);
  const auto &ds = acosh->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::acosh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AddCaseOutputsAreElementwiseSum) {
  auto cases = CollectTestCases();
  const TestCase *add = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_add") {
      add = &c;
      break;
    }
  }
  ASSERT_NE(add, nullptr);
  ASSERT_EQ(add->data_sets.size(), 1u);
  const auto &ds = add->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.inputs[1].AsFloat();
  const float *z = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(z[i], x[i] + y[i]);
  }
}

namespace {

const TestCase *FindCase(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name)
      return &c;
  }
  return nullptr;
}

} // namespace

TEST(BackendTestCase, SubCaseOutputsAreElementwiseDifference) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_sub");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.inputs[1].AsFloat();
  const float *z = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(z[i], x[i] - y[i]);
  }
}

TEST(BackendTestCase, MulCaseOutputsAreElementwiseProduct) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_mul");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.inputs[1].AsFloat();
  const float *z = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(z[i], x[i] * y[i]);
  }
}

TEST(BackendTestCase, DivCaseOutputsAreElementwiseQuotient) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_cc_div");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.inputs[1].AsFloat();
  const float *z = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(z[i], x[i] / y[i]);
  }
}

TEST(BackendTestCase, SubMulDivOnnxCasesArePresent) {
  // Mirrors the upstream-ONNX-mirrored cases exported by RegisterSubCases,
  // RegisterMulCases and RegisterDivCases for the float-32 variants.
  const std::vector<std::string> expected_names = {
      "test_sub_example", "test_sub",         "test_sub_bcast", "test_mul_example", "test_mul",
      "test_mul_bcast",   "test_div_example", "test_div",       "test_div_bcast",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing upstream ONNX case: " << name;
  }
}

TEST(BackendTestCase, SubExampleCaseHasExpectedValues) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_sub_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].element_count(), 3);
  const float *z = ds.outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(z[0], -2.0f);
  EXPECT_FLOAT_EQ(z[1], 0.0f);
  EXPECT_FLOAT_EQ(z[2], 2.0f);
}

TEST(BackendTestCase, MulExampleCaseHasExpectedValues) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_mul_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].element_count(), 3);
  const float *z = ds.outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(z[0], 4.0f);
  EXPECT_FLOAT_EQ(z[1], 10.0f);
  EXPECT_FLOAT_EQ(z[2], 18.0f);
}

TEST(BackendTestCase, DivExampleCaseHasExpectedValues) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_div_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].element_count(), 2);
  const float *z = ds.outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(z[0], 3.0f);
  EXPECT_FLOAT_EQ(z[1], 2.0f);
}

TEST(BackendTestCase, SubMulDivBroadcastCasesHaveBroadcastShapes) {
  auto cases = CollectTestCases();
  for (const char *name : {"test_sub_bcast", "test_mul_bcast", "test_div_bcast"}) {
    const TestCase *tc = FindCase(cases, name);
    ASSERT_NE(tc, nullptr) << name;
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << name;
    EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{5})) << name;
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << name;
  }
}

TEST(BackendTestCase, BlackmanWindowCasesArePresent) {
  auto cases = CollectTestCases();
  const TestCase *periodic = nullptr;
  const TestCase *symmetric = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_blackmanwindow")
      periodic = &c;
    if (c.name == "test_cc_blackmanwindow_symmetric")
      symmetric = &c;
  }
  ASSERT_NE(periodic, nullptr);
  ASSERT_NE(symmetric, nullptr);

  // Both variants take a scalar int32 size and produce a 1-D float window.
  for (const TestCase *tc : {periodic, symmetric}) {
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 10);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    ASSERT_EQ(tc->model.ref_opset_import().size(), 1u);
    EXPECT_EQ(tc->model.ref_opset_import()[0].version(), 17);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const auto &op_type = graph.ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "BlackmanWindow");
  }

  // The symmetric variant carries a ``periodic = 0`` attribute; the periodic
  // variant relies on the default and has no attribute.
  ASSERT_EQ(symmetric->model.ref_graph().ref_node()[0].ref_attribute().size(), 1u);
  const auto &attr = symmetric->model.ref_graph().ref_node()[0].ref_attribute()[0];
  EXPECT_EQ(std::string(attr.ref_name().data(), attr.ref_name().size()), "periodic");
  EXPECT_EQ(attr.type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attr.i(), 0);
  EXPECT_EQ(periodic->model.ref_graph().ref_node()[0].ref_attribute().size(), 0u);

  // Output values match the Blackman window reference formula.
  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 0.42;
  constexpr double a1 = -0.5;
  constexpr double a2 = 0.08;
  const int32_t size = 10;
  for (const TestCase *tc : {periodic, symmetric}) {
    const double divisor =
        (tc == periodic) ? static_cast<double>(size) : static_cast<double>(size - 1);
    const float *y = tc->data_sets[0].outputs[0].AsFloat();
    for (int32_t n = 0; n < size; ++n) {
      const double k = static_cast<double>(n) / divisor;
      const float expected =
          static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k) + a2 * std::cos(4.0 * kPi * k));
      EXPECT_FLOAT_EQ(y[n], expected);
    }
  }
}

} // namespace Test
