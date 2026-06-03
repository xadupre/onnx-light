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
using onnx_backend_test::CollectMathTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectMathTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, AcosCaseOutputsMatchStdAcos) {
  auto cases = CollectTestCases("Acos");
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
  auto cases = CollectTestCases("Acosh");
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
  auto cases = CollectTestCases("Add");
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
  auto cases = CollectTestCases("Sub");
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
  auto cases = CollectTestCases("Mul");
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
  auto cases = CollectTestCases("Div");
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

TEST(BackendTestCase, SigmoidCaseOutputsMatchLogisticFunction) {
  auto cases = CollectTestCases("Sigmoid");
  const TestCase *tc = FindCase(cases, "test_cc_sigmoid");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], 1.0f / (1.0f + std::exp(-x[i])), 1e-6f);
  }
}

TEST(BackendTestCase, SoftplusCaseOutputsMatchSoftplusFunction) {
  auto cases = CollectTestCases("Softplus");
  const TestCase *tc = FindCase(cases, "test_cc_softplus");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::log1p(std::exp(x[i])), 1e-6f);
  }
}

TEST(BackendTestCase, SoftsignCaseOutputsMatchSoftsignFunction) {
  auto cases = CollectTestCases("Softsign");
  const TestCase *tc = FindCase(cases, "test_cc_softsign");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], x[i] / (1.0f + std::fabs(x[i])), 1e-6f);
  }
}

TEST(BackendTestCase, ExpCaseOutputsMatchStdExp) {
  auto cases = CollectTestCases("Exp");
  const TestCase *tc = FindCase(cases, "test_cc_exp");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::exp(x[i]), 1e-6f);
  }
}

TEST(BackendTestCase, ErfCaseOutputsMatchStdErf) {
  auto cases = CollectTestCases("Erf");
  const TestCase *tc = FindCase(cases, "test_cc_erf");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::erf(x[i]), 1e-6f);
  }
}

TEST(BackendTestCase, LogCaseOutputsMatchStdLog) {
  auto cases = CollectTestCases("Log");
  const TestCase *tc = FindCase(cases, "test_cc_log");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::log(x[i]), 1e-6f);
  }
}

TEST(BackendTestCase, SoftmaxCaseOutputsAreNormalizedAlongAxis) {
  auto cases = CollectTestCases("Softmax");
  const TestCase *tc = FindCase(cases, "test_cc_softmax");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.outputs[0].shape.size(), 2u);
  const int64_t rows = ds.outputs[0].shape[0];
  const int64_t cols = ds.outputs[0].shape[1];
  for (int64_t r = 0; r < rows; ++r) {
    float sum = 0.0f;
    for (int64_t c = 0; c < cols; ++c) {
      sum += y[static_cast<size_t>(r * cols + c)];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
  }
}

TEST(BackendTestCase, AcosAcoshAsinAsinhOnnxCasesArePresent) {
  // Mirrors the upstream-ONNX-mirrored cases exported by RegisterAcosCases,
  // RegisterAcoshCases, RegisterAsinCases and RegisterAsinhCases.
  const std::vector<std::string> expected_names = {
      "test_acos_example", "test_acos", "test_acosh_example", "test_acosh",
      "test_asin_example", "test_asin", "test_asinh_example", "test_asinh",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing upstream ONNX case: " << name;
  }
}

TEST(BackendTestCase, AcosExampleCaseMatchesStdAcos) {
  auto cases = CollectTestCases("Acos");
  const TestCase *tc = FindCase(cases, "test_acos_example");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::acos(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AcoshExampleCaseMatchesStdAcosh) {
  auto cases = CollectTestCases("Acosh");
  const TestCase *tc = FindCase(cases, "test_acosh_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::acosh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AsinExampleCaseMatchesStdAsin) {
  auto cases = CollectTestCases("Asin");
  const TestCase *tc = FindCase(cases, "test_asin_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::asin(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AsinhExampleCaseMatchesStdAsinh) {
  auto cases = CollectTestCases("Asinh");
  const TestCase *tc = FindCase(cases, "test_asinh_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::asinh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AcosRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Acos");
  const TestCase *tc = FindCase(cases, "test_acos");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::acos(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AcoshRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Acosh");
  const TestCase *tc = FindCase(cases, "test_acosh");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::acosh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AsinRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Asin");
  const TestCase *tc = FindCase(cases, "test_asin");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::asin(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AsinhRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Asinh");
  const TestCase *tc = FindCase(cases, "test_asinh");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::asinh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AtanAtanhOnnxCasesArePresent) {
  // Mirrors the upstream-ONNX-mirrored cases exported by RegisterAtanCases and
  // RegisterAtanhCases.
  const std::vector<std::string> expected_names = {
      "test_atan_example",
      "test_atan",
      "test_atanh_example",
      "test_atanh",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing upstream ONNX case: " << name;
  }
}

TEST(BackendTestCase, AtanExampleCaseMatchesStdAtan) {
  auto cases = CollectTestCases("Atan");
  const TestCase *tc = FindCase(cases, "test_atan_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::atan(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AtanhExampleCaseMatchesStdAtanh) {
  auto cases = CollectTestCases("Atanh");
  const TestCase *tc = FindCase(cases, "test_atanh_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::atanh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AtanRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Atan");
  const TestCase *tc = FindCase(cases, "test_atan");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::atan(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, AtanhRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Atanh");
  const TestCase *tc = FindCase(cases, "test_atanh");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::atanh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, CosCoshOnnxCasesArePresent) {
  // Mirrors the upstream-ONNX-mirrored cases exported by RegisterCosCases and
  // RegisterCoshCases.
  const std::vector<std::string> expected_names = {
      "test_cos_example",
      "test_cos",
      "test_cosh_example",
      "test_cosh",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing upstream ONNX case: " << name;
  }
}

TEST(BackendTestCase, CosCaseOutputsMatchStdCos) {
  auto cases = CollectTestCases("Cos");
  const TestCase *tc = FindCase(cases, "test_cc_cos");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::cos(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, CoshCaseOutputsMatchStdCosh) {
  auto cases = CollectTestCases("Cosh");
  const TestCase *tc = FindCase(cases, "test_cc_cosh");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::cosh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, CosExampleCaseMatchesStdCos) {
  auto cases = CollectTestCases("Cos");
  const TestCase *tc = FindCase(cases, "test_cos_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::cos(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, CoshExampleCaseMatchesStdCosh) {
  auto cases = CollectTestCases("Cosh");
  const TestCase *tc = FindCase(cases, "test_cosh_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs[0].element_count(), 3);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::cosh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, CosRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Cos");
  const TestCase *tc = FindCase(cases, "test_cos");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::cos(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, CoshRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Cosh");
  const TestCase *tc = FindCase(cases, "test_cosh");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::cosh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, SinSinhOnnxCasesArePresent) {
  const std::vector<std::string> expected_names = {
      "test_sin_example",
      "test_sin",
      "test_sinh_example",
      "test_sinh",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing ONNX/math case: " << name;
  }
}

TEST(BackendTestCase, SinCaseOutputsMatchStdSin) {
  auto cases = CollectTestCases("Sin");
  const TestCase *tc = FindCase(cases, "test_cc_sin");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::sin(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, SinhCaseOutputsMatchStdSinh) {
  auto cases = CollectTestCases("Sinh");
  const TestCase *tc = FindCase(cases, "test_cc_sinh");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::sinh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, SinRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Sin");
  const TestCase *tc = FindCase(cases, "test_sin");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::sin(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, SinhRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Sinh");
  const TestCase *tc = FindCase(cases, "test_sinh");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::sinh(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, TanCaseOutputsMatchStdTan) {
  auto cases = CollectTestCases("Tan");
  const TestCase *tc = FindCase(cases, "test_cc_tan");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::tan(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, TanhCaseOutputsMatchStdTanh) {
  auto cases = CollectTestCases("Tanh");
  const TestCase *tc = FindCase(cases, "test_cc_tanh");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::tanh(x[i]), 1e-6f);
  }
}

TEST(BackendTestCase, TanRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Tan");
  const TestCase *tc = FindCase(cases, "test_tan");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::tan(x[i]), 1e-5f);
  }
}

TEST(BackendTestCase, TanhRandomCaseHasUpstreamShape) {
  auto cases = CollectTestCases("Tanh");
  const TestCase *tc = FindCase(cases, "test_tanh");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  const std::vector<int64_t> expected_shape = {3, 4, 5};
  EXPECT_EQ(ds.inputs[0].shape, expected_shape);
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_NEAR(y[i], std::tanh(x[i]), 1e-6f);
  }
}

TEST(BackendTestCase, AddSubMulDivOnnxCasesArePresent) {
  // Mirrors the upstream-ONNX-mirrored cases exported by RegisterAddCases,
  // RegisterSubCases, RegisterMulCases and RegisterDivCases. The Add, Mul and
  // Div entries cover the full set of numeric dtypes registered by their
  // respective kernels (FLOAT plus all supported signed/unsigned integer
  // variants from ``onnx.backend.test.case.node.{add,mul,div}``).
  const std::vector<std::string> expected_names = {
      "test_add",
      "test_add_int8",
      "test_add_int16",
      "test_add_uint8",
      "test_add_uint16",
      "test_add_uint32",
      "test_add_uint64",
      "test_add_bcast",
      "test_sub_example",
      "test_sub",
      "test_sub_bcast",
      "test_mul_example",
      "test_mul",
      "test_mul_int8",
      "test_mul_int16",
      "test_mul_uint8",
      "test_mul_uint16",
      "test_mul_uint32",
      "test_mul_uint64",
      "test_mul_bcast",
      "test_div_example",
      "test_div",
      "test_div_int8",
      "test_div_int16",
      "test_div_int32_trunc",
      "test_div_uint8",
      "test_div_uint16",
      "test_div_uint32",
      "test_div_uint64",
      "test_div_bcast",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing upstream ONNX case: " << name;
  }
}

TEST(BackendTestCase, MatMulCasesArePresent) {
  auto cases = CollectTestCases("MatMul");
  for (const char *name :
       {"test_cc_matmul_2d", "test_cc_matmul_vector_matrix", "test_cc_matmul_batch_broadcast"}) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing MatMul case: " << name;
  }
}

TEST(BackendTestCase, MatMulCaseShapesMatchExpectedSignatures) {
  auto cases = CollectTestCases("MatMul");

  const TestCase *two_d = FindCase(cases, "test_cc_matmul_2d");
  ASSERT_NE(two_d, nullptr);
  ASSERT_EQ(two_d->data_sets.size(), 1u);
  EXPECT_EQ(two_d->data_sets[0].inputs[0].shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(two_d->data_sets[0].inputs[1].shape, (std::vector<int64_t>{3, 4}));
  EXPECT_EQ(two_d->data_sets[0].outputs[0].shape, (std::vector<int64_t>{2, 4}));

  const TestCase *vector_matrix = FindCase(cases, "test_cc_matmul_vector_matrix");
  ASSERT_NE(vector_matrix, nullptr);
  ASSERT_EQ(vector_matrix->data_sets.size(), 1u);
  EXPECT_EQ(vector_matrix->data_sets[0].inputs[0].shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(vector_matrix->data_sets[0].inputs[1].shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(vector_matrix->data_sets[0].outputs[0].shape, (std::vector<int64_t>{2}));

  const TestCase *batch = FindCase(cases, "test_cc_matmul_batch_broadcast");
  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->data_sets.size(), 1u);
  EXPECT_EQ(batch->data_sets[0].inputs[0].shape, (std::vector<int64_t>{2, 2, 3}));
  EXPECT_EQ(batch->data_sets[0].inputs[1].shape, (std::vector<int64_t>{1, 3, 4}));
  EXPECT_EQ(batch->data_sets[0].outputs[0].shape, (std::vector<int64_t>{2, 2, 4}));
}

TEST(BackendTestCase, AbsUpstreamOnnxCaseMatchesReference) {
  // Mirrors the upstream ``onnx.backend.test.case.node.abs.Abs`` export:
  // a single rank-3 ``[3, 4, 5]`` float input whose elementwise absolute
  // value is the expected output.
  auto cases = CollectTestCases("Abs");
  const TestCase *tc = FindCase(cases, "test_abs");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(y[i], std::fabs(x[i]));
  }
}

TEST(BackendTestCase, SubExampleCaseHasExpectedValues) {
  auto cases = CollectTestCases("Sub");
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
  auto cases = CollectTestCases("Mul");
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
  auto cases = CollectTestCases("Div");
  const TestCase *tc = FindCase(cases, "test_div_example");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs[0].element_count(), 2);
  const float *z = ds.outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(z[0], 3.0f);
  EXPECT_FLOAT_EQ(z[1], 2.0f);
}

TEST(BackendTestCase, DivInt32TruncCaseHasExpectedValues) {
  // Mirrors the upstream ``test_div_int32_trunc`` case: ``[-3, 3, -3, 3] /
  // [2, 2, -2, -2] == [-1, 1, 1, -1]`` under C/C++ truncating signed integer
  // division.
  auto cases = CollectTestCases("Div");
  const TestCase *tc = FindCase(cases, "test_div_int32_trunc");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  ASSERT_EQ(ds.outputs[0].element_count(), 4);
  const int32_t *z = ds.outputs[0].AsInt32();
  EXPECT_EQ(z[0], -1);
  EXPECT_EQ(z[1], 1);
  EXPECT_EQ(z[2], 1);
  EXPECT_EQ(z[3], -1);
}

TEST(BackendTestCase, AddSubMulDivBroadcastCasesHaveBroadcastShapes) {
  auto cases = CollectTestCases();
  for (const char *name :
       {"test_add_bcast", "test_sub_bcast", "test_mul_bcast", "test_div_bcast"}) {
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
  auto cases = CollectTestCases("BlackmanWindow");
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
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 10);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
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

TEST(BackendTestCase, FloorCeilRoundOnnxCasesArePresent) {
  const std::vector<std::string> expected_names = {
      "test_floor_example", "test_floor", "test_ceil_example", "test_ceil", "test_round",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing ONNX/math case: " << name;
  }
}

TEST(BackendTestCase, FloorCaseOutputsMatchStdFloor) {
  auto cases = CollectTestCases("Floor");
  const TestCase *tc = FindCase(cases, "test_cc_floor");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(y[i], std::floor(x[i]));
  }
}

TEST(BackendTestCase, CeilCaseOutputsMatchStdCeil) {
  auto cases = CollectTestCases("Ceil");
  const TestCase *tc = FindCase(cases, "test_cc_ceil");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.outputs[0].AsFloat();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(y[i], std::ceil(x[i]));
  }
}

TEST(BackendTestCase, RoundCaseImplementsBankersRounding) {
  auto cases = CollectTestCases("Round");
  const TestCase *tc = FindCase(cases, "test_cc_round");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  // x = {0.9, 2.5, 2.3, 1.5, -4.5, -2.5}
  // Banker's rounding: 0.9->1, 2.5->2, 2.3->2, 1.5->2, -4.5->-4, -2.5->-2.
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.outputs[0].element_count(), 6);
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 2.0f);
  EXPECT_FLOAT_EQ(y[2], 2.0f);
  EXPECT_FLOAT_EQ(y[3], 2.0f);
  EXPECT_FLOAT_EQ(y[4], -4.0f);
  EXPECT_FLOAT_EQ(y[5], -2.0f);
}

TEST(BackendTestCase, HannWindowCasesArePresent) {
  auto cases = CollectTestCases("HannWindow");
  const TestCase *periodic = nullptr;
  const TestCase *symmetric = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_hannwindow")
      periodic = &c;
    if (c.name == "test_cc_hannwindow_symmetric")
      symmetric = &c;
  }
  ASSERT_NE(periodic, nullptr);
  ASSERT_NE(symmetric, nullptr);

  for (const TestCase *tc : {periodic, symmetric}) {
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 10);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(tc->model.ref_opset_import().size(), 1u);
    EXPECT_EQ(tc->model.ref_opset_import()[0].version(), 17);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const auto &op_type = graph.ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "HannWindow");
  }

  // Output values match the Hann window reference formula.
  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 0.5;
  constexpr double a1 = -0.5;
  const int32_t size = 10;
  for (const TestCase *tc : {periodic, symmetric}) {
    const double divisor =
        (tc == periodic) ? static_cast<double>(size) : static_cast<double>(size - 1);
    const float *y = tc->data_sets[0].outputs[0].AsFloat();
    for (int32_t n = 0; n < size; ++n) {
      const double k = static_cast<double>(n) / divisor;
      const float expected = static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k));
      EXPECT_FLOAT_EQ(y[n], expected);
    }
  }
}

TEST(BackendTestCase, HammingWindowCasesArePresent) {
  auto cases = CollectTestCases("HammingWindow");
  const TestCase *periodic = nullptr;
  const TestCase *symmetric = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_hammingwindow")
      periodic = &c;
    if (c.name == "test_cc_hammingwindow_symmetric")
      symmetric = &c;
  }
  ASSERT_NE(periodic, nullptr);
  ASSERT_NE(symmetric, nullptr);

  for (const TestCase *tc : {periodic, symmetric}) {
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 10);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(tc->model.ref_opset_import().size(), 1u);
    EXPECT_EQ(tc->model.ref_opset_import()[0].version(), 17);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const auto &op_type = graph.ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "HammingWindow");
  }

  // Output values match the Hamming window reference formula.
  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 25.0 / 46.0;
  constexpr double a1 = -21.0 / 46.0;
  const int32_t size = 10;
  for (const TestCase *tc : {periodic, symmetric}) {
    const double divisor =
        (tc == periodic) ? static_cast<double>(size) : static_cast<double>(size - 1);
    const float *y = tc->data_sets[0].outputs[0].AsFloat();
    for (int32_t n = 0; n < size; ++n) {
      const double k = static_cast<double>(n) / divisor;
      const float expected = static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k));
      EXPECT_FLOAT_EQ(y[n], expected);
    }
  }
}

TEST(BackendTestCase, ClipOnnxCasesArePresent) {
  const std::vector<std::string> expected_names = {
      "test_clip_example",          "test_clip",
      "test_clip_inbounds",         "test_clip_outbounds",
      "test_clip_splitbounds",      "test_clip_min_greater_than_max",
      "test_clip_default_min",      "test_clip_default_max",
      "test_clip_default_inbounds", "test_clip_default_int8_min",
      "test_clip_default_int8_max", "test_clip_default_int8_inbounds",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindCase(cases, name), nullptr) << "Missing ONNX/math case: " << name;
  }
}

TEST(BackendTestCase, ClipExampleClampsToMinAndMax) {
  auto cases = CollectTestCases("Clip");
  const TestCase *tc = FindCase(cases, "test_clip_example");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 3u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.outputs[0].element_count(), 3);
  EXPECT_FLOAT_EQ(y[0], -1.0f);
  EXPECT_FLOAT_EQ(y[1], 0.0f);
  EXPECT_FLOAT_EQ(y[2], 1.0f);
}

TEST(BackendTestCase, ClipMinGreaterThanMaxReplacesAllValuesByMax) {
  auto cases = CollectTestCases("Clip");
  const TestCase *tc = FindCase(cases, "test_clip_min_greater_than_max");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *y = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.outputs[0].element_count(), 3);
  // Per ONNX semantics: Min(max, Max(input, min)) collapses to max when
  // min > max.
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 1.0f);
  EXPECT_FLOAT_EQ(y[2], 1.0f);
}

TEST(BackendTestCase, ClipDefaultMaxNodeOmitsTrailingMinInput) {
  auto cases = CollectTestCases("Clip");
  // ``test_clip_default_max`` declares the optional ``min`` input as an
  // empty string, matching the upstream ``make_node("Clip", ["x", "",
  // "max"], ...)`` pattern. The data set itself only carries the two
  // *named* tensors (x and max).
  const TestCase *tc = FindCase(cases, "test_clip_default_max");
  ASSERT_NE(tc, nullptr);
  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const auto &node_inputs = graph.ref_node()[0].ref_input();
  ASSERT_EQ(node_inputs.size(), 3u);
  EXPECT_EQ(std::string(node_inputs[1].data(), node_inputs[1].size()), "");
  ASSERT_EQ(tc->data_sets[0].inputs.size(), 2u);
}

} // namespace Test
