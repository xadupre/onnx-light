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

TEST(BackendTestCase, AndOnnxCasesArePresent) {
  // Upstream-ONNX-mirrored cases exported by ``RegisterAndCases``.
  const std::vector<std::string> expected_names = {
      "test_and2d",         "test_and3d",         "test_and4d",         "test_and_bcast3v1d",
      "test_and_bcast3v2d", "test_and_bcast4v2d", "test_and_bcast4v3d", "test_and_bcast4v4d",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    bool found = false;
    for (const auto &c : cases) {
      if (c.name == name) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Missing upstream ONNX And test case: " << name;
  }
}

TEST(BackendTestCase, AndOnnxBroadcastCaseShapesAndOutput) {
  // ``test_and_bcast4v4d`` exercises full NumPy-style broadcasting between
  // ``{1, 4, 1, 6}`` and ``{3, 1, 5, 6}`` resulting in output ``{3, 4, 5, 6}``.
  auto cases = CollectTestCases();
  const TestCase *tc = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_and_bcast4v4d") {
      tc = &c;
      break;
    }
  }
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{1, 4, 1, 6}));
  EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 1, 5, 6}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5, 6}));
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(ds.outputs[0].element_count(), 3 * 4 * 5 * 6);

  // Spot-check broadcasting correctness against the input data.
  const uint8_t *x = ds.inputs[0].data.data();
  const uint8_t *y = ds.inputs[1].data.data();
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i0 = 0; i0 < 3; ++i0) {
    for (int64_t i1 = 0; i1 < 4; ++i1) {
      for (int64_t i2 = 0; i2 < 5; ++i2) {
        for (int64_t i3 = 0; i3 < 6; ++i3) {
          // x has shape {1, 4, 1, 6}; strides (in elements) are {0, 6, 0, 1}.
          const int64_t ox = i1 * 6 + i3;
          // y has shape {3, 1, 5, 6}; strides (in elements) are {30, 0, 6, 1}.
          const int64_t oy = i0 * 30 + i2 * 6 + i3;
          const int64_t oz = ((i0 * 4 + i1) * 5 + i2) * 6 + i3;
          EXPECT_EQ(z[oz], (x[ox] != 0 && y[oy] != 0) ? 1 : 0);
        }
      }
    }
  }
}

namespace {

const TestCase *FindLogicalCase(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name)
      return &c;
  }
  return nullptr;
}

} // namespace

TEST(BackendTestCase, OrOnnxCasesArePresent) {
  // Upstream-ONNX-mirrored cases exported by ``RegisterOrCases``.
  const std::vector<std::string> expected_names = {
      "test_or2d",         "test_or3d",         "test_or4d",         "test_or_bcast3v1d",
      "test_or_bcast3v2d", "test_or_bcast4v2d", "test_or_bcast4v3d", "test_or_bcast4v4d",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr)
        << "Missing upstream ONNX Or test case: " << name;
  }
}

TEST(BackendTestCase, XorOnnxCasesArePresent) {
  // Upstream-ONNX-mirrored cases exported by ``RegisterXorCases``.
  const std::vector<std::string> expected_names = {
      "test_xor2d",         "test_xor3d",         "test_xor4d",         "test_xor_bcast3v1d",
      "test_xor_bcast3v2d", "test_xor_bcast4v2d", "test_xor_bcast4v3d", "test_xor_bcast4v4d",
  };
  auto cases = CollectTestCases();
  for (const auto &name : expected_names) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr)
        << "Missing upstream ONNX Xor test case: " << name;
  }
}

TEST(BackendTestCase, OrCaseOutputsAreElementwiseOr) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindLogicalCase(cases, "test_or2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  const uint8_t *x = ds.inputs[0].data.data();
  const uint8_t *y = ds.inputs[1].data.data();
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], (x[i] != 0 || y[i] != 0) ? 1 : 0);
  }
}

TEST(BackendTestCase, XorCaseOutputsAreElementwiseXor) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindLogicalCase(cases, "test_xor2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  const uint8_t *x = ds.inputs[0].data.data();
  const uint8_t *y = ds.inputs[1].data.data();
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], ((x[i] != 0) != (y[i] != 0)) ? 1 : 0);
  }
}

TEST(BackendTestCase, OrXorBroadcastCasesHaveBroadcastShapes) {
  auto cases = CollectTestCases();
  for (const char *name : {"test_or_bcast4v4d", "test_xor_bcast4v4d"}) {
    const TestCase *tc = FindLogicalCase(cases, name);
    ASSERT_NE(tc, nullptr) << name;
    const auto &ds = tc->data_sets[0];
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{1, 4, 1, 6})) << name;
    EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 1, 5, 6})) << name;
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5, 6})) << name;
  }
}

TEST(BackendTestCase, GreaterLessCasesArePresent) {
  // Local ``test_cc_*`` smoke cases registered alongside the upstream-ONNX
  // mirrored ``test_greater``/``test_less``/``test_greater_bcast``/
  // ``test_less_bcast`` cases (see ``RegisterGreaterCases``/``RegisterLessCases``).
  auto cases = CollectTestCases();
  for (const char *name :
       {"test_cc_greater", "test_cc_greater_bcast", "test_cc_less", "test_cc_less_bcast",
        "test_greater", "test_greater_bcast", "test_less", "test_less_bcast"}) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr) << "Missing Greater/Less case: " << name;
  }
}

TEST(BackendTestCase, GreaterCaseOutputsAreElementwiseGreater) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindLogicalCase(cases, "test_greater");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  const float *x = reinterpret_cast<const float *>(ds.inputs[0].data.data());
  const float *y = reinterpret_cast<const float *>(ds.inputs[1].data.data());
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], x[i] > y[i] ? 1 : 0);
  }
}

TEST(BackendTestCase, LessCaseOutputsAreElementwiseLess) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindLogicalCase(cases, "test_less");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  const float *x = reinterpret_cast<const float *>(ds.inputs[0].data.data());
  const float *y = reinterpret_cast<const float *>(ds.inputs[1].data.data());
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], x[i] < y[i] ? 1 : 0);
  }
}

TEST(BackendTestCase, GreaterLessBroadcastCasesHaveBroadcastShapes) {
  auto cases = CollectTestCases();
  for (const char *name : {"test_greater_bcast", "test_less_bcast"}) {
    const TestCase *tc = FindLogicalCase(cases, name);
    ASSERT_NE(tc, nullptr) << name;
    const auto &ds = tc->data_sets[0];
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << name;
    EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{5})) << name;
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << name;
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL)) << name;
  }
}

} // namespace Test
