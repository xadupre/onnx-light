// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectLogicalTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectLogicalTestCases(registry, op_type);
  return registry;
}
} // namespace
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
  auto cases = CollectTestCases("And");
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
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
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
  auto cases = CollectTestCases("And");
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
  auto cases = CollectTestCases("And");
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
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
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
  auto cases = CollectTestCases("Or");
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
  auto cases = CollectTestCases("Xor");
  for (const auto &name : expected_names) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr)
        << "Missing upstream ONNX Xor test case: " << name;
  }
}

TEST(BackendTestCase, OrCaseOutputsAreElementwiseOr) {
  auto cases = CollectTestCases("Or");
  const TestCase *tc = FindLogicalCase(cases, "test_or2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  const uint8_t *x = ds.inputs[0].data.data();
  const uint8_t *y = ds.inputs[1].data.data();
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], (x[i] != 0 || y[i] != 0) ? 1 : 0);
  }
}

TEST(BackendTestCase, XorCaseOutputsAreElementwiseXor) {
  auto cases = CollectTestCases("Xor");
  const TestCase *tc = FindLogicalCase(cases, "test_xor2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
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
  // mirrored ``test_greater``/``test_less`` cases for every numeric input
  // dtype supported by ``kernel::Greater``/``kernel::Less`` (FLOAT, INT8,
  // INT16, UINT8, UINT16, UINT32, UINT64) plus the float broadcast cases
  // (see ``RegisterGreaterCases``/``RegisterLessCases``).
  auto cases = CollectTestCases();
  for (const char *name :
       {"test_cc_greater",     "test_cc_greater_bcast", "test_cc_less",        "test_cc_less_bcast",
        "test_greater",        "test_greater_int8",     "test_greater_int16",  "test_greater_uint8",
        "test_greater_uint16", "test_greater_uint32",   "test_greater_uint64", "test_greater_bcast",
        "test_less",           "test_less_int8",        "test_less_int16",     "test_less_uint8",
        "test_less_uint16",    "test_less_uint32",      "test_less_uint64",    "test_less_bcast"}) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr) << "Missing Greater/Less case: " << name;
  }
}

TEST(BackendTestCase, GreaterCaseOutputsAreElementwiseGreater) {
  auto cases = CollectTestCases("Greater");
  const TestCase *tc = FindLogicalCase(cases, "test_greater");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
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
  auto cases = CollectTestCases("Less");
  const TestCase *tc = FindLogicalCase(cases, "test_less");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL))
        << name;
  }
}

TEST(BackendTestCase, GreaterLessIntegerCasesUseRequestedDtype) {
  auto cases = CollectTestCases();
  struct Expected {
    const char *name;
    int32_t dtype;
  };
  for (const Expected &e : {
           Expected{"test_greater_int8", onnx_backend_test::DataType::INT8},
           Expected{"test_greater_int16", onnx_backend_test::DataType::INT16},
           Expected{"test_greater_uint8", onnx_backend_test::DataType::UINT8},
           Expected{"test_greater_uint16", onnx_backend_test::DataType::UINT16},
           Expected{"test_greater_uint32", onnx_backend_test::DataType::UINT32},
           Expected{"test_greater_uint64", onnx_backend_test::DataType::UINT64},
           Expected{"test_less_int8", onnx_backend_test::DataType::INT8},
           Expected{"test_less_int16", onnx_backend_test::DataType::INT16},
           Expected{"test_less_uint8", onnx_backend_test::DataType::UINT8},
           Expected{"test_less_uint16", onnx_backend_test::DataType::UINT16},
           Expected{"test_less_uint32", onnx_backend_test::DataType::UINT32},
           Expected{"test_less_uint64", onnx_backend_test::DataType::UINT64},
       }) {
    const TestCase *tc = FindLogicalCase(cases, e.name);
    ASSERT_NE(tc, nullptr) << e.name;
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u) << e.name;
    EXPECT_EQ(ds.inputs[0].data_type, e.dtype) << e.name;
    EXPECT_EQ(ds.inputs[1].data_type, e.dtype) << e.name;
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL))
        << e.name;
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << e.name;
    EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 4, 5})) << e.name;
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << e.name;
  }
}

TEST(BackendTestCase, GreaterOrEqualCasesArePresent) {
  // Local ``test_cc_*`` smoke cases registered alongside the upstream-ONNX
  // mirrored ``test_greater_equal*`` cases for every numeric input dtype
  // supported by ``kernel::GreaterOrEqual`` (FLOAT, INT8, INT16, UINT8,
  // UINT16, UINT32, UINT64) plus the float broadcast case (see
  // ``RegisterGreaterOrEqualCases``).
  auto cases = CollectTestCases();
  for (const char *name :
       {"test_cc_greater_or_equal", "test_cc_greater_or_equal_bcast", "test_greater_equal",
        "test_greater_equal_int8", "test_greater_equal_int16", "test_greater_equal_uint8",
        "test_greater_equal_uint16", "test_greater_equal_uint32", "test_greater_equal_uint64",
        "test_greater_equal_bcast"}) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr) << "Missing GreaterOrEqual case: " << name;
  }
}

TEST(BackendTestCase, GreaterOrEqualCaseOutputsAreElementwiseGreaterOrEqual) {
  auto cases = CollectTestCases("GreaterOrEqual");
  const TestCase *tc = FindLogicalCase(cases, "test_greater_equal");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  const float *x = reinterpret_cast<const float *>(ds.inputs[0].data.data());
  const float *y = reinterpret_cast<const float *>(ds.inputs[1].data.data());
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], x[i] >= y[i] ? 1 : 0);
  }
}

TEST(BackendTestCase, GreaterOrEqualBroadcastCaseHasBroadcastShapes) {
  auto cases = CollectTestCases("GreaterOrEqual");
  const TestCase *tc = FindLogicalCase(cases, "test_greater_equal_bcast");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{5}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
}

TEST(BackendTestCase, EqualCasesArePresent) {
  // Local ``test_cc_*`` smoke cases registered alongside the upstream-ONNX
  // mirrored ``test_equal*`` cases for every dtype covered by upstream
  // ``Equal.export()`` (INT32, INT8, INT16, UINT8, UINT16, UINT32, UINT64),
  // the broadcast variant, and the two STRING variants from
  // ``Equal.export_equal_string``/``export_equal_string_broadcast`` — see
  // ``RegisterEqualCases``.
  auto cases = CollectTestCases("Equal");
  for (const char *name :
       {"test_cc_equal", "test_cc_equal_bcast", "test_equal", "test_equal_int8", "test_equal_int16",
        "test_equal_uint8", "test_equal_uint16", "test_equal_uint32", "test_equal_uint64",
        "test_equal_bcast", "test_equal_string", "test_equal_string_broadcast"}) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr) << "Missing Equal case: " << name;
  }
}

TEST(BackendTestCase, EqualCaseOutputsAreElementwiseEqual) {
  auto cases = CollectTestCases("Equal");
  const TestCase *tc = FindLogicalCase(cases, "test_equal");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  const int32_t *x = reinterpret_cast<const int32_t *>(ds.inputs[0].data.data());
  const int32_t *y = reinterpret_cast<const int32_t *>(ds.inputs[1].data.data());
  const uint8_t *z = ds.outputs[0].data.data();
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], x[i] == y[i] ? 1 : 0);
  }
}

TEST(BackendTestCase, EqualBroadcastCaseHasBroadcastShapes) {
  auto cases = CollectTestCases("Equal");
  const TestCase *tc = FindLogicalCase(cases, "test_equal_bcast");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{5}));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5}));
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
}

TEST(BackendTestCase, EqualIntegerCasesUseRequestedDtype) {
  auto cases = CollectTestCases("Equal");
  struct Expected {
    const char *name;
    int32_t dtype;
  };
  for (const Expected &e : {
           Expected{"test_equal", onnx_backend_test::DataType::INT32},
           Expected{"test_equal_int8", onnx_backend_test::DataType::INT8},
           Expected{"test_equal_int16", onnx_backend_test::DataType::INT16},
           Expected{"test_equal_uint8", onnx_backend_test::DataType::UINT8},
           Expected{"test_equal_uint16", onnx_backend_test::DataType::UINT16},
           Expected{"test_equal_uint32", onnx_backend_test::DataType::UINT32},
           Expected{"test_equal_uint64", onnx_backend_test::DataType::UINT64},
       }) {
    const TestCase *tc = FindLogicalCase(cases, e.name);
    ASSERT_NE(tc, nullptr) << e.name;
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u) << e.name;
    EXPECT_EQ(ds.inputs[0].data_type, e.dtype) << e.name;
    EXPECT_EQ(ds.inputs[1].data_type, e.dtype) << e.name;
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL))
        << e.name;
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << e.name;
    EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 4, 5})) << e.name;
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4, 5})) << e.name;
  }
}

TEST(BackendTestCase, EqualStringCasesHaveExpectedShapesAndDtype) {
  auto cases = CollectTestCases("Equal");
  for (const char *name : {"test_equal_string", "test_equal_string_broadcast"}) {
    const TestCase *tc = FindLogicalCase(cases, name);
    ASSERT_NE(tc, nullptr) << name;
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u) << name;
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING))
        << name;
    EXPECT_EQ(ds.inputs[1].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING))
        << name;
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL))
        << name;
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{2})) << name;
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2})) << name;
  }
  const TestCase *eq_str = FindLogicalCase(cases, "test_equal_string");
  ASSERT_NE(eq_str, nullptr);
  EXPECT_EQ(eq_str->data_sets[0].outputs[0].data[0], 1);
  EXPECT_EQ(eq_str->data_sets[0].outputs[0].data[1], 0);
  const TestCase *eq_str_bcast = FindLogicalCase(cases, "test_equal_string_broadcast");
  ASSERT_NE(eq_str_bcast, nullptr);
  EXPECT_EQ(eq_str_bcast->data_sets[0].inputs[1].shape, (std::vector<int64_t>{1}));
  EXPECT_EQ(eq_str_bcast->data_sets[0].outputs[0].data[0], 1);
  EXPECT_EQ(eq_str_bcast->data_sets[0].outputs[0].data[1], 0);
}

TEST(BackendTestCase, WhereCasesArePresent) {
  auto cases = CollectTestCases("Where");
  EXPECT_NE(FindLogicalCase(cases, "test_where_example"), nullptr);
  EXPECT_NE(FindLogicalCase(cases, "test_where_bcast"), nullptr);
}

TEST(BackendTestCase, WhereCaseOutputsSelectExpectedElements) {
  auto cases = CollectTestCases("Where");
  const TestCase *tc = FindLogicalCase(cases, "test_where_example");
  ASSERT_NE(tc, nullptr);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 3u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  const float *z = reinterpret_cast<const float *>(ds.outputs[0].data.data());
  ASSERT_EQ(ds.outputs[0].element_count(), 4);
  EXPECT_FLOAT_EQ(z[0], 1.0f);
  EXPECT_FLOAT_EQ(z[1], 6.0f);
  EXPECT_FLOAT_EQ(z[2], 3.0f);
  EXPECT_FLOAT_EQ(z[3], 8.0f);
}

TEST(BackendTestCase, BitwiseCasesArePresent) {
  // Local ``test_cc_bitwise_*`` smoke cases plus upstream-ONNX-mirrored
  // cases registered by ``RegisterBitwise{And,Or,Xor,Not}Cases``.
  auto cases = CollectTestCases();
  for (const char *name : {
           // Local smoke cases.
           "test_cc_bitwise_and",
           "test_cc_bitwise_or",
           "test_cc_bitwise_xor",
           "test_cc_bitwise_not",
           // Upstream BitwiseAnd cases.
           "test_bitwise_and_i32_2d",
           "test_bitwise_and_i16_3d",
           "test_bitwise_and_ui64_bcast_3v1d",
           "test_bitwise_and_ui8_bcast_4v3d",
           // Upstream BitwiseOr cases.
           "test_bitwise_or_i32_2d",
           "test_bitwise_or_i16_4d",
           "test_bitwise_or_ui64_bcast_3v1d",
           "test_bitwise_or_ui8_bcast_4v3d",
           // Upstream BitwiseXor cases.
           "test_bitwise_xor_i32_2d",
           "test_bitwise_xor_i16_3d",
           "test_bitwise_xor_ui64_bcast_3v1d",
           "test_bitwise_xor_ui8_bcast_4v3d",
           // Upstream BitwiseNot cases.
           "test_bitwise_not_2d",
           "test_bitwise_not_3d",
           "test_bitwise_not_4d",
       }) {
    EXPECT_NE(FindLogicalCase(cases, name), nullptr) << "Missing bitwise case: " << name;
  }
}

TEST(BackendTestCase, BitwiseAndI32CaseOutputsAreElementwiseAnd) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindLogicalCase(cases, "test_bitwise_and_i32_2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  const int32_t *x = reinterpret_cast<const int32_t *>(ds.inputs[0].data.data());
  const int32_t *y = reinterpret_cast<const int32_t *>(ds.inputs[1].data.data());
  const int32_t *z = reinterpret_cast<const int32_t *>(ds.outputs[0].data.data());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_EQ(z[i], x[i] & y[i]);
  }
}

TEST(BackendTestCase, BitwiseNot2dCaseOutputsAreElementwiseNot) {
  auto cases = CollectTestCases();
  const TestCase *tc = FindLogicalCase(cases, "test_bitwise_not_2d");
  ASSERT_NE(tc, nullptr);
  const auto &ds = tc->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.inputs[0].data_type, ds.outputs[0].data_type);
  EXPECT_EQ(ds.inputs[0].shape, ds.outputs[0].shape);
}

} // namespace Test