// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

inline void ExpectTensorBitEqual(const runtime::Tensor &actual, const runtime::Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  EXPECT_EQ(actual.string_data, expected.string_data);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  EXPECT_EQ(std::vector<uint8_t>(actual.bytes(), actual.bytes() + actual.size_bytes()),
            std::vector<uint8_t>(expected.bytes(), expected.bytes() + expected.size_bytes()));
}

inline void ExpectBackendCaseTensorEqual(const runtime::Tensor &actual,
                                         const runtime::Tensor &expected, const TestCase &tc) {
  using runtime::DataType;
  if (tc.name.find("test_cc_supported_dtype_") != 0 ||
      tc.name.find("test_cc_supported_dtype_Transpose_") == 0 ||
      (expected.data_type != DataType::FLOAT && expected.data_type != DataType::DOUBLE &&
       expected.data_type != DataType::FLOAT16 && expected.data_type != DataType::BFLOAT16)) {
    ExpectTensorBitEqual(actual, expected);
    return;
  }
  ASSERT_EQ(actual.data_type, expected.data_type);
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  auto read_value = [](const runtime::Tensor &tensor, int64_t i) -> double {
    switch (tensor.data_type) {
    case DataType::FLOAT:
      return tensor.AsFloat()[i];
    case DataType::DOUBLE:
      return tensor.AsDouble()[i];
    case DataType::FLOAT16:
      return runtime::Float16BitsToFloat(reinterpret_cast<const uint16_t *>(tensor.bytes())[i]);
    case DataType::BFLOAT16:
      return runtime::Bfloat16BitsToFloat(reinterpret_cast<const uint16_t *>(tensor.bytes())[i]);
    default:
      throw std::invalid_argument("Unsupported floating-point comparison dtype.");
    }
  };
  for (int64_t i = 0; i < expected.element_count(); ++i) {
    const double value = read_value(actual, i);
    const double reference = read_value(expected, i);
    if (std::isnan(reference)) {
      EXPECT_TRUE(std::isnan(value)) << "element " << i;
    } else if (std::isinf(reference)) {
      EXPECT_EQ(value, reference) << "element " << i;
    } else {
      EXPECT_NEAR(value, reference, tc.atol + tc.rtol * std::abs(reference)) << "element " << i;
    }
  }
}

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
