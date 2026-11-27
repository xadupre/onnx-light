// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

inline void ExpectTensorBitEqual(const runtime::Tensor &actual, const runtime::Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  EXPECT_EQ(actual.string_data, expected.string_data);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  size_t byte_index = 0;
  while (byte_index < actual.size_bytes() &&
         actual.bytes()[byte_index] == expected.bytes()[byte_index]) {
    ++byte_index;
  }
  if (byte_index == actual.size_bytes()) {
    return;
  }

  const auto actual_byte = static_cast<unsigned int>(actual.bytes()[byte_index]);
  const auto expected_byte = static_cast<unsigned int>(expected.bytes()[byte_index]);
  if (actual.data_type == expected.data_type && actual.shape == expected.shape) {
    switch (actual.data_type) {
    case runtime::DataType::FLOAT: {
      const size_t float_index = byte_index / sizeof(float);
      ADD_FAILURE() << "Tensor mismatch at byte " << byte_index << " (actual 0x" << std::hex
                    << actual_byte << ", expected 0x" << expected_byte << std::dec << "); element "
                    << float_index << " differs: actual " << std::setprecision(9)
                    << actual.AsFloat()[float_index] << ", expected "
                    << expected.AsFloat()[float_index];
      return;
    }
    case runtime::DataType::DOUBLE: {
      const size_t double_index = byte_index / sizeof(double);
      ADD_FAILURE() << "Tensor mismatch at byte " << byte_index << " (actual 0x" << std::hex
                    << actual_byte << ", expected 0x" << expected_byte << std::dec << "); element "
                    << double_index << " differs: actual " << std::setprecision(17)
                    << actual.AsDouble()[double_index] << ", expected "
                    << expected.AsDouble()[double_index];
      return;
    }
    case runtime::DataType::FLOAT16:
    case runtime::DataType::BFLOAT16: {
      const size_t half_index = byte_index / sizeof(uint16_t);
      const auto *actual_bits = reinterpret_cast<const uint16_t *>(actual.bytes());
      const auto *expected_bits = reinterpret_cast<const uint16_t *>(expected.bytes());
      const float actual_value = actual.data_type == runtime::DataType::FLOAT16
                                     ? runtime::Float16BitsToFloat(actual_bits[half_index])
                                     : runtime::Bfloat16BitsToFloat(actual_bits[half_index]);
      const float expected_value = expected.data_type == runtime::DataType::FLOAT16
                                       ? runtime::Float16BitsToFloat(expected_bits[half_index])
                                       : runtime::Bfloat16BitsToFloat(expected_bits[half_index]);
      ADD_FAILURE() << "Tensor mismatch at byte " << byte_index << " (actual 0x" << std::hex
                    << actual_byte << ", expected 0x" << expected_byte << std::dec << "); element "
                    << half_index << " differs: actual " << std::setprecision(9) << actual_value
                    << ", expected " << expected_value;
      return;
    }
    default:
      break;
    }
  }
  ADD_FAILURE() << "Tensor mismatch at byte " << byte_index << ": actual 0x" << std::hex
                << actual_byte << ", expected 0x" << expected_byte;
}

inline void ExpectBackendCaseTensorEqual(const runtime::Tensor &actual,
                                         const runtime::Tensor &expected, const TestCase &tc) {
  using runtime::DataType;
  const bool softmax_reference = tc.name.find("test_cc_reference_logsoftmax_") == 0 ||
                                 tc.name.find("test_cc_reference_sce_") == 0;
  if ((!softmax_reference && tc.name.find("test_cc_supported_dtype_") != 0) ||
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
