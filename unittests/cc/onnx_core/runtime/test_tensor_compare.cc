// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tensor_compare.h"

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::CompareTensors;
using core::runtime::DataType;
using core::runtime::FloatToFloat16Bits;
using core::runtime::Shape;
using core::runtime::Tensor;
using core::runtime::TensorComparison;

namespace Test {

namespace {

Tensor MakeFloat16(const Shape &shape, const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint16_t bits = FloatToFloat16Bits(values[i]);
    bytes[2 * i] = static_cast<uint8_t>(bits & 0xFF);
    bytes[2 * i + 1] = static_cast<uint8_t>((bits >> 8) & 0xFF);
  }
  return Tensor("", static_cast<int32_t>(DataType::FLOAT16), shape, std::move(bytes));
}

} // namespace

TEST(CompareTensors, IdenticalFloat) {
  Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_TRUE(result.close) << result.message;
}

TEST(CompareTensors, WithinTolerance) {
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {1.0f + 1e-6f, 2.0f - 1e-6f});
  EXPECT_TRUE(CompareTensors(a, b).close);
}

TEST(CompareTensors, OutsideTolerance) {
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {1.0f, 2.5f});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_FALSE(result.close);
  EXPECT_NE(result.message.find("value mismatch at index 1"), std::string::npos) << result.message;
  EXPECT_EQ(result.max_abs_error_index, 1);
  EXPECT_NEAR(result.max_abs_error, 0.5, 1e-6);
  EXPECT_EQ(result.max_rel_error_index, 1);
  EXPECT_NEAR(result.max_rel_error, 0.5 / 2.5, 1e-6);
}

TEST(CompareTensors, ErrorStatsAcrossElements) {
  // Largest absolute error at index 2 (|30-33|=3), largest relative error at
  // index 0 (|1-1.5|/1.5 = 0.333 > 3/33 = 0.0909).
  Tensor a = Tensor::FromFloat("", {3}, {1.0f, 20.0f, 30.0f});
  Tensor b = Tensor::FromFloat("", {3}, {1.5f, 20.0f, 33.0f});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_FALSE(result.close);
  EXPECT_EQ(result.max_abs_error_index, 2);
  EXPECT_NEAR(result.max_abs_error, 3.0, 1e-5);
  EXPECT_EQ(result.max_rel_error_index, 0);
  EXPECT_NEAR(result.max_rel_error, 0.5 / 1.5, 1e-5);
}

TEST(CompareTensors, IdenticalReportsZeroError) {
  Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor b = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_TRUE(result.close);
  EXPECT_EQ(result.max_abs_error, 0.0);
  EXPECT_EQ(result.max_rel_error, 0.0);
  EXPECT_EQ(result.max_abs_error_index, 0);
  EXPECT_EQ(result.max_rel_error_index, 0);
}

TEST(CompareTensors, CustomTolerance) {
  Tensor a = Tensor::FromFloat("", {1}, {1.0f});
  Tensor b = Tensor::FromFloat("", {1}, {1.2f});
  EXPECT_FALSE(CompareTensors(a, b).close);
  EXPECT_TRUE(CompareTensors(a, b, /*rtol=*/0.0, /*atol=*/0.5).close);
}

TEST(CompareTensors, DataTypeMismatch) {
  Tensor a = Tensor::FromFloat("", {1}, {1.0f});
  Tensor b = Tensor::FromDouble("", {1}, {1.0});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_FALSE(result.close);
  EXPECT_NE(result.message.find("data_type mismatch"), std::string::npos) << result.message;
}

TEST(CompareTensors, ShapeMismatch) {
  Tensor a = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor b = Tensor::FromFloat("", {1, 2}, {1.0f, 2.0f});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_FALSE(result.close);
  EXPECT_NE(result.message.find("shape mismatch"), std::string::npos) << result.message;
}

TEST(CompareTensors, NanUnequalByDefault) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor a = Tensor::FromFloat("", {1}, {nan});
  Tensor b = Tensor::FromFloat("", {1}, {nan});
  EXPECT_FALSE(CompareTensors(a, b).close);
  EXPECT_TRUE(CompareTensors(a, b, 1e-5, 1e-8, /*equal_nan=*/true).close);
}

TEST(CompareTensors, NanPositionMismatch) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor a = Tensor::FromFloat("", {2}, {nan, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  const TensorComparison result = CompareTensors(a, b, 1e-5, 1e-8, /*equal_nan=*/true);
  EXPECT_FALSE(result.close);
  EXPECT_NE(result.message.find("NaN position mismatch at index 0"), std::string::npos)
      << result.message;
}

TEST(CompareTensors, InfinityMatches) {
  const float inf = std::numeric_limits<float>::infinity();
  Tensor a = Tensor::FromFloat("", {2}, {inf, -inf});
  Tensor b = Tensor::FromFloat("", {2}, {inf, -inf});
  EXPECT_TRUE(CompareTensors(a, b).close);
  Tensor c = Tensor::FromFloat("", {2}, {inf, inf});
  EXPECT_FALSE(CompareTensors(a, c).close);
}

TEST(CompareTensors, InfinityPositionMismatch) {
  const float inf = std::numeric_limits<float>::infinity();
  Tensor a = Tensor::FromFloat("", {2}, {inf, 2.0f});
  Tensor b = Tensor::FromFloat("", {2}, {3.0f, 2.0f});
  const TensorComparison result = CompareTensors(a, b);
  EXPECT_FALSE(result.close);
  EXPECT_NE(result.message.find("infinity mismatch at index 0"), std::string::npos)
      << result.message;
}

TEST(CompareTensors, IntegerExact) {
  Tensor a = Tensor::From<int64_t>("", {3}, {1, 2, 3});
  Tensor b = Tensor::From<int64_t>("", {3}, {1, 2, 3});
  EXPECT_TRUE(CompareTensors(a, b).close);
  Tensor c = Tensor::From<int64_t>("", {3}, {1, 2, 4});
  EXPECT_FALSE(CompareTensors(a, c).close);
}

TEST(CompareTensors, StringExact) {
  Tensor a = Tensor::FromStrings("", {2}, {"a", "b"});
  Tensor b = Tensor::FromStrings("", {2}, {"a", "b"});
  EXPECT_TRUE(CompareTensors(a, b).close);
  Tensor c = Tensor::FromStrings("", {2}, {"a", "c"});
  const TensorComparison result = CompareTensors(a, c);
  EXPECT_FALSE(result.close);
  EXPECT_NE(result.message.find("string mismatch"), std::string::npos) << result.message;
}

TEST(CompareTensors, Float16WithinTolerance) {
  Tensor a = MakeFloat16({3}, {1.0f, 2.0f, 3.0f});
  Tensor b = MakeFloat16({3}, {1.0f, 2.0f, 3.0f});
  EXPECT_TRUE(CompareTensors(a, b).close);
  Tensor c = MakeFloat16({3}, {1.0f, 2.0f, 3.5f});
  EXPECT_FALSE(CompareTensors(a, c).close);
}

} // namespace Test
