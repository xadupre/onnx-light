// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/kernels/tensor/cast_float8.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::DequantizeLinear;
using onnx_kernels::kernel::DynamicQuantizeLinear;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::QuantizeLinear;

namespace Test {

TEST(KernelClass, QuantizeLinearDefaultUint8) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  Tensor y = q(x, scale);
  ASSERT_EQ(y.element_count(), 6);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  // round-half-to-even(x / 2) + 0, saturated to [0, 255].
  EXPECT_EQ(static_cast<int>(y.data[0]), 0);
  EXPECT_EQ(static_cast<int>(y.data[1]), 1);
  EXPECT_EQ(static_cast<int>(y.data[2]), 2);   // 1.5 -> 2 (round half to even)
  EXPECT_EQ(static_cast<int>(y.data[3]), 255); // saturates above
  EXPECT_EQ(static_cast<int>(y.data[4]), 0);   // saturates below 0
  EXPECT_EQ(static_cast<int>(y.data[5]), 0);   // saturates below 0
}

TEST(KernelClass, QuantizeLinearInt8WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, -2.0f, 300.0f});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  // y_zero_point is INT8 = -10.
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::INT8), {},
                  std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
  Tensor y = q(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT8));
  const int8_t *py = reinterpret_cast<const int8_t *>(y.data.data());
  EXPECT_EQ(static_cast<int>(py[0]), -10);
  EXPECT_EQ(static_cast<int>(py[1]), -9);
  EXPECT_EQ(static_cast<int>(py[2]), -11);
  EXPECT_EQ(static_cast<int>(py[3]), 127); // 300/2 + (-10) = 140 -> saturates at INT8 max
}

TEST(KernelClass, QuantizeLinearUint16WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, 3.0f, 200000.0f});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  const uint16_t zp_value = 32767;
  std::vector<uint8_t> zp_bytes(sizeof(uint16_t));
  std::memcpy(zp_bytes.data(), &zp_value, sizeof(uint16_t));
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::UINT16), {}, zp_bytes);
  Tensor y = q(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT16));
  const uint16_t *py = reinterpret_cast<const uint16_t *>(y.data.data());
  EXPECT_EQ(py[0], static_cast<uint16_t>(32767));
  EXPECT_EQ(py[1], static_cast<uint16_t>(32768));
  EXPECT_EQ(py[2], static_cast<uint16_t>(32769));
  EXPECT_EQ(py[3], std::numeric_limits<uint16_t>::max());
}

TEST(KernelClass, QuantizeLinearInt16WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, 3.0f, -100000.0f});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  const int16_t zp_value = -1024;
  std::vector<uint8_t> zp_bytes(sizeof(int16_t));
  std::memcpy(zp_bytes.data(), &zp_value, sizeof(int16_t));
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::INT16), {}, zp_bytes);
  Tensor y = q(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT16));
  const int16_t *py = reinterpret_cast<const int16_t *>(y.data.data());
  EXPECT_EQ(py[0], static_cast<int16_t>(-1024));
  EXPECT_EQ(py[1], static_cast<int16_t>(-1023));
  EXPECT_EQ(py[2], static_cast<int16_t>(-1022));
  EXPECT_EQ(py[3], std::numeric_limits<int16_t>::min());
}

TEST(KernelClass, QuantizeLinearRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {0.0f, 1.0f, 2.0f});
  Tensor scale = Tensor::FromFloat("", {}, {1.0f});
  // Non-scalar y_scale (per-axis quantization is not supported).
  Tensor bad_scale = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(q(x, bad_scale), std::invalid_argument);
  // x with non-FLOAT element type.
  Tensor bad_x = Tensor::FromInt32("", {3}, {0, 1, 2});
  EXPECT_THROW(q(bad_x, scale), std::invalid_argument);
  // Unsupported zero-point element type (INT32).
  const Tensor zp_int32 = Tensor::FromInt32("", {}, {0});
  EXPECT_THROW(q(x, scale, zp_int32), std::invalid_argument);
}

TEST(KernelClass, DequantizeLinearDefaultUint8) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  Tensor x = Tensor::FromUint8("", {4}, {0, 3, 128, 255});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  Tensor y = d(x, scale);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  // (x - 0) * 2.0
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 6.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 256.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 510.0f);
}

TEST(KernelClass, DequantizeLinearInt8WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  Tensor x = Tensor::FromInt8("", {4}, {-10, -9, 0, 127});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::INT8), {},
                  std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
  Tensor y = d(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  // (x - (-10)) * 2.0
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 20.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 274.0f);
}

TEST(KernelClass, DequantizeLinearRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  Tensor x = Tensor::FromUint8("", {3}, {0, 1, 2});
  Tensor scale = Tensor::FromFloat("", {}, {1.0f});
  // Non-scalar x_scale.
  Tensor bad_scale = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(d(x, bad_scale), std::invalid_argument);
  // Mismatched x_zero_point dtype (INT8 vs UINT8 x).
  const Tensor zp_int8("", static_cast<int32_t>(onnx_kernels::DataType::INT8), {},
                       std::vector<uint8_t>(1, 0));
  EXPECT_THROW(d(x, scale, zp_int8), std::invalid_argument);
  // Unsupported x element type (FLOAT).
  Tensor bad_x = Tensor::FromFloat("", {3}, {0.0f, 1.0f, 2.0f});
  EXPECT_THROW(d(bad_x, scale), std::invalid_argument);
}

TEST(KernelClass, DequantizeLinearUint16WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  Tensor x = Tensor::FromUint16("", {4}, {30000, 31000, 32768, 33000});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  const uint16_t zp_value = 32767;
  std::vector<uint8_t> zp_bytes(sizeof(uint16_t));
  std::memcpy(zp_bytes.data(), &zp_value, sizeof(uint16_t));
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::UINT16), {}, zp_bytes);
  Tensor y = d(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  // (x - 32767) * 2.0
  EXPECT_FLOAT_EQ(y.AsFloat()[0], -5534.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], -3534.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 466.0f);
}

TEST(KernelClass, DequantizeLinearInt16WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  Tensor x = Tensor::FromInt16("", {4}, {-300, -30, -1025, 1270});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  const int16_t zp_value = -1024;
  std::vector<uint8_t> zp_bytes(sizeof(int16_t));
  std::memcpy(zp_bytes.data(), &zp_value, sizeof(int16_t));
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::INT16), {}, zp_bytes);
  Tensor y = d(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  // (x - (-1024)) * 2.0
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1448.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 1988.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], -2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 4588.0f);
}

TEST(KernelClass, DequantizeLinearFloat8E4M3FNNoZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  const std::vector<float> fvals = {0.0f, 0.5f, 1.0f, 448.0f, -104.0f};
  std::vector<uint8_t> bytes(fvals.size());
  for (size_t i = 0; i < fvals.size(); ++i) {
    bytes[i] = onnx_kernels::kernel::FloatToFloat8E4M3FNBits(fvals[i]);
  }
  const Tensor x("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN), {5}, bytes);
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  Tensor y = d(x, scale);
  ASSERT_EQ(y.element_count(), 5);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 1.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 896.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[4], -208.0f);
}

TEST(KernelClass, DequantizeLinearFloat8E5M2NoZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  const std::vector<float> fvals = {0.0f, 0.5f, 1.0f, 49152.0f, -96.0f};
  std::vector<uint8_t> bytes(fvals.size());
  for (size_t i = 0; i < fvals.size(); ++i) {
    bytes[i] = onnx_kernels::kernel::FloatToFloat8E5M2Bits(fvals[i]);
  }
  const Tensor x("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E5M2), {5}, bytes);
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  Tensor y = d(x, scale);
  ASSERT_EQ(y.element_count(), 5);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 1.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 98304.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[4], -192.0f);
}

TEST(KernelClass, DequantizeLinearFloat8E4M3FNWithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  DequantizeLinear d{ctx};
  const std::vector<float> fvals = {0.0f, 0.5f, 1.0f, 448.0f, -104.0f};
  std::vector<uint8_t> bytes(fvals.size());
  for (size_t i = 0; i < fvals.size(); ++i) {
    bytes[i] = onnx_kernels::kernel::FloatToFloat8E4M3FNBits(fvals[i]);
  }
  const Tensor x("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN), {5}, bytes);
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  const Tensor zp("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT8E4M3FN), {1},
                  std::vector<uint8_t>{onnx_kernels::kernel::FloatToFloat8E4M3FNBits(0.0f)});
  Tensor y = d(x, scale, zp);
  ASSERT_EQ(y.element_count(), 5);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 1.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[3], 896.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[4], -208.0f);
}

TEST(KernelClass, DynamicQuantizeLinearStraddleZero) {
  // Mirrors the upstream ``DynamicQuantizeLinear.export()`` test:
  // expected scale 0.0196078438 and zero point 153.
  const KernelContext ctx{DefaultOpset(11)};
  DynamicQuantizeLinear d{ctx};
  Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, -3.0f, -2.5f, 1.34f, 0.5f});
  auto [y, y_scale, y_zero_point] = d(x);

  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  ASSERT_EQ(y.element_count(), 6);
  ASSERT_EQ(y_scale.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  ASSERT_TRUE(y_scale.shape.empty());
  ASSERT_EQ(y_zero_point.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  ASSERT_TRUE(y_zero_point.shape.empty());

  EXPECT_NEAR(y_scale.AsFloat()[0], 0.0196078438f, 1e-7f);
  EXPECT_EQ(static_cast<int>(y_zero_point.data[0]), 153);
}

TEST(KernelClass, DynamicQuantizeLinearAllNegative) {
  // Mirrors ``DynamicQuantizeLinear.export_max_adjusted()``: all-negative
  // input, max gets clipped to 0; expected scale 0.0156862754 and zero
  // point 255.
  const KernelContext ctx{DefaultOpset(11)};
  DynamicQuantizeLinear d{ctx};
  Tensor x = Tensor::FromFloat("", {6}, {-1.0f, -2.1f, -1.3f, -2.5f, -3.34f, -4.0f});
  auto [y, y_scale, y_zero_point] = d(x);

  EXPECT_NEAR(y_scale.AsFloat()[0], 0.0156862754f, 1e-7f);
  EXPECT_EQ(static_cast<int>(y_zero_point.data[0]), 255);
}

TEST(KernelClass, DynamicQuantizeLinearAllPositive) {
  // Mirrors ``DynamicQuantizeLinear.export_min_adjusted()``: all-positive
  // 2-D input, min gets clipped to 0; expected scale 0.0156862754 and
  // zero point 0.
  const KernelContext ctx{DefaultOpset(11)};
  DynamicQuantizeLinear d{ctx};
  Tensor x = Tensor::FromFloat(
      "", {3, 4}, {1.0f, 2.1f, 1.3f, 2.5f, 3.34f, 4.0f, 1.5f, 2.6f, 3.9f, 4.0f, 3.0f, 2.345f});
  auto [y, y_scale, y_zero_point] = d(x);

  EXPECT_NEAR(y_scale.AsFloat()[0], 0.0156862754f, 1e-7f);
  EXPECT_EQ(static_cast<int>(y_zero_point.data[0]), 0);
  ASSERT_EQ(y.shape.size(), 2u);
  EXPECT_EQ(y.shape[0], 3);
  EXPECT_EQ(y.shape[1], 4);
}

} // namespace Test
