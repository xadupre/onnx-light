// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::QuantizeLinear;

namespace Test {

TEST(BackendKernelClass, QuantizeLinearDefaultUint8) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {6}, {0.0f, 2.0f, 3.0f, 1000.0f, -254.0f, -1000.0f});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  Tensor y = q(x, scale);
  ASSERT_EQ(y.element_count(), 6);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::UINT8));
  // round-half-to-even(x / 2) + 0, saturated to [0, 255].
  EXPECT_EQ(static_cast<int>(y.data[0]), 0);
  EXPECT_EQ(static_cast<int>(y.data[1]), 1);
  EXPECT_EQ(static_cast<int>(y.data[2]), 2);   // 1.5 -> 2 (round half to even)
  EXPECT_EQ(static_cast<int>(y.data[3]), 255); // saturates above
  EXPECT_EQ(static_cast<int>(y.data[4]), 0);   // saturates below 0
  EXPECT_EQ(static_cast<int>(y.data[5]), 0);   // saturates below 0
}

TEST(BackendKernelClass, QuantizeLinearInt8WithZeroPoint) {
  const KernelContext ctx{DefaultOpset(13)};
  QuantizeLinear q{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {0.0f, 2.0f, -2.0f, 300.0f});
  Tensor scale = Tensor::FromFloat("", {}, {2.0f});
  // y_zero_point is INT8 = -10.
  const Tensor zp("", static_cast<int32_t>(TensorProto::DataType::INT8), {},
                  std::vector<uint8_t>(1, static_cast<uint8_t>(static_cast<int8_t>(-10))));
  Tensor y = q(x, scale, zp);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT8));
  const int8_t *py = reinterpret_cast<const int8_t *>(y.data.data());
  EXPECT_EQ(static_cast<int>(py[0]), -10);
  EXPECT_EQ(static_cast<int>(py[1]), -9);
  EXPECT_EQ(static_cast<int>(py[2]), -11);
  EXPECT_EQ(static_cast<int>(py[3]), 127); // 300/2 + (-10) = 140 -> saturates at INT8 max
}

TEST(BackendKernelClass, QuantizeLinearRejectsBadInputs) {
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

} // namespace Test
