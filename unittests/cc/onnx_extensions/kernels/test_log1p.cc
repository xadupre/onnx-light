// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;
using core::backend_test::DefaultOpset;
using onnx_kernels::kernel::Log1p;

namespace Test {

TEST(KernelLog1p, FloatAndHalfPreserveSmallValuesAndSpecialValues) {
  const float infinity = std::numeric_limits<float>::infinity();
  const std::vector<float> values = {-0.0f,     0.0f,     -0x1p-24f,
                                     0x1p-24f,  -1.0f,    1.0f,
                                     -infinity, infinity, std::numeric_limits<float>::quiet_NaN(),
                                     -2.0f,     -0.9375f};
  const Log1p kernel{KernelContext{DefaultOpset(29)}};
  for (DataType dtype : {DataType::FLOAT, DataType::FLOAT16, DataType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int32_t>(dtype));
    Tensor x = dtype == DataType::FLOAT16    ? MakeFloat16Tensor("", {11}, values)
               : dtype == DataType::BFLOAT16 ? MakeBfloat16Tensor("", {11}, values)
                                             : Tensor::FromFloat("", {11}, values);
    const Tensor y = kernel(x);
    EXPECT_EQ(y.data_type, x.data_type);
    EXPECT_EQ(y.shape, x.shape);
    const Tensor promoted = dtype == DataType::FLOAT ? y : PromoteToFloat32(y);
    const float *actual = promoted.AsFloat();
    EXPECT_EQ(actual[0], 0.0f);
    EXPECT_EQ(actual[1], 0.0f);
    EXPECT_TRUE(std::signbit(actual[0]));
    EXPECT_FALSE(std::signbit(actual[1]));
    EXPECT_LT(actual[2], 0);
    EXPECT_GT(actual[3], 0);
    EXPECT_FLOAT_EQ(actual[2], values[2]);
    EXPECT_FLOAT_EQ(actual[3], values[3]);
    EXPECT_EQ(actual[4], -infinity);
    const float tolerance = dtype == DataType::BFLOAT16 ? 0.008f : 0.001f;
    EXPECT_NEAR(actual[5], std::log1p(1.0f), tolerance);
    EXPECT_TRUE(std::isnan(actual[6]));
    EXPECT_EQ(actual[7], infinity);
    EXPECT_TRUE(std::isnan(actual[8]));
    EXPECT_TRUE(std::isnan(actual[9]));
    EXPECT_NEAR(actual[10], std::log1p(values[10]), tolerance);
    kernel(x, x);
    const Tensor inplace = dtype == DataType::FLOAT ? x : PromoteToFloat32(x);
    EXPECT_TRUE(std::signbit(inplace.AsFloat()[0]));
    EXPECT_GT(inplace.AsFloat()[3], 0);
    EXPECT_FLOAT_EQ(inplace.AsFloat()[5], actual[5]);
  }
}

TEST(KernelLog1p, DoublePreservesPrecisionAndDomainBoundaries) {
  const Log1p kernel{KernelContext{DefaultOpset(29)}};
  const double infinity = std::numeric_limits<double>::infinity();
  const double above_minus_one = std::nextafter(-1.0, 0.0);
  const double below_minus_one = std::nextafter(-1.0, -infinity);
  const Tensor x =
      Tensor::FromDouble("", {13},
                         {-0.0, 0.0, -1e-20, 1e-20, -1.0, 1.0, -infinity, infinity,
                          std::numeric_limits<double>::quiet_NaN(), above_minus_one,
                          below_minus_one, std::numeric_limits<double>::max(), 1.000000000000001});
  const Tensor y = kernel(x);
  const double *actual = y.AsDouble();
  EXPECT_EQ(actual[0], 0.0);
  EXPECT_EQ(actual[1], 0.0);
  EXPECT_TRUE(std::signbit(actual[0]));
  EXPECT_FALSE(std::signbit(actual[1]));
  EXPECT_DOUBLE_EQ(actual[2], -1e-20);
  EXPECT_DOUBLE_EQ(actual[3], 1e-20);
  EXPECT_EQ(actual[4], -infinity);
  EXPECT_DOUBLE_EQ(actual[5], std::log1p(1.0));
  EXPECT_TRUE(std::isnan(actual[6]));
  EXPECT_EQ(actual[7], infinity);
  EXPECT_TRUE(std::isnan(actual[8]));
  EXPECT_DOUBLE_EQ(actual[9], std::log1p(above_minus_one));
  EXPECT_TRUE(std::isnan(actual[10]));
  EXPECT_DOUBLE_EQ(actual[11], std::log1p(std::numeric_limits<double>::max()));
  EXPECT_DOUBLE_EQ(actual[12], std::log1p(1.000000000000001));
}

TEST(KernelLog1p, ScalarAndEmptyPreserveShapeAndType) {
  const Log1p kernel{KernelContext{DefaultOpset(29)}};
  for (DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    for (const Shape &shape : {Shape{}, Shape{2, 0, 3}}) {
      SCOPED_TRACE(static_cast<int32_t>(dtype));
      const std::vector<float> values =
          shape.empty() ? std::vector<float>{-0.0f} : std::vector<float>{};
      const Tensor x =
          dtype == DataType::DOUBLE
              ? Tensor::FromDouble(
                    "", shape, shape.empty() ? std::vector<double>{-0.0} : std::vector<double>{})
          : dtype == DataType::FLOAT16  ? MakeFloat16Tensor("", shape, values)
          : dtype == DataType::BFLOAT16 ? MakeBfloat16Tensor("", shape, values)
                                        : Tensor::FromFloat("", shape, values);
      const Tensor y = kernel(x);
      EXPECT_EQ(y.shape, shape);
      EXPECT_EQ(y.data_type, x.data_type);
      EXPECT_EQ(y.element_count(), shape.empty() ? 1 : 0);
      if (shape.empty()) {
        EXPECT_TRUE(dtype == DataType::DOUBLE
                        ? std::signbit(y.AsDouble()[0])
                        : std::signbit(dtype == DataType::FLOAT
                                           ? y.AsFloat()[0]
                                           : PromoteToFloat32(y).AsFloat()[0]));
      }
    }
  }
}

TEST(KernelLog1p, ValidatesOutputAndUnsupportedTypes) {
  const Log1p kernel{KernelContext{DefaultOpset(29)}};
  const Tensor x = Tensor::FromFloat("", {2}, {0, 1});
  Tensor wrong_type = Tensor::FromDouble("", {2}, {0, 0});
  Tensor wrong_shape = Tensor::FromFloat("", {1, 2}, {0, 0});
  Tensor wrong_size("", DataType::FLOAT, {2}, std::vector<uint8_t>(sizeof(float)));
  Tensor malformed_input("", DataType::FLOAT, {2}, std::vector<uint8_t>(sizeof(float)));
  EXPECT_THROW(kernel(x, wrong_type), std::invalid_argument);
  EXPECT_THROW(kernel(x, wrong_shape), std::invalid_argument);
  EXPECT_THROW(kernel(x, wrong_size), std::invalid_argument);
  EXPECT_THROW(kernel(malformed_input), std::invalid_argument);
  EXPECT_THROW(kernel(malformed_input, wrong_size), std::invalid_argument);
  const Tensor oversized("", DataType::FLOAT, {std::numeric_limits<int64_t>::max()}, {});
  EXPECT_THROW(kernel(oversized), std::invalid_argument);
  EXPECT_THROW(kernel(Tensor::FromInt32("", {1}, {1})), std::invalid_argument);
  EXPECT_THROW(kernel(Tensor::FromInt32("", {0}, {})), std::invalid_argument);
}

TEST(KernelLog1p, RegistersTypedTuningAndInplaceCapability) {
  onnx_kernels::RegisterKernelFunctions();
  Log1p kernel{KernelContext{DefaultOpset(29)}};
  EXPECT_TRUE(Log1p::CanRunInPlace());
  for (DataType dtype :
       {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16, DataType::BFLOAT16}) {
    const auto key = kernel.TuningKey(static_cast<int32_t>(dtype));
    EXPECT_EQ(key.kernel, "Log1p");
    ASSERT_NE(GetKernelTuningRegistry().FindSchema(key), nullptr);
    kernel.Configure(KernelTuningParameters{key, {{"parallel.minimum_elements", int64_t{1}}}});
    EXPECT_EQ(kernel.tuning().parallel_minimum_elements, 1);
  }
  EXPECT_EQ(kernel.TuningKey(static_cast<int32_t>(DataType::INT32)).device,
            core::symbolic::Device::kUndefined);
}

} // namespace Test
