// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Constant;
using onnx_backend_test::kernel::ConstantOfShape;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

TEST(BackendKernelClass, ConstantClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.5f, 0.0f});
  Tensor y = constant_kernel(value);
  ASSERT_EQ(y.data_type, value.data_type);
  EXPECT_EQ(y.shape, value.shape);
  ASSERT_EQ(y.element_count(), value.element_count());
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.5f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
}

TEST(BackendKernelClass, ConstantRejectsMismatchedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Constant constant_kernel{ctx};
  Tensor value = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(constant_kernel(value, bad_shape), std::invalid_argument);
  Tensor bad_type("", TensorProto::DataType::INT32, {2}, std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(constant_kernel(value, bad_type), std::invalid_argument);
}

TEST(BackendKernelClass, ConstantOfShapeFloatOnes) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {3}, {2, 3, 1});
  const Tensor value = Tensor::FromFloat("", {1}, {1.0f});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3, 1}));
  ASSERT_EQ(y.element_count(), 6);
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], 1.0f);
  }
}

TEST(BackendKernelClass, ConstantOfShapeInt64Fill) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {2}, {2, 2});
  const Tensor value = Tensor::FromInt64("", {1}, {static_cast<int64_t>(-7)});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], -7);
  EXPECT_EQ(py[1], -7);
  EXPECT_EQ(py[2], -7);
  EXPECT_EQ(py[3], -7);
}

TEST(BackendKernelClass, ConstantOfShapeDefaultValueIsFloatZero) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor shape = Tensor::FromInt64("", {1}, {static_cast<int64_t>(4)});
  Tensor y = kernel(shape, Tensor());
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(py[i], 0.0f);
  }
}

TEST(BackendKernelClass, ConstantOfShapeEmptyShapeProducesScalar) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  // An empty 1-D ``shape`` input produces a scalar output.
  const Tensor shape = Tensor::FromInt64("", {0}, {});
  const Tensor value = Tensor::FromFloat("", {1}, {2.5f});
  Tensor y = kernel(shape, value);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{}));
  ASSERT_EQ(y.element_count(), 1);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 2.5f);
}

TEST(BackendKernelClass, ConstantOfShapeRejectsNonInt64Shape) {
  const KernelContext ctx{DefaultOpset(20)};
  ConstantOfShape kernel{ctx};
  const Tensor bad_shape = Tensor::FromInt32("", {2}, {2, 3});
  const Tensor value = Tensor::FromFloat("", {1}, {0.0f});
  EXPECT_THROW(kernel(bad_shape, value), std::invalid_argument);
}

} // namespace Test
