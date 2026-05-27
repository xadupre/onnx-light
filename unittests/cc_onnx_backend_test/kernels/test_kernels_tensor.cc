// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Cast;
using onnx_backend_test::kernel::Concat;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

TEST(BackendKernelClass, ConcatClassConcatenatesAxis0) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor y = concat_kernel({x0, x1}, /*axis=*/0);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 2}));
  ASSERT_EQ(y.element_count(), 8);
  const float *py = y.AsFloat();
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(py[i], static_cast<float>(i + 1));
  }
}

TEST(BackendKernelClass, ConcatClassConcatenatesNegativeAxis) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  Tensor y = concat_kernel({x0, x1}, /*axis=*/-1);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 5}));
  ASSERT_EQ(y.element_count(), 10);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, ConcatClassRejectsMismatchedShape) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  EXPECT_THROW((void)concat_kernel({x0, x1}, /*axis=*/1), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatClassRejectsScalar) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)concat_kernel({x}, /*axis=*/0), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatInPlaceWritesToPreallocatedOutput) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  Tensor y("out", TensorProto::DataType::FLOAT, {2, 5}, std::vector<uint8_t>(10 * sizeof(float)));
  concat_kernel({x0, x1}, /*axis=*/-1, y);
  EXPECT_EQ(y.name, "out");
  ASSERT_EQ(y.element_count(), 10);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, ConcatInPlaceRejectsMismatchedShape) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {3, 2},
                   std::vector<uint8_t>(6 * sizeof(float)));
  EXPECT_THROW(concat_kernel({x0, x1}, /*axis=*/0, bad_shape), std::invalid_argument);
}

TEST(BackendKernelClass, CastClassFloatToDouble) {
  Cast cast_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {3}, {-1.5f, 0.0f, 2.25f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::DOUBLE));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::DOUBLE));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3}));
  const double *py = y.AsDouble();
  EXPECT_DOUBLE_EQ(py[0], -1.5);
  EXPECT_DOUBLE_EQ(py[1], 0.0);
  EXPECT_DOUBLE_EQ(py[2], 2.25);
}

TEST(BackendKernelClass, CastClassFloatToInt32TruncatesTowardZero) {
  Cast cast_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::INT32));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], -1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 4);
}

TEST(BackendKernelClass, CastClassInt64ToFloat) {
  Cast cast_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromInt64("", {4}, {-3, 0, 7, 42});
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -3.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 7.0f);
  EXPECT_FLOAT_EQ(py[3], 42.0f);
}

TEST(BackendKernelClass, CastClassIdentityCopiesBytes) {
  Cast cast_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape, x.shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
}

TEST(BackendKernelClass, CastClassRejectsUnsupportedTo) {
  Cast cast_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {1}, {1.0f});
  // BOOL is not in the supported numeric set for the kernel today.
  EXPECT_THROW((void)cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::BOOL)),
               std::invalid_argument);
}

TEST(BackendKernelClass, CastInPlaceRejectsDtypeMismatch) {
  Cast cast_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  // Output dtype declared as FLOAT but the caller asks for INT32.
  Tensor wrong_out("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::INT32), wrong_out),
               std::invalid_argument);
}

} // namespace Test
