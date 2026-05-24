// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::And;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Or;
using onnx_backend_test::kernel::Xor;

namespace Test {

TEST(BackendKernelClass, AndClassMatchesReference) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, AndClassBroadcastsScalar) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {}, {1});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrClassMatchesReference) {
  Or or_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = or_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorClassMatchesReference) {
  Xor xor_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = xor_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, LogicalRejectsNonBoolTensors) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 0.0f});
  Tensor y("", TensorProto::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)and_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, AndInPlaceWritesToPreallocatedOutput) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
  and_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrInPlaceWritesToPreallocatedOutput) {
  Or or_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  or_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorInPlaceWritesToPreallocatedOutput) {
  Xor xor_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  xor_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

} // namespace Test
